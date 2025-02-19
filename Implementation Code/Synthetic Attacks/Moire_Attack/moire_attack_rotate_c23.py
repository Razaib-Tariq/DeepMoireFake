import numpy as np
import json
import os
import sys
import time
from mosaicing_demosaicing_v2 import *
import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchattacks
from torchattacks.attack import Attack
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

from image_transformer import ImageTransformer
from utils import *



# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class Morie_attack(Attack):
    r"""
    Distance Measure : L_inf bound on sensor noise
    Arguments:
        model (nn.Module): Victim model to attack.
        steps (int): number of steps. (DEFAULT: 50)
        batch_size (int): batch size
        scale_factor (int): zoom in the images on the LCD. （DEFAULT: 3）

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`, `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    """

    def __init__(self, model, img_h, img_w, noise_budget, scale_factor, steps=1, batch_size= 50, targeted=False):
        super(Morie_attack, self).__init__("Morie_attack", model)
        self.steps = steps
        self.targeted = targeted
        self.img_w = img_w
        self.img_h = img_h
        self.scale_factor = scale_factor
        self.noise_budget = noise_budget
        self.lr = noise_budget / steps
        noise = np.zeros([batch_size, self.img_h * self.scale_factor * 3, self.img_w * self.scale_factor * 3])
        self.noise = torch.from_numpy(noise).to(self.device)
        self.noise.requires_grad = True
        self.adv_loss = nn.CrossEntropyLoss()

    def simulate_LCD_display(self, input_img):
        """ Simulate the display of raw images on LCD screen
        Input:
            original images (tensor): batch x height x width x channel
        Output:
            LCD images (tensor): batch x (height x scale_factor)  x (width x scale_factor) x channel
        """
        input_img = np.asarray(input_img.cpu().detach())
        batch_size, h, w, c = input_img.shape

        simulate_imgs = np.zeros((batch_size, h * 3, w * 3, 3), dtype=np.float32)
        red = np.repeat(input_img[:, :, :, 0], 3, axis=1)
        green = np.repeat(input_img[:, :, :, 1], 3, axis=1)
        blue = np.repeat(input_img[:, :, :, 2], 3, axis=1)

        for y in range(w):
            simulate_imgs[:, :, y * 3, 0] = red[:, :, y]
            simulate_imgs[:, :, y * 3 + 1, 1] = green[:, :, y]
            simulate_imgs[:, :, y * 3 + 2, 2] = blue[:, :, y]
        simulate_imgs = torch.from_numpy(simulate_imgs).to(self.device)

        return simulate_imgs

    def demosaic_and_denoise(self, input_img):
        """ Apply demosaicing to the images
        Input:
            images (tensor): batch x (height x scale_factor) x (width x scale_factor)
        Output:
            demosaicing images (tensor): batch x (height x scale_factor) x (width x scale_factor) x channel
        """
        demosaicing_imgs = demosaicing_CFA_Bayer_bilinear(input_img)
        return demosaicing_imgs

    def simulate_CFA(self, input_img):
        """ Simulate the raw reading of the camera sensor using bayer CFA
        Input:
            images (tensor): batch x (height x scale_factor) x (width x scale_factor) x channel
        Output:
            mosaicing images (tensor): batch x (height x scale_factor) x (width x scale_factor)
        """
        mosaicing_imgs = mosaicing_CFA_Bayer(input_img)
        return mosaicing_imgs

    def random_rotation_3(self, org_images, lcd_images):
        """ Simulate the 3D rotation during the shooting
        Input:
            images (tensor): batch x height x width x channel
        Rotate angle:
            theta (int): (-20, 20)
            phi (int): (-20, 20)
            gamma (int): (-20, 20)
        Output:
            rotated original images (tensor): batch x height x width x channel
            rotated LCD images (tensor): batch x (height x scale_factor) x (width x scale_factor) x channel
        """
        rotate_images = np.zeros(org_images.size())
        rotate_lcd_images = np.zeros(lcd_images.size())

        for n, img in enumerate(org_images):
            Trans_org = ImageTransformer(img)
            theta, phi, gamma, rotate_img = Trans_org.rotate_along_axis(True)
            rotate_images[n, :] = rotate_img
            Trans_lcd = ImageTransformer(lcd_images[n])
            _, _, _, rotate_lcd_img = Trans_lcd.rotate_along_axis(False, theta, phi, gamma)
            rotate_lcd_images[n, :] = rotate_lcd_img

        rotate_images = torch.from_numpy(rotate_images).to(self.device)
        rotate_lcd_images = torch.from_numpy(rotate_lcd_images).to(self.device)

        return rotate_images, rotate_lcd_images

    def forward(self, org_imgs, org_labels, targeted_labels):
        r"""
        Overridden.
        """
        batch_size = org_imgs.size(0) 
        noise = np.zeros([batch_size, self.img_h * self.scale_factor * 3, self.img_w * self.scale_factor * 3])
        self.noise = torch.from_numpy(noise).to(self.device)
        self.noise.requires_grad = True
        
        org_images = org_imgs.clone().detach().to(self.device)
        org_labels = org_labels.clone().detach().to(self.device)

        # compute the original prediction
        temp_outputs = self.model(org_imgs.clone().detach().to(self.device))
        org_percentage = F.softmax(temp_outputs, dim=1) * 100
        del temp_outputs

        resize_before_lcd = F.interpolate(org_images, scale_factor=self.scale_factor, mode="bilinear")
        resize_before_lcd = resize_before_lcd.permute(0, 2, 3, 1)
        lcd_images = self.simulate_LCD_display(resize_before_lcd)

        temp_images = org_images.clone().detach().permute(0, 2, 3, 1)

        rotate_images, rotate_lcd_images = self.random_rotation_3(temp_images, lcd_images)
        rotate_images = rotate_images.to(self.device)
        rotate_lcd_images = rotate_lcd_images.to(self.device).detach()

        dim_images = adjust_contrast_and_brightness(rotate_images, beta=-60)

        ## compute the rotate prediction
        rotate_images = rotate_images.permute(0, 3, 1, 2)
        rotate_images = rotate_images.float()
        rotate_outputs = self.model(rotate_images)
        _, rotate_pre = torch.max(rotate_outputs.data, 1)
        rotate_percentage = F.softmax(rotate_outputs.clone().detach(), dim=1) * 100

        ## compute the dim prediction
        dim_images = dim_images.permute(0, 3, 1, 2)
        dim_images = dim_images.float()
        dim_outputs = self.model(dim_images)
        _, dim_pre = torch.max(dim_outputs.data, 1)
        dim_percentage = F.softmax(dim_outputs.clone().detach(), dim=1) * 100


        ## Deliver the MA
        for step in range(self.steps):
            print("Step: {}/{}".format(step, self.steps))

            cfa_img = self.simulate_CFA(rotate_lcd_images)
            cfa_img_noise = cfa_img + self.noise

            demosaic_img = self.demosaic_and_denoise(cfa_img_noise)
            demosaic_img = demosaic_img.permute(0, 3, 1, 2)

            ## Adjust the brightness
            brighter_img = adjust_contrast_and_brightness(demosaic_img, beta=60)

            at_images = F.interpolate(brighter_img, [299, 299], mode='bilinear')
            at_images = at_images.float()
            at_outputs = self.model(at_images)
            _, at_pre = torch.max(at_outputs.data, 1)

            at_percentage = F.softmax(at_outputs.clone().detach(), dim=1) * 100

            if self.targeted:
                adv_cost = self.adv_loss(at_outputs, targeted_labels.to(self.device))
            else:
                adv_cost = -1 * self.adv_loss(at_outputs, org_labels)

            total_cost = adv_cost
            print("Loss: ", total_cost, "Adv loss: ", adv_cost)

            total_cost.backward()
            gradient = self.noise.grad
            self.noise = self.noise.detach() - self.lr * torch.sign(gradient)
            self.noise = torch.clamp(self.noise, min=-self.noise_budget, max=self.noise_budget).detach()
            self.noise.requires_grad = True

        at_images = torch.clamp(at_images, min=0, max=255).detach()

        return at_images, rotate_images, dim_images, \
               at_pre, rotate_pre, dim_pre, \
               org_percentage, at_percentage, rotate_percentage, dim_percentage



class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):
        input = input / 255.0
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std
    

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Recursively iterate through subdirectories to collect image paths and assign labels
        for subdir, dirs, files in os.walk(root):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    # Add image path
                    image_path = os.path.join(subdir, file)
                    self.image_paths.append(image_path)

                    # Assign label based on parent folder names
                    if 'Real' in subdir:
                        self.labels.append(0)  # Label for 'Real' class
                    elif 'Fake' in subdir:
                        self.labels.append(1)  # Label for 'Fake' class
                    else:
                        # If subdirectory doesn't have 'Real' or 'Fake', assign label -1 or any other logic
                        self.labels.append(-1)  # Label -1 for undefined

        if len(self.image_paths) == 0:
            raise ValueError("No images found in the directory structure. Please check the dataset path.")
        
        if len(self.labels) == 0:
            raise ValueError("No labels could be assigned. Ensure folders contain 'Real' or 'Fake' in their path.")
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, image_path

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model = nn.Sequential(
        norm_layer,
        models.inception_v3(pretrained=True)
    ).to(device)
    model = nn.DataParallel(model).to(device)
    model = model.eval()

    dataset_path = '/media/'
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])
    
    normal_data = CustomDataset(root=dataset_path, transform=transform)
    normal_loader = torch.utils.data.DataLoader(normal_data, batch_size=10, shuffle=False,num_workers=4)

    # Moire_attack 
    noise_budget = 2
    scale_factor = 3
    steps = 1
    batch_size = 1

    attack = Morie_attack(model, img_h=299, img_w=299, noise_budget=noise_budget, scale_factor=scale_factor, steps=steps, batch_size=batch_size, targeted=False)

    Save_results = 'True'
    if Save_results == 'True':
        savedir = '/media/'
        create_dir(savedir)

    total = 0
    suc_cnt_at = 0
    suc_cnt_dim = 0
    suc_cnt_rotate = 0

    # For-loop to process images one by one, even if they're part of a batch
    for images, labels, image_paths in normal_loader:
        images = images * 255.0

        # Check if the directory has been processed and skip if all images already exist
        for i in range(len(images)):  # Iterate through each image in the batch
            image = images[i:i+1]  # Select a single image
            label = labels[i:i+1]  # Select the corresponding label
            original_path = image_paths[i]  # Select the image path

            # Get the relative path from the original image path
            relative_path = os.path.relpath(original_path, dataset_path)
            
            # Combine the save directory with the relative path to maintain folder structure
            result_path = os.path.join(savedir, relative_path)
            
            # Check if the processed image already exists
            if os.path.exists(result_path):
                print(f"Image already processed and saved at: {result_path}, skipping attack.")
                continue
            targeted_labels = torch.randint(0, 999, (1,), dtype=torch.int64)

            # Moire (if the image wasn't processed)
            at_images, rotate_images, dim_images, at_labels, rotate_labels, dim_labels, org_percentages, at_percentages, rotate_percentages, dim_percentages = attack(image, label, targeted_labels)

            if Save_results == 'True':
                img_at = at_images.detach().cpu().numpy()[0]  # Get the image at index 0 since it's single now
                img_at = np.moveaxis(img_at, 0, 2)  # (C, H, W) -> (H, W, C)
                img_at = Image.fromarray(img_at.astype(np.uint8))
                    
                # Create the directories if they don't exist
                os.makedirs(os.path.dirname(result_path), exist_ok=True)
                    
                # Save the image to the new path
                img_at.save(result_path)

                print(f"Saved image to: {result_path}")

            total += 1
            suc_cnt_rotate += (rotate_labels != label.to(device)).sum()
            suc_cnt_dim += (dim_labels != label.to(device)).sum()
            suc_cnt_at += (at_labels != label.to(device)).sum()

            Succ_total_rotate = suc_cnt_rotate / total
            Succ_total_dim = suc_cnt_dim / total
            Succ_total_at = suc_cnt_at / total

            print("Total rotate Success rate: ", Succ_total_rotate.item())
            print("Total dim Success rate: ", Succ_total_dim.item())
            print("Total attack Success rate: ", Succ_total_at.item())

        torch.cuda.empty_cache()

    print("Final Rotate Success rate: ", Succ_total_rotate.item())
    print("Final Dim Success rate: ", Succ_total_dim.item())
    print("Final Attack Success rate: ", Succ_total_at.item())

