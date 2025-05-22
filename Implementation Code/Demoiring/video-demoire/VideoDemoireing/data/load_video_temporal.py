import numpy as np
import torch
import argparse
import cv2, os, glob
import re
import torch.utils.data as data
import torchvision.transforms as transforms
import random
from PIL import Image
from PIL import ImageFile



class data_loader(data.Dataset):

    def __init__(self, args, image_list, mode='train'):
        self.image_list = image_list
        self.args = args
        self.mode = mode
        self.loader = args.LOADER
        self.frames_each_video = args.frames_each_video
        file_type = image_list[0].split('.')[-1]
        self.file_type = '.%s' % file_type

    def __getitem__(self, index):
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        data = {}

        image_in_gt = self.image_list[index]
        filename = os.path.basename(image_in_gt)

        number_strs = re.findall(r'\d+', filename)
        number = int(number_strs[0]) if number_strs else index

        image_in = filename
        base_path = self.args.TEST_DATASET 

        assert self.args.NUM_AUX_FRAMES > 0

        if self.mode == 'val':
            path_tar = os.path.join(base_path, filename)
            path_src = os.path.join(base_path, filename)

            path_src_auxs = []
            for i in range(1, self.args.NUM_AUX_FRAMES + 1):
                if i % 2 == 0:
                    number_tmp = number + i // 2 * self.args.FRAME_INTERVAL
                else:
                    number_tmp = number - (i + 1) // 2 * self.args.FRAME_INTERVAL
                    if number_tmp < 0:
                        number_tmp = number
                aux_filename = f"frame{number_tmp}_face0{self.file_type}"
                path_aux = os.path.join(base_path, aux_filename)
                if not os.path.isfile(path_aux):
                    path_aux = path_src
                if self.args.MODE == 'single':
                    path_aux = path_src
                path_src_auxs.append(path_aux)

            labels = default_loader([path_tar])
            moire_imgs = default_loader([path_src])
            moire_imgs_aux = default_loader(path_src_auxs)

        elif self.mode == 'test':
            path_src = os.path.join(base_path, filename)

            path_src_auxs = []
            for i in range(1, self.args.NUM_AUX_FRAMES + 1):
                if i % 2 == 0:
                    number_tmp = number + i // 2 * self.args.FRAME_INTERVAL
                else:
                    number_tmp = number - (i + 1) // 2 * self.args.FRAME_INTERVAL
                    if number_tmp < 0:
                        number_tmp = number
                aux_filename = f"frame{number_tmp}_face0{self.file_type}"
                path_aux = os.path.join(base_path, aux_filename)
                if not os.path.isfile(path_aux):
                    path_aux = path_src
                if self.args.MODE == 'single':
                    path_aux = path_src
                path_src_auxs.append(path_aux)

            moire_imgs = default_loader([path_src])
            moire_imgs_aux = default_loader(path_src_auxs)

        else:
            print('Unrecognized mode! Please select either "val" or "test"')
            raise NotImplementedError

        data['number'] = filename
        data['in_img'] = moire_imgs
        data['in_img_aux'] = moire_imgs_aux
        if self.mode != 'test':
            data['label'] = labels

        return data



    def __len__(self):
        return len(self.image_list)


def default_loader(path_set):
    imgs = []
    for path in path_set:
        img = Image.open(path).convert('RGB')
        img = default_toTensor(img)
        imgs.append(img)

    return imgs


def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def crop_loader(crop_size_x, crop_size_y, x, y, path_set, pad_size=100, pad=False):
    imgs = []
    for path in path_set:
        img = Image.open(path).convert('RGB')
        if pad:
            img = add_margin(img, pad_size, pad_size, pad_size, pad_size, (123, 117, 104))
        img = img.crop((x, y, x + crop_size_x, y + crop_size_y))
        img = default_toTensor(img)
        imgs.append(img)

    return imgs


def crop_loader_mask(crop_size_x, crop_size_y, x, y, path_set):
    imgs = []
    for path in path_set:
        img = Image.open(path).convert('RGB')
        img = img.crop((x, y, x + crop_size_x, y + crop_size_y))
        img = 1 - default_toTensor(img)
        imgs.append(img)

    return imgs
    
    
def crop_loader_flow(crop_size_x, crop_size_y, x, y, path_set):
    imgs = []
    for path in path_set:
        img = np.load(path)['flow']
        img = img[y:(y+crop_size_y), x:(x+crop_size_x), :]
        img = default_toTensor(img)
        imgs.append(img)
        
    return imgs


def resize_loader(resize_size_h, resize_size_w, path_set):
    imgs = []
    for path in path_set:
        img = Image.open(path).convert('RGB')
        img = img.resize((resize_size_w, resize_size_h), Image.BICUBIC)
        img = default_toTensor(img)
        imgs.append(img)

    return imgs


def default_toTensor(img):
    t_list = [transforms.ToTensor()]
    composed_transform = transforms.Compose(t_list)

    return composed_transform(img)
