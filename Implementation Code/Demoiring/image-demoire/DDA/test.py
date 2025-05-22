import argparse
import os
import random
import sys
import numpy as np
import torch
from torch import nn
# from torch.nn import MSELoss, L1Loss
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
import matplotlib
import matplotlib.pyplot as plt
from Util.util_collections import tensor2im, save_single_image, Time2Str, setup_logging, im_score
from dataset.dataset import Moire_dataset, AIMMoire_dataset,AIMMoire_dataset_test,FHDMI_dataset_test, CustomMoireDataset, CustomMoireDataset_mouth
from torchnet import meter
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
import torchvision
import logging
import numpy as np
import cv2
from glob import glob
import pdb
from torchvision import transforms
from skimage import color
from PIL import Image

mytrans = transforms.ToTensor()
image_train_path_moire = None
image_train_path_clean = None
image_train_path_demoire = None



def log(*args):
    args_list = map(str,args)
    tmp = ''.join(args_list)
    logging.info(tmp)

################

def test(args, model):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    global image_train_path_clean, image_train_path_demoire, image_train_path_moire

    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #args.save_prefix = args.save_prefix + '/' + args.name + '_Test_psnr_' + Time2Str()

    if not os.path.exists(args.save_prefix):
        os.makedirs(args.save_prefix)
    setup_logging(args.save_prefix + '/log.txt')

    log('torch devices = ', args.device)
    log('save_path = ', args.save_prefix)

    if args.dataset == 'aim':
        test_dataset = AIMMoire_dataset_test(args.testdata_path + '/test')
    elif args.dataset == 'fhdmi':
        test_dataset = FHDMI_dataset_test(args.testdata_path + '/test')
    elif args.dataset == 'custom':
        test_dataset = CustomMoireDataset(args.testdata_path, moire_type=args.moire_type)
    elif args.dataset == 'custom_mouth':
        test_dataset = CustomMoireDataset_mouth(args.testdata_path)    
    else:
        raise ValueError('no this choice:' + args.dataset)
    
    test_dataloader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False, num_workers=args.num_worker, drop_last=False)
    
    model = nn.DataParallel(model)
    checkpoint = torch.load(args.Test_pretrained_path)

    model.load_state_dict(checkpoint)
    
    model = model.to(torch.device('cuda'))
    model.eval()
    infer(model, test_dataloader, args)
    

def infer(model, loader, args):
    test_dataloader = loader

    for ii, (moire, moire_img_path) in tqdm(enumerate(test_dataloader)):
        if isinstance(moire_img_path, (list, tuple)):
            original_image_path = moire_img_path[0]
        else:
            original_image_path = moire_img_path

        relative_image_path = os.path.relpath(original_image_path, args.testdata_path)
        path_parts = relative_image_path.split(os.sep)
        
        if len(path_parts) > 1:
            real_or_fake = path_parts[0]
            subfolder_name = os.path.join(*path_parts[1:-1]) if len(path_parts[1:-1]) > 0 else ""
        else:
            real_or_fake = "unknown"
            subfolder_name = ""

        output_image_path = os.path.join(args.save_prefix, real_or_fake, subfolder_name, os.path.basename(original_image_path))
        output_image_path = os.path.splitext(output_image_path)[0] + '_processed.png'
        
        # 이미 해당 파일이 존재하면 건너뜁니다.
        if os.path.exists(output_image_path):
            print(f"Skipping {output_image_path} as it already exists.")
            continue

        with torch.no_grad():
            moire = moire.to(args.device)
            output = model(moire)

            try:
                original_image = Image.open(original_image_path)
                original_size = original_image.size  # (width, height)
            except FileNotFoundError:
                print(f"File not found: {original_image_path}")
                continue
            except AttributeError:
                print(f"Unexpected label format: {moire_img_path}")
                continue

            output_image_np = tensor2im(output)

            output_image_pil = Image.fromarray(output_image_np)
            output_image_resized = output_image_pil.resize(original_size, Image.BICUBIC)

            output_dir = os.path.dirname(output_image_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            output_image_resized.save(output_image_path)

    print('Inference complete.')






def save_single_image(image, path):
    image = Image.fromarray(image)
    image.save(path)
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--testdata_path', type=str, required=True, help='Path to the test dataset')
    parser.add_argument('--Test_pretrained_path', type=str, required=True, help='Path to the pre-trained model')
    parser.add_argument('--save_prefix', type=str, required=True, help='Directory to save output images')
    parser.add_argument('--batchsize', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--num_worker', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for inference')
    parser.add_argument('--dataset', type=str, default='custom', help='Dataset type')
    parser.add_argument('--moire_type', type=str, default='real', help='Type of moire images (real or fake)')
    args = parser.parse_args()

    from Net.MBCNN import MBCNN  # assuming the model is defined in Net.MBCNN

    net = MBCNN(64)
    test(args, net)
        
def val(model, loader, args):
    global image_train_path_clean, image_train_path_demoire, image_train_path_moire

    psnr_output_meter = meter.AverageValueMeter()
    psnr_input_meter = meter.AverageValueMeter()
    ssim_output_meter = meter.AverageValueMeter()
    ciede_output_meter = meter.AverageValueMeter()
    test_dataloader = loader

    if args.dataset == 'fhdmi':
        M, N = [512, 640]
    else:
        M, N = [512, 512]
    H = np.zeros((M, N), dtype=np.float32)
    D0 = 5
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u - M / 2)**2 + (v - N / 2)**2)
            H[u, v] = np.exp(-D**2 / (2 * D0 * D0))

    HPF = 1 - H

    for ii, (moire, clear, label) in tqdm(enumerate(test_dataloader)):
        with torch.no_grad():
            moire = moire.to(args.device)
            moire_list, width_index, h_space, w_space, crop_sz_h, crop_sz_w = crop(moire, args, HPF)
            output_list = []
            for i, moire_patch in enumerate(moire_list):
                model.apply(lambda m: setattr(m, 'width_mult', args.width_list[width_index[i]]))
                moire_patch = torch.unsqueeze(mytrans(moire_patch), 0).to(args.device)
                if args.arch == 'MBCNN':
                    _, _, output_1 = model(moire_patch)
                else:
                    output_1 = model(moire_patch)
                output_list.append(tensor2im(torch.squeeze(output_1)))
            output = combine(output_list, h_space, w_space, crop_sz_h, crop_sz_w, args)
        
        if args.dataset == 'aim':
            clear = clear[2].to(args.device)
        clear = tensor2im(torch.squeeze(clear))
        moire = tensor2im(moire[0])
        psnr_output = peak_signal_noise_ratio(output, clear)
        ciede_output = ciede2000(output, clear)

        psnr_output_meter.add(psnr_output)
        ciede_output_meter.add(ciede_output)
        ssim_output = ssim(output, clear, multichannel=True)
        ssim_output_meter.add(ssim_output)
        psnr_input = peak_signal_noise_ratio(moire, clear)
        psnr_input_meter.add(psnr_input)

        # 모아레 패턴이 제거된 이미지를 저장하는 부분 추가
        output_image_path = os.path.join(args.save_prefix, f'output_{ii}.png')
        save_single_image(output, output_image_path)

    log('Test dataset_PSNR = ', psnr_output_meter.value()[0])
    log('Test dataset_CIEDE = ', ciede_output_meter.value()[0])
    log('Test dataset_SSIM = ', ssim_output_meter.value()[0])

    return psnr_output_meter.value()[0]

        
        
###################

def crop(img, args, HPF):
    img = tensor2im(torch.squeeze(img))
    n_channels = len(img.shape)
    if n_channels == 2:
        h, w = img.shape
    elif n_channels == 3:
        h, w, c = img.shape
    else:
        raise ValueError('Wrong image shape - {}'.format(n_channels))
    if args.dataset == 'fhdmi':
        crop_sz_h = 512
        crop_sz_w = 640
        h_space = [0, 512]
        w_space = [0, 640, 1280]
    else:
        crop_sz_h = 512
        crop_sz_w = 512
        h_space = [0, 512]
        w_space = [0, 512]
        
    index = 0
    moire_list=[]   
    score_list=[]
    for x in h_space:
        for y in w_space:
            if n_channels == 2:
                crop_img = img[x:x + crop_sz_h, y:y + crop_sz_w]
            else:
                crop_img = img[x:x + crop_sz_h, y:y + crop_sz_w,:]
            moire_list.append(crop_img)
            score_list.append(im_score(np.uint8(crop_img*255), HPF))

    index = np.argsort(score_list)
    if args.dataset == 'fhdmi':
        width_index = np.array([0, 0, 0, 0, 0, 0])
        width_index[index[0:2]] = 2
        width_index[index[2:4]] = 1
        width_index[index[4::]] = 0
    else:
        width_index = np.array([0, 0, 0, 0])
        width_index[index[0]] = 2
        width_index[index[1]] = 1
        width_index[index[1:3]] = 0
    return moire_list,width_index, h_space, w_space, crop_sz_h, crop_sz_w

def combine(output_list, h_space, w_space, crop_sz_h, crop_sz_w, args):
    index=0
    if args.dataset == 'fhdmi':
        clear_img = np.zeros((1024, 1920, 3), 'float32')
    else:
        clear_img = np.zeros((1024, 1024, 3), 'float32')
    index = 0 
    for x in h_space:
        for y in w_space:
            #clear = cv2.resize(output_list[index],(643,516))
            clear_img[x:x + crop_sz_h, y:y + crop_sz_w, :] += output_list[index]
            index += 1

    clear_img=clear_img.astype(np.uint8)
    
    return clear_img

def ciede2000(out, gt):
    from skimage import color
    deltaE = np.absolute(color.deltaE_ciede2000(color.rgb2lab(gt), color.rgb2lab(out)))
    return np.mean(deltaE)