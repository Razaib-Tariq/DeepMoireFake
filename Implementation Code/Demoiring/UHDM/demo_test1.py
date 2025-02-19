import datetime as dt
import logging
import lpips
import numpy as np
import torch
import argparse
import cv2
import torch.utils.data as data
import torchvision
import random
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os
from model.model import model_fn_decorator
from model.nets import my_model
from dataset.load_data import *
from tqdm import tqdm
from utils.loss_util import *
from utils.common import *
from config.config import args
from PIL import Image
from PIL import ImageFile
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def pixel_unshuffle(input, downscale_factor):
    '''
    input: batchSize * c * k*w * k*h
    kdownscale_factor: k

    batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
    '''
    c = input.shape[1]

    kernel = torch.zeros(size=[downscale_factor * downscale_factor * c,
                               1, downscale_factor, downscale_factor],
                         device=input.device)
    for y in range(downscale_factor):
        for x in range(downscale_factor):
            kernel[x + y * downscale_factor::downscale_factor*downscale_factor, 0, y, x] = 1
    return F.conv2d(input, kernel, stride=downscale_factor, groups=c)

class PixelUnshuffle(nn.Module):
    def __init__(self, downscale_factor):
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor
    def forward(self, input):
        '''
        input: batchSize * c * k*w * k*h
        kdownscale_factor: k

        batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
        '''

        return pixel_unshuffle(input, self.downscale_factor)

def demo_test(args, TestImgLoader, model, save_path, device):
    print("Starting demo_test...")
    tbar = tqdm(TestImgLoader)
    for batch_idx, data in enumerate(tbar):
        try:
            model.eval()
            test_model_fn(args, data, model, save_path, device)
            desc = 'Test demo'
            tbar.set_description(desc)
            tbar.update()
        except Exception as e:
            print(f"Error during testing batch {batch_idx}: {e}")

def init():
    print("Initializing...")
    args.TEST_RESULT_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, 'test_result')
    mkdir(args.TEST_RESULT_DIR)
    args.NETS_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, 'net_checkpoints')
    
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")

    if num_gpus > 0 and 0 <= args.GPU_ID < num_gpus:
        device = torch.device(f"cuda:{args.GPU_ID}")
        print(f"Using GPU: {torch.cuda.get_device_name(args.GPU_ID)}")
    else:
        print(f"Invalid GPU_ID: {args.GPU_ID}. Using CPU instead.")
        device = torch.device("cpu")

    random.seed(args.SEED)
    np.random.seed(args.SEED)
    torch.manual_seed(args.SEED)
    torch.cuda.manual_seed_all(args.SEED)
    if args.SEED == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    return device

def load_checkpoint(model):
    print("Loading checkpoint...")
    if args.LOAD_PATH:
        load_path = args.LOAD_PATH
        print('Loading checkpoint from:', load_path)
    else:
        print('Please specify a checkpoint path in the config file!!!')
        raise NotImplementedError

    try:
        if load_path.endswith('.pth'):
            if torch.cuda.is_available() and args.GPU_ID >= 0:
                model_state_dict = torch.load(load_path)
            else:
                model_state_dict = torch.load(load_path, map_location=torch.device('cpu'))
        else:
            if torch.cuda.is_available() and args.GPU_ID >= 0:
                model_state_dict = torch.load(load_path)['state_dict']
            else:
                model_state_dict = torch.load(load_path, map_location=torch.device('cpu'))['state_dict']
        model.load_state_dict(model_state_dict)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

def set_logging(log_path):
    print("Setting up logging...")
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)

    formatter = logging.Formatter('%(message)s')

    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

def test_model_fn(args, data, model, save_path, device):
    print("Running test_model_fn...")
    try:
        in_img = data['in_img'].to(device)
        number = data['number']
        b, c, h, w = in_img.size()

        w_pad = (math.ceil(w/32)*32 - w) // 2
        h_pad = (math.ceil(h/32)*32 - h) // 2
        w_odd_pad = w_pad
        h_odd_pad = h_pad
        if w % 2 == 1:
            w_odd_pad += 1
        if h % 2 == 1:
            h_odd_pad += 1

        in_img = img_pad(in_img, w_pad=w_pad, h_pad=h_pad, w_odd_pad=w_odd_pad, h_odd_pad=h_odd_pad)

        with torch.no_grad():
            out_1, out_2, out_3 = model(in_img)
            if h_pad != 0:
                out_1 = out_1[:, :, h_pad:-h_odd_pad, :]
            if w_pad != 0:
                out_1 = out_1[:, :, :, w_pad:-w_odd_pad]

        if args.SAVE_IMG:
            out_save = out_1.detach().cpu()
            img_save_path = os.path.join(save_path, f'test_{number[0]}.{args.SAVE_IMG}')
            # Create the directory if it does not exist
            os.makedirs(os.path.dirname(img_save_path), exist_ok=True)
            torchvision.utils.save_image(out_save, img_save_path)
    except Exception as e:
        print(f"Error in test_model_fn: {e}")

def create_demo_dataset(args, data_path):
    print(f"Creating demo dataset for path: {data_path}")
    def _list_image_files_recursively(data_dir):
        file_list = []
        for home, dirs, files in os.walk(data_dir):
            for filename in files:
                ext = filename.split(".")[-1]
                if ext.lower() in ["jpg", "jpeg", "png", "gif", "webp"]:
                    file_list.append(os.path.join(home, filename))
        file_list.sort()
        return file_list

    data_files = _list_image_files_recursively(data_dir=data_path)
    if len(data_files) == 0:
        print(f"No images found in {data_path}, skipping this folder.")
        return None

    dataset = demo_data_loader(data_files)

    data_loader = data.DataLoader(
        dataset, batch_size=args.BATCH_SIZE, shuffle=True, num_workers=args.WORKER, drop_last=True
    )

    return data_loader

class demo_data_loader(data.Dataset):

    def __init__(self, image_list):
        self.image_list = image_list

    def __getitem__(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        data = {}
        path_src = self.image_list[index]
        number = os.path.split(path_src)[-1]
        number = number.split('.')[0]

        img = Image.open(path_src).convert('RGB')
        img = default_toTensor(img)

        data['in_img'] = img
        data['number'] = number

        return data

    def __len__(self):
        return len(self.image_list)

def process_subfolder(subfolder_path, category, save_prefix, model, device):
    print(f"Processing subfolder: {subfolder_path}")
    save_path = os.path.join(save_prefix, category, os.path.basename(subfolder_path))
    
    # Check if the output folder already exists
    if os.path.exists(save_path):
        print(f"Output folder {save_path} already exists, skipping this subfolder.")
        return
    
    log_path = os.path.join(save_path, 'customer_result.log')
    mkdir(save_path)
    
    set_logging(log_path)
    logging.warning(dt.datetime.now())
    logging.warning(f'Load model from {args.LOAD_PATH}')
    logging.warning(f'Save image results to {save_path}')
    logging.warning(f'Save logger to {log_path}')
    
    args.BATCH_SIZE = 1
    
    DemoImgLoader = create_demo_dataset(args, data_path=subfolder_path)
    if DemoImgLoader is None:
        return
    
    demo_test(args, DemoImgLoader, model, save_path, device)

def main():
    device = init()

    # Load model once
    model = my_model(
        en_feature_num=args.EN_FEATURE_NUM,
        en_inter_num=args.EN_INTER_NUM,
        de_feature_num=args.DE_FEATURE_NUM,
        de_inter_num=args.DE_INTER_NUM,
        sam_number=args.SAM_NUMBER,
    ).to(device)
    
    load_checkpoint(model)

    demo_datasets = args.DEMO_DATASET
    save_prefix = args.SAVE_PREFIX
    
    for dataset_root in demo_datasets:
        print(f"Processing dataset root: {dataset_root}")
        category = 'Real' if 'Real' in dataset_root else 'Fake'
        for subfolder in os.listdir(dataset_root):
            subfolder_path = os.path.join(dataset_root, subfolder)
            if os.path.isdir(subfolder_path):
                print(f"Processing subfolder: {subfolder_path}")
                process_subfolder(subfolder_path, category, save_prefix, model, device)

if __name__ == '__main__':
    main()
