# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR), NAFNet(https://github.com/megvii-research/NAFNet) 
# Copyright 2018-2020 BasicSR Authors
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

from torch.utils import data as data
from torchvision.transforms.functional import normalize

from data.data_util import (paired_paths_from_folder)
from data.transform import augment, paired_random_crop, imfrombytes, padding
from utils.utils import FileClient, img2tensor
import os


class VDMDataset(data.Dataset):
    def __init__(self, opt):
        super(VDMDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.mode = opt['mode']
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']

        # self.paths = paired_paths_from_folder(
        #     [self.lq_folder, self.gt_folder], ['lq', 'gt'])

        # # Dynamically determine max frame index
        # self.max_frame_index = max(
        #     int(os.path.basename(p['lq_path']).split('.')[0].split('_')[1])
        #     for p in self.paths
        # )
        self.paths = paired_paths_from_folder(
            [self.lq_folder, self.gt_folder], ['lq', 'gt'])

        # Handle unequal folder sizes safely
        self.max_frame_index = max(
            int(os.path.basename(p[k]).split('.')[0].split('_')[1])
            for p in self.paths
            for k in ('lq_path', 'gt_path')  # check whichever exists
            if k in p                        # skip missing keys
)
    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        paths_entry = self.paths[index]
        if 'lq_path' not in paths_entry or 'gt_path' not in paths_entry:
            raise IndexError(f"Missing required lq/gt path at index {index}")

        lq_path = paths_entry['lq_path']
        gt_path = paths_entry['gt_path']

        filename = os.path.basename(lq_path)
        frame = int(filename.split('.')[0].split('_')[1])
        file_type = filename.split('.')[-1]
        video_folder = os.path.dirname(lq_path)

        try:
            img_bytes = self.file_client.get(lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)
        except Exception:
            raise Exception(f"lq path {lq_path} not working")

        try:
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
        except Exception:
            raise Exception(f"gt path {gt_path} not working")

        # Auxiliary frames
        if self.mode == 'multi':
            frame = max(1, min(frame, self.max_frame_index - 1))  # safe bounds
            frame_next = f"frame_{frame + 1:05d}.{file_type}"
            frame_prev = f"frame_{frame - 1:05d}.{file_type}"
            next_path = os.path.join(video_folder, frame_next)
            prev_path = os.path.join(video_folder, frame_prev)

            try:
                img_bytes = self.file_client.get(next_path, 'next')
                img_next = imfrombytes(img_bytes, float32=True)
            except Exception:
                raise Exception(f"next path {next_path} not working")

            try:
                img_bytes = self.file_client.get(prev_path, 'prev')
                img_prev = imfrombytes(img_bytes, float32=True)
            except Exception:
                raise Exception(f"prev path {prev_path} not working")

            imgs = [img_gt, img_lq, img_next, img_prev]

            if self.opt['phase'] == 'train':
                gt_size = self.opt['gt_size']
                imgs = padding(imgs, gt_size)
                imgs = paired_random_crop(imgs, gt_size)
                imgs = augment(imgs, self.opt['use_flip'], self.opt['use_rot'])

            img_gt, img_lq, img_next, img_prev = img2tensor(imgs, bgr2rgb=True, float32=True)

            if self.mean is not None or self.std is not None:
                normalize(img_lq, self.mean, self.std, inplace=True)
                normalize(img_gt, self.mean, self.std, inplace=True)
                normalize(img_next, self.mean, self.std, inplace=True)
                normalize(img_prev, self.mean, self.std, inplace=True)

            return {
                'lq': img_lq,
                'gt': img_gt,
                'next': img_next,
                'prev': img_prev,
                'lq_path': lq_path,
                'gt_path': gt_path,
                'next_path': next_path,
                'prev_path': prev_path
            }


    def __len__(self):
        return len(self.paths)
