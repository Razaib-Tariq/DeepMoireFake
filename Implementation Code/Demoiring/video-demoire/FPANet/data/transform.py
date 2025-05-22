# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR), NAFNet(https://github.com/megvii-research/NAFNet) 
# Copyright 2018-2020 BasicSR Authors
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

import cv2
import random
import numpy as np
import torch

def imfrombytes(content, flag='color', float32=False):
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
        float32 (bool): Whether to change to float32., If True, will also norm
            to [0, 1]. Default: False.

    Returns:
        ndarray: Loaded image array.
    """
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {
        'color': cv2.IMREAD_COLOR,
        'grayscale': cv2.IMREAD_GRAYSCALE,
        'unchanged': cv2.IMREAD_UNCHANGED
    }
    if img_np is None:
        raise Exception('None .. !!!')
    img = cv2.imdecode(img_np, imread_flags[flag])
    if float32:
        img = img.astype(np.float32) / 255.
    return img

def paired_random_crop(imgs, gt_patch_size):
    """Paired random crop.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """
    if len(imgs) == 2:
        img_gts, img_lqs = imgs
        if not isinstance(img_gts, list):
            img_gts = [img_gts]
        if not isinstance(img_lqs, list):
            img_lqs = [img_lqs]
        img_next = None
        img_prev = None
    elif len(imgs) == 4:
        img_gts, img_lqs, img_next, img_prev = imgs
        if not isinstance(img_gts, list):
            img_gts = [img_gts]
        if not isinstance(img_lqs, list):
            img_lqs = [img_lqs]
        if not isinstance(img_next, list):
            img_next = [img_next]
        if not isinstance(img_prev, list):
            img_prev = [img_prev]

    h_lq, w_lq, _ = img_lqs[0].shape
    
    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - gt_patch_size)
    left = random.randint(0, w_lq - gt_patch_size)

    # crop lq patch
    img_lqs = [
        v[top:top + gt_patch_size, left:left + gt_patch_size, ...]
        for v in img_lqs
    ]
    img_gts = [
        v[top:top + gt_patch_size, left:left + gt_patch_size, ...]
        for v in img_gts
    ]
    
    if img_next is not None:
        img_next = [
            v[top:top + gt_patch_size, left:left + gt_patch_size, ...]
            for v in img_next
        ]
        img_prev = [
            v[top:top + gt_patch_size, left:left + gt_patch_size, ...]
            for v in img_prev
        ]
    
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
        
    if img_next is not None:
        if len(img_next) == 1:
            img_next = img_next[0]
        if len(img_prev) == 1:
            img_prev = img_prev[0]
            
        return [img_gts, img_lqs, img_next, img_prev]

    return [img_gts, img_lqs]

def padding(imgs, gt_size):
    if len(imgs) == 2:
        img_gts, img_lqs = imgs
    elif len(imgs) == 4:
        img_gts, img_lqs, img_next, img_prev = imgs

    h, w, _ = img_lqs.shape

    h_pad = max(0, gt_size - h)
    w_pad = max(0, gt_size - w)
    
    if h_pad == 0 and w_pad == 0 and len(imgs) == 4:
        return [img_gts, img_lqs, img_next, img_prev]
    elif h_pad == 0 and w_pad == 0 and len(imgs) == 2:
        return [img_gts, img_lqs]

    
    img_lqs = cv2.copyMakeBorder(img_lqs, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    img_gts = cv2.copyMakeBorder(img_gts, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    
    if img_next is not None:
        img_next = cv2.copyMakeBorder(img_next, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
        img_prev = cv2.copyMakeBorder(img_prev, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
        return [img_gts, img_lqs, img_next, img_prev]
    
    return [img_gts, img_lqs]


def augment(imgs, hflip=True, rotation=True, ):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """
    hflip = hflip and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
            if img.shape[2] == 6:
                img = img[:,:,[3,4,5,0,1,2]].copy() # swap left/right
        if rot90:
            img = img.transpose(1, 0, 2)
        return img


    if not isinstance(imgs, list):
        imgs = [imgs]
        
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]
        
    return imgs



