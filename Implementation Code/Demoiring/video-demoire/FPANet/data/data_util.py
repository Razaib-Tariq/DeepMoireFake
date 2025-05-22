# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR), NAFNet(https://github.com/megvii-research/NAFNet) 
# Copyright 2018-2020 BasicSR Authors
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

import cv2
import numpy as np
import torch
from os import path as osp
from torch.nn import functional as F
import os

def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(
                        entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


def paired_paths_from_folder(folders, keys):
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    input_files = sorted([f for f in os.listdir(input_folder) if not f.startswith('.')])
    gt_files = sorted([f for f in os.listdir(gt_folder) if not f.startswith('.')])

    # Create maps from base names to full paths
    input_map = {osp.splitext(f)[0]: osp.join(input_folder, f) for f in input_files}
    gt_map = {osp.splitext(f)[0]: osp.join(gt_folder, f) for f in gt_files}

    # Union of keys to cover all files
    all_keys = sorted(set(input_map.keys()).union(set(gt_map.keys())))

    paired_paths = []
    for key in all_keys:
        entry = {}
        if key in input_map:
            entry[f'{input_key}_path'] = input_map[key]
        if key in gt_map:
            entry[f'{gt_key}_path'] = gt_map[key]
        paired_paths.append(entry)

    return paired_paths


