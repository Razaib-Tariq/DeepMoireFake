# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR), NAFNet(https://github.com/megvii-research/NAFNet) 
# Copyright 2018-2020 BasicSR Authors
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

import logging
import torch
from os import path as osp

from data import create_dataloader, create_dataset
from network.base_model import Model
from train import parse_options
from utils.utils import get_root_logger, get_time_str, make_exp_dirs, dict2str
from utils.logger_util import get_root_logger, get_env_info

def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='FPANet', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        if 'test' in phase:
            dataset_opt['phase'] = 'test'
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(
            test_set,
            dataset_opt,
            sampler=None,
            seed=opt['manual_seed'])
        logger.info(
            f"Number of test images in VDMoire: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = Model(opt)

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        model.validation(
            test_loader,
            current_iter=opt['name'],
            save_img=opt['val']['save_img'])


if __name__ == '__main__':
    main()
