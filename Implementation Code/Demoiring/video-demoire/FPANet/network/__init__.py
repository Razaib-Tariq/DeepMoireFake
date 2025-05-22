# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR), NAFNet(https://github.com/megvii-research/NAFNet) 
# Copyright 2018-2020 BasicSR Authors
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
import importlib
from os import path as osp

from utils.utils import get_root_logger
from network.FPANet import FPANet
from copy import deepcopy


def define_network(opt):
    """Create model.

    Args:
        opt (dict): Configuration. It constains:
            model_type (str): Model type.
    """
    net_type = opt.pop('type')
    model = FPANet(**opt)

    logger = get_root_logger()
    logger.info(f'FPANet is created.')
    return model
