# ------------------------------------------------------------------------
# Modified from EDVR (https://github.com/xinntao/EDVR/tree/master/LICENSE)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from network.DCNv2.dcn_v2 import DCN_sep as DCN
except ImportError:
    raise ImportError('Failed to import DCNv2 module.')

''' Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels. [From EDVR]
    '''

class PAM(nn.Module):
    def __init__(self, nf=64, groups=8, wn=None):
        super(PAM, self).__init__()
        wn = lambda x: torch.nn.utils.weight_norm(x)
        
        self.L3_offset_conv1 = wn(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True))  
        self.L3_offset_conv2 = wn(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))
        self.L3_offset_conv2_fuse = wn(nn.Conv2d(nf*2, nf, 3, 1, 1, bias=True))
        
        self.L3_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        self.L3_offset_conv3 = wn(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))
        self.dw = wn(nn.Conv2d(nf, nf//2, 3, 1, 1, bias=True))
         
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_fea_l, ref_fea_l, offset):
        if offset is None:
            f_offset = torch.cat([nbr_fea_l, ref_fea_l], dim=1)
            f_offset = self.lrelu(self.L3_offset_conv1(f_offset))
            f_offset = self.lrelu(self.L3_offset_conv2(f_offset))
            fea = self.lrelu(self.L3_dcnpack(nbr_fea_l, f_offset))
            f_offset = F.interpolate(f_offset, scale_factor=2, mode='bilinear', align_corners=False)
            f_offset = self.dw(f_offset)
        else:
            f_offset = torch.cat([nbr_fea_l, ref_fea_l], dim=1)
            f_offset = self.lrelu(self.L3_offset_conv1(f_offset))
            f_offset = self.lrelu(self.L3_offset_conv2_fuse(torch.cat([f_offset, offset*2], dim=1)))
            f_offset = self.lrelu(self.L3_offset_conv3(f_offset))
            fea = self.lrelu(self.L3_dcnpack(nbr_fea_l, f_offset))
            f_offset = F.interpolate(f_offset, scale_factor=2, mode='bilinear', align_corners=False)
            f_offset = self.dw(f_offset)
        return fea, f_offset


