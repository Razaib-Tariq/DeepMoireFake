
import os
from .MAT.MAT import MAT
import pickle
import json
import torch
from torch import nn

class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()


        # with open('runs/%s/config.pkl'%name,'rb') as f:
        #     config = pickle.load(f)
        self.net =  MAT()
        state_dict = torch.load('/pretrained_checkpoints/mat/multi-attention/pretrained/ff_c23.pth')['state_dict']
        # print parameter names
        # for param_name in state_dict.keys():
        #     print(param_name)
        
        self.net.load_state_dict(state_dict, strict=True) # This one make sure you load a correct model

    def forward(self,x, return_ft=False):
        x = self.net.forward(x, return_ft=return_ft)
        return x
if __name__ == '__main__':
    model = Detector().cuda()
    print("Successfully loaded ")
    x = torch.randn(8, 3, 380, 380).cuda()
    y = model(x)
    print(y.shape)