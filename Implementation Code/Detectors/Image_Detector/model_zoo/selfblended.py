
from torch import nn
import torch
from efficientnet_pytorch import EfficientNet

class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        self.net = EfficientNet.from_pretrained("efficientnet-b4",advprop=True,num_classes=2)
        pre_trained = "/pretrained_checkpoints/self_blended/FFraw.tar"
        cnn_sd=torch.load(pre_trained)["model"]
        for key in list(cnn_sd.keys()):
            cnn_sd[key.replace('net.', '')] = cnn_sd.pop(key)
        self.net.load_state_dict(cnn_sd, strict=True)

    def forward(self,x):
        x = self.net.forward(x)
        return x