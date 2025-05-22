import torch
import torch.nn as nn
import torch.nn.functional as F

from network.network_utils import LayerNorm2d
# from basicsr.models.archs.dconv_pytorch import *
from network.PAM import *


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


### Building frequency selection module(FSM) ###
class FSM(nn.Module):
    def __init__(self, in_channels, reduction=8, bias=False):
        super().__init__()
        self.mag = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.pha = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        
        d = max(int(in_channels/reduction), 4)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.conv_du = nn.Sequential(
            nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), 
            nn.LeakyReLU(0.2)
        )

        self.freq_fusion = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, stride=1,
                      groups=1, bias=True),
        )

        self.fc = nn.ModuleList([])
        for i in range(2):
            self.fc.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1,bias=bias))
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp):
        x = inp
        ### Frequency branch ###
        _, _, H, W = x.shape
        
        fre = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(fre)
        pha = torch.angle(fre)
        mag = self.mag(mag)
        pha = self.pha(pha)

        in_feat = [mag, pha]
        
        freq_feats = torch.cat(in_feat, dim=1)
        feats_U = self.freq_fusion(freq_feats)
        
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fc]
        mag_softmax = self.softmax(attention_vectors[0])
        pha_softmax = self.softmax(attention_vectors[1])

        feats_mag = in_feat[0] * mag_softmax
        feats_pha = in_feat[1] * pha_softmax
        
        real = feats_mag * torch.cos(feats_pha)
        imag = feats_mag * torch.sin(feats_pha)
        freq_out = torch.complex(real, imag)
        out = torch.fft.irfft2(freq_out, s=(H, W), norm='backward')
        
        return out


### UHDM ###
class CSAF(nn.Module):
    def __init__(self, in_chnls, ratio=4):
        super(CSAF, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress1 = nn.Conv2d(in_chnls, in_chnls // ratio, 1, 1, 0)
        self.compress2 = nn.Conv2d(in_chnls // ratio, in_chnls // ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls // ratio, in_chnls, 1, 1, 0)

    def forward(self, x0, x2, x4):
        out0 = self.squeeze(x0)
        out2 = self.squeeze(x2)
        out4 = self.squeeze(x4)
        out = torch.cat([out0, out2, out4], dim=1)
        out = self.compress1(out)
        out = F.relu(out)
        out = self.compress2(out)
        out = F.relu(out)
        out = self.excitation(out)
        out = F.sigmoid(out)
        w0, w2, w4 = torch.chunk(out, 3, dim=1)
        x = x0 * w0 + x2 * w2 + x4 * w4

        return x
    
### Building Simple Feature Extraction Block(SFEB) ###
class SFEB(nn.Module):
    def __init__(self, c):
        super().__init__()
        in_channel = c
        out_channel = c * 2
        
        self.norm1 = LayerNorm2d(c)
        
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv1_2 = nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv1_4 = nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
         
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1, groups=out_channel,
                               bias=True)
        self.conv2_2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1, groups=out_channel,
                               bias=True)
        self.conv2_4 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1, groups=out_channel,
                               bias=True)
        
        # Simple Gate
        self.sg = SimpleGate()
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=out_channel // 2, out_channels=out_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.sca_2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=out_channel // 2, out_channels=out_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.sca_4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=out_channel // 2, out_channels=out_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        
        self.conv3 = nn.Conv2d(out_channel // 2, in_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv3_2 = nn.Conv2d(out_channel // 2, in_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv3_4 = nn.Conv2d(out_channel // 2, in_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
       
        
    def forward(self, inp):
        x, x_2, x_4 = inp
        
        x = self.conv2(self.conv1(self.norm1(x)))
        x_2 = self.conv2_2(self.conv1_2(self.norm1(x_2)))
        x_4 = self.conv2_4(self.conv1_4(self.norm1(x_4)))
       
        x = self.sg(x)
        x_2 = self.sg(x_2)
        x_4 = self.sg(x_4)
        
        x = x * self.sca(x)
        x_2 = x_2 * self.sca_2(x_2)
        x_4 = x_4 * self.sca_2(x_4)
        
        x = self.conv3(x)
        x_2 = self.conv3_2(x_2)
        x_4 = self.conv3_4(x_4)
        
        _, _, H_x, H_y = inp[0].shape
        x_2 = F.interpolate(x_2, size=(H_x, H_y), mode='bilinear')
        x_4 = F.interpolate(x_4, size=(H_x, H_y), mode='bilinear')
        return x, x_2, x_4
    
### Buliding Frequency Spatial Fusion Block(FSFBlock) ###
class FSFBlock(nn.Module):
    def __init__(self, c, drop_out_rate=0.):
        super().__init__()
        self.FSM = FSM(c)
        self.SFEB = SFEB(c)
        self.CSAF = CSAF(3*c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        
        fsm_out= self.FSM(x)
        fsm_out = self.dropout1(fsm_out)
        
        fsm_out = inp + fsm_out * self.beta
        x = fsm_out
        x_2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        x_4 = F.interpolate(x, scale_factor=0.25, mode='bilinear')
        inp = [x, x_2, x_4]
        
        x, x_2, x_4 = self.SFEB(inp)

        x = self.CSAF(x, x_2, x_4)
        x = self.dropout2(x)
        
        return fsm_out + x * self.gamma


class FPANet(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.scale_dec_2 = nn.Sequential(nn.Conv2d(width * 2 , width, 1, bias=False), FSFBlock(width))
        self.scale_dec_4 = nn.Sequential(nn.Conv2d(width * 4 , width, 1, bias=False), FSFBlock(width))
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.PAM = nn.ModuleList()
        self.FUSION = nn.ModuleList()

        chan = width
        
        for idx, num in enumerate(enc_blk_nums):
            self.encoders.append(
                nn.Sequential(
                    *[FSFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[FSFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            
            chan = chan // 2
            
            self.PAM.append(PAM(chan))
            
            self.decoders.append(
                nn.Sequential(
                    *[FSFBlock(chan) for _ in range(num)]
                )
            )
            self.FUSION.append(
                nn.Sequential(
                    nn.Conv2d(chan*3, chan, 3, 1, 1),
                    nn.Conv2d(chan, chan, 1)
                )
            )
            

    def forward(self, data):
        if len(data) != 1:
            inp = data[0]
            prev_inp = data[1]
            next_inp = data[2]
        else:
            inp = data[0]
            prev_inp = data[0]
            next_inp = data[0]
            
        B, C, H, W = inp.shape
        inp_2 = F.interpolate(inp, scale_factor=0.5)
        inp_4 = F.interpolate(inp_2, scale_factor=0.5)
        
        x = self.intro(inp)
        prev_x = self.intro(prev_inp)
        next_x = self.intro(next_inp)

        encs = []
        prev_encs = []
        next_encs = []

        for encoder, down in zip(self.encoders, self.downs):
            prev_x = encoder(prev_x)
            next_x = encoder(next_x)
            x = encoder(x)
            encs.append(x)
            prev_encs.append(prev_x)
            next_encs.append(next_x)
            prev_x = down(prev_x)
            next_x = down(next_x)
            x = down(x)

        x = self.middle_blks(x)
        
        p_align_offset = None
        n_align_offset = None
        for decoder, up, enc_skip, prev_enc_skip, next_enc_skip, pam, fusion in zip(self.decoders, self.ups, encs[::-1], prev_encs[::-1], next_encs[::-1], self.PAM, self.FUSION):
            x = up(x)
            align_prev_x, p_align_offset = pam(enc_skip, prev_enc_skip, p_align_offset)
            align_next_x, n_align_offset = pam(enc_skip, next_enc_skip, n_align_offset)
            
            x = fusion(torch.cat([x, align_prev_x, align_next_x], dim=1))
            x = x + enc_skip
            x = decoder(x)
            if x.shape[2] == H // 2 and x.shape[3] == W // 2:
                out_2 = self.scale_dec_2(x)
                
            if x.shape[2] == H // 4 and x.shape[3] == W // 4:
                out_4 = self.scale_dec_4(x)

        x = self.ending(x)
        x_2 = self.ending(out_2)
        x_4 = self.ending(out_4)
        
        x = x + inp
        x_2 = x_2 + inp_2
        x_4 = x_4 + inp_4
        return_out = [x, x_2, x_4]
        
        return return_out
        


if __name__ == '__main__':
    img_channel = 3
    width = 48

    enc_blks = [2, 2, 4]
    middle_blk_num = 12
    dec_blks = [2, 2, 4]
    
    net = FPANet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks).cuda()


    inp_shape = (1, 3, 384, 384)

    inp = torch.randn(inp_shape).cuda()
    prev_inp = torch.randn(inp_shape).cuda()
    data = [inp, prev_inp]
    # inp = torch.randn(inp_shape)
    # inp = torch.randn(inp_shape)

    out = net(data)
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)
