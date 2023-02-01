
# from tokenize import group
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

from paddle2torch import *
from diffaug import DiffMask

import numpy as np
import math
# from DiffAugment import DiffAugment, DiffMask
# from gauss_filter import get_gaussian_filter
device = 'cuda'



class SeWeight(nn.Module):
    def __init__(self, in_size, out_size, reduction=4):
        super(SeWeight, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(in_size // reduction),
            nn.LayerNorm([in_size // reduction, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, out_size, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(in_size),
            nn.LayerNorm([out_size, 1, 1]),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.se(x)

class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter([1], dtype='float32')*torch.zeros([1])
        # self.weight = nn.Parameter(torch.ones(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            # noise = image.new_empty(batch, 1, height, width).normal_()
            noise = torch.randn((batch, 1, height, width))
        
        # print("ssssss:", self.weight)
        # return image + 0.1*self.weight * noise
        # print("ssssss:", var)
        return image + self.weight * noise

class SAConv2dLayer(nn.Module):
    def __init__(self, inc, outc, kernel_size, stride=1, padding=None, bias=True, activation=True, weight_norm = True, attension=False, dropout=False, groups=1, inorm=True, print=False, init="kaiming"):
        super(SAConv2dLayer, self).__init__()
        self.padding = kernel_size//2 if padding == None else padding
        if (weight_norm):
            self.conv = nn.utils.weight_norm(nn.Conv2d(inc, outc, kernel_size, stride=stride, padding=self.padding, groups = groups),dim=None)
            # self.conv = nn.utils.spectral_norm(nn.Conv2d(inc, outc, kernel_size, stride=stride, padding=self.padding, groups = groups))
        else:
            self.conv = nn.Conv2d(inc, outc, kernel_size, stride=stride, padding=self.padding, groups = groups)

        if init == "kaiming":
            # nn.init.kaiming_normal_(self.conv.weight, nonlinearity='leaky_relu')
            nn.init.orthogonal_(self.conv.weight)
            # num_input = self.conv.weight.size(-1)
            # nn.init.uniform_(self.conv.weight, -np.sqrt(6 / num_input), np.sqrt(6 / num_input))
        if init == "sin":
            num_input = self.conv.weight.size(-1)
            nn.init.uniform_(self.conv.weight, -np.sqrt(6 / num_input), np.sqrt(6 / num_input))
            # nn.init.uniform_(self.conv.weight, -1 / num_input, 1 / num_input)  #first layer

        self.sew = None
        if attension:
            self.sew = SeWeight(inc, outc)

        self.activation = None
        if activation:
            self.activation = nn.LeakyReLU(0.2)
            # self.activation = nn.ELU()

        self.dropout = None
        if dropout:
            self.dropout = nn.Dropout(p=0.1)

        self.inorm = inorm

        self.inject_noise = NoiseInjection()
        self.norm = nn.InstanceNorm2d(outc)

        self.gauss_std = 1.5
        # self.gauss_std = 0.0002
        # self.gauss = get_gaussian_filter(kernel_size=3, sigma=self.gauss_std, channels=outc)
        self.outc = outc

        self.mean_out = 0

        self.print = print
        self.nas_reg = 0
        # self.nas = nn.Parameter(0.5*torch.ones([1,outc,1,1])+0.01*torch.randn([1,outc,1,1]))
        # self.nas = nn.Parameter(torch.ones([1,outc,1,1]))
        # self.nas = nn.Parameter(0.5*torch.ones([1,outc,1,1])+0.001*torch.randn([1,outc,1,1]))

    # def get_new_kernels(self):
    #     self.gauss_std *= 0.925
    #     self.gauss = get_gaussian_filter(kernel_size=3, sigma=self.gauss_std, channels=self.outc).to(device)

    def forward(self, x):
        # x = self.inject_noise(x)
        out = self.conv(x)
        if self.sew != None:
            out=out*self.sew(x)
        # if activation is sigmoid, norm should before act; if activation is relu, norm should after act.
        # out = self.norm(out)
        # out = self.gauss(out)
        if self.activation != None:
            out = self.activation(out)
            # out = self.norm(out)
        if self.dropout != None:
            out = self.dropout(out)
        # self.mean_out = out.abs().mean()
        # ktkt = 0.5+0.5*torch.sin((self.nas-0.25)*2*np.pi*100.)
        # out = self.nas*out
        # out = ktkt*out
        # self.nas_reg = (0.51*torch.sin(self.nas*np.pi)).abs().mean() + self.nas.abs().mean()
        # self.nas_reg = (0.1*(1.+torch.sin((self.nas-0.25)*2*np.pi))).mean() + self.nas.abs().mean()
        # self.nas_reg = (0.01*(1.1+torch.sin((self.nas-0.5)*np.pi))).mean() - 0.01*self.nas.view(1, -1).var()
        # self.nas_reg =(0.01*(1.+torch.sin((ktkt-0.25)*2*np.pi))).mean()
        # if self.training == False and self.print:
        #     print("ssss:", self.nas.view(1, -1))
        #     # print("ssss:", ktkt.view(1, -1))
        #     print("ssss out_c:", self.outc)
        #     print("ssss actul_c:", F.relu(self.nas.abs()-0.1).ceil().sum())
        return out

class LinearLayer(nn.Module):
    def __init__(self, inc, outc, activation=True, weight_scale=False, weight_norm=True, dropout=False):
        super().__init__()
        self.w_lr = 1.0 / math.sqrt(inc) if weight_scale else None

        usebias = True
        self.bias = None
        if weight_scale:
            usebias = False
            # self.bias = nn.Parameter( torch.randn(outc) )

        if weight_norm:
            # self.linear = nn.utils.spectral_norm(nn.Linear(inc, outc, bias=usebias))
            self.linear = nn.utils.weight_norm(nn.Linear(inc, outc), dim=None)
        else:
            self.linear = nn.Linear(inc, outc)
        # nn.init.kaiming_normal_(self.linear.weight, nonlinearity="leaky_relu")
        nn.init.orthogonal_(self.linear.weight)

        self.activation = None
        if activation:
            self.activation = nn.LeakyReLU(0.2)

        self.dropout = None
        if dropout:
            self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        out = self.linear(x)
        if self.bias != None:
            out = out * self.w_lr
            bias = self.bias.repeat(x.shape[0], 1)
            out = out + bias
        if self.activation != None:
            out = self.activation(out)
        if self.dropout != None:
            out = self.dropout(out)
        return out

class DownSampleResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, downsample=True, print=False):
        super().__init__()
        self.downsample = downsample

        self.conv1 = SAConv2dLayer(in_channel, out_channel, 1, activation=False, print=print)
        # self.downsample1 = nn.UpsamplingBilinear2d(scale_factor=0.5)
        self.downsample1 = nn.Upsample(scale_factor=0.5)
        # self.downsample1 = SAConv2dLayer(out_channel, out_channel, 2, stride=2, groups=in_channel, activation=False, print=print)
        

        self.down_conv = SAConv2dLayer(in_channel, out_channel, 3, stride=2, activation=False)

        self.conv2 = SAConv2dLayer(in_channel, in_channel, 3, print=print)
        self.conv3 = SAConv2dLayer(in_channel, out_channel, 3, activation=False, print=print)
        # self.downsample2 = nn.UpsamplingBilinear2d(scale_factor=0.5)
        self.downsample2 = nn.Upsample(scale_factor=0.5)
        # self.downsample2 = SAConv2dLayer(out_channel, out_channel, 2, stride=2, groups=out_channel, activation=False, print=print)

    def forward(self, input):
        if self.downsample:
            skip = self.downsample1(self.conv1(input))
        else:
            skip = self.conv1(input)

        out = self.conv2(input)
        out = self.conv3(out)

        if self.downsample:
            out = self.downsample2(out)

        out = (out + skip) / math.sqrt(2)
        # out = out + skip
        return out

class DownSampleResBlock2(nn.Module):
    def __init__(self, in_channel, out_channel, downsample=True, print=False):
        super().__init__()
        self.downsample = downsample

        self.conv1 = SAConv2dLayer(in_channel, out_channel, kernel_size=2, stride=2, padding=0, activation=False)

    def forward(self, input):
        out = self.conv1(input)
        return out

class Encoder(nn.Module):
    def __init__(self, unin_batch=2):
        super().__init__()
        self.max_diff_ratio = 0.01

        self.real_center = nn.Parameter([1, 512], dtype='float32', is_bias=True)
        self.fake_center = nn.Parameter([1, 512], dtype='float32', is_bias=True)

        self.convs = nn.Sequential(
            SAConv2dLayer(3, 32, 1, activation=False),
            DownSampleResBlock(32, 64),         #128x128
            DownSampleResBlock(64, 128),        #64x64
            DownSampleResBlock(128, 128),       #32x32
            DownSampleResBlock(128, 256),       #16x16
            DownSampleResBlock(256, 256),       #8x8
            DownSampleResBlock(256, 512),       #4x4
            DownSampleResBlock(512, 512),       #2x2

            # DownSampleResBlock(128, 256),       #32x32
            # DownSampleResBlock(256, 512),       #16x16
            # DownSampleResBlock(512, 512*2),       #8x8
            # DownSampleResBlock(512*2, 512*4),       #4x4
            # DownSampleResBlock(512*4, 512*4),       #2x2
        )

        # self.final_feature = LinearLayer(512 * 4 *4, 512*4, activation=False, dropout=False)
        self.final_feature = LinearLayer(512, 512, activation=False, dropout=False)

        # self.batch_dis = BatchDiscriminator(512*4*4, 128)
        # self.batch_dis = BatchDiscriminator(512, 128)
        # self.initialize_module(self)
    
    def initialize_module(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.02, 0.02)
                # nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
                # nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
                # nn.init.normal_(m.weight)
                # if m.bias is not None:
                #         nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Conv1d):
                nn.init.uniform_(m.weight, -0.02, 0.02)
                # nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
    
    def set_max_diff_ratio(self, max_diff_ratio):
        self.max_diff_ratio = max_diff_ratio

    def forward(self, input):
        input = DiffMask(input, 0.75)
        # input = DiffMask(input, 0.5)
        # input = DiffMask(input, self.max_diff_ratio)
        # input = DiffAugment(input, policy='color,translation,cutout')
        # input = DiffAugment(input, policy='cutout')
        # input = input + 0.01 * torch.randn_like(input).to(device)
        out = self.convs(input)
        batch, channel, height, width = out.shape
        # feature = out.view(batch, -1)
        ### to do mean here

        feature = out.mean([-2, -1])
        
        # layer_norm = nn.LayerNorm(feature.shape[1:])
        # feature = layer_norm(feature)
        feature = self.final_feature(feature)
        # latent = feature
        # logit, _ = self.batch_dis(latent)
        # mean = torch.mean(feature, dim=1, keepdim=True)
        # std = torch.std(feature, dim=1, keepdim=True)
        # feature_norm = (feature-mean)/(std+1e-12)
        # feature = feature_s #+ torch.randn_like(feature_s).to(device)

        return 0.01*feature
        # return 0.1*feature

if __name__ == "__main__":
    net = Encoder()
    net.eval()
    img = torch.randn((2, 3, 256, 256))
    for i in range(1000):
        feature = net(img)
        print("ssssssssssss", feature.shape)

        