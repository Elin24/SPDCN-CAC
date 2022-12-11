# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, NL='relu', same_padding=False, bn=False, dilation=1):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = []
        if dilation==1:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        if NL == 'relu' :
            self.relu = nn.ReLU(inplace=True)
        elif NL == 'prelu':
            self.relu = nn.PReLU() 
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class COMPSER(nn.Module):
    def __init__(self, indim):
        super(COMPSER, self).__init__()
        
        self.decoder = nn.Sequential(
                                     Conv2d(indim, 256, 3, same_padding=True, NL='relu'),
                                     Conv2d(256,  256, 3, same_padding=True, NL='relu'),
                                     nn.PixelShuffle(8),
                                     nn.Conv2d(4, 1, 3, padding=1),
                                     nn.ReLU(inplace=True)
                                     )
        
        self.weights_normal_init(self.decoder, dev=0.005)

    def forward(self, x):
        x = self.decoder(x)
        return x
    
    def weights_normal_init(self, model, dev=0.01):
        if isinstance(model, list):
            for m in model:
                self.weights_normal_init(m, dev)
        else:
            for m in model.modules():
                if isinstance(m, nn.Conv2d):                
                    m.weight.data.normal_(0.0, dev)
                    if m.bias is not None:
                        m.bias.data.fill_(0.0)
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0.0, dev)