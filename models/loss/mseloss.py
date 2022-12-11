# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import math


def agaussian(kernel_size):
    if type(kernel_size) == int:
        kernel_size = (kernel_size, kernel_size)
    muh, muw = kernel_size[0] // 2, kernel_size[1] // 2
    sigmah, sigmaw = math.floor(kernel_size[0] / 4), math.floor(kernel_size[1] // 4)
    gaussh = lambda x: math.exp(-(x - muh) ** 2 / float(2 * sigmah ** 2))
    gaussw = lambda x: math.exp(-(x - muw) ** 2 / float(2 * sigmaw ** 2))
    hseq = torch.tensor([gaussh(x) for x in range(kernel_size[0])]).unsqueeze(1)
    wseq = torch.tensor([gaussw(x) for x in range(kernel_size[1])]).unsqueeze(0)

    kernels = Variable(hseq @ wseq)
    weight = kernels.reshape(1, 1, kernel_size[0], kernel_size[1])
    weight = weight / weight.sum()
    return weight

class MSELoss(nn.modules.loss._Loss):
    def __init__(self, factor, reduction='mean') -> None:
        super().__init__()
        
        self.gsize = 17
        self.register_buffer('gaussian', agaussian(self.gsize))
        self.factor = factor

        self.lossfunc = nn.MSELoss(reduce=reduction)
    
    def forward(self, den, dot, box_size=None):
        if box_size is not None:
            tar = []
            ksc = torch.sigmoid((box_size - 64) / 16) * 2 + 1
            for i in range(dot.size(0)):
                k1 = int(ksc[i, 0].item() * 8) | 1
                k2 = int(ksc[i, 1].item() * 8) | 1
                g = agaussian((k1, k2)).to(dot)
                tar.append(F.conv2d(dot[i:i+1], g, stride=1, padding=(k1 // 2, k2 // 2)))
            tar = torch.cat(tar, dim=0)
        else:
            tar = F.conv2d(dot, self.gaussian, stride=1, padding=self.gsize // 2)
        tar = tar * self.factor
        loss = self.lossfunc(den, tar)
        return loss
    