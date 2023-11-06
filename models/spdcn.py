# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from .midlayer import ROIAlign, GroupOp
from .encoder import Vgg19FPN
from .decoder import COMPSER

class SPDCN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Vgg19FPN.Vgg19FPN()
        feadim = self.encoder.outdim()
        self.roi = 16
        self.roialign = ROIAlign(self.roi, 1./8)
        self.cross = GroupOp()
        self.decoder = COMPSER.COMPSER(feadim)
        self.factor = config.FACTOR


    def forward(self, image, boxes):
        bsize = torch.stack((boxes[:, 4] - boxes[:, 2], boxes[:, 3] - boxes[:, 1]), dim=-1)
        bs_mean = bsize.view(-1, 3, 2).float().mean(dim=1)

        b, _, imh, imw = image.shape
        clsfea, denfea = self.encoder(image, bs_mean)

        patches = self.roialign(clsfea, boxes)
        anchors_patchs = patches.view(b, 3, -1, self.roi, self.roi).mean(dim=1)
        anchor_cls = anchors_patchs.mean(dim=(-1, -2), keepdim=True)
        
        mask = self.cross(clsfea, anchor_cls)
        denmap = self.decoder(denfea * mask)
        
        return denmap
    