# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as tF
from .geomloss import SamplesLoss

class Cost:
    def __init__(self, factor=128) -> None:
        self.factor = factor
        self.box_scale = [2, 2]

    def __call__(self, x, y):
        X, Y = x.clone(), y.clone()
        X[:, :, 0] = X[:, :, 0] / self.box_scale[0]
        X[:, :, 1] = X[:, :, 1] / self.box_scale[1]
        Y[:, :, 0] = Y[:, :, 0] / self.box_scale[0]
        Y[:, :, 1] = Y[:, :, 1] / self.box_scale[1]
        x_col = X.unsqueeze(-2) / self.factor
        y_row = Y.unsqueeze(-3) / self.factor
        C = torch.sum((x_col - y_row) ** 2, -1) ** 0.5
        return C

per_cost = Cost(factor=128)
eps = 1e-8

class GeneralizedLoss(nn.modules.loss._Loss):
    def __init__(self, factor=100, reduction='mean') -> None:
        super().__init__()
        self.factor = factor
        self.reduction = reduction
        self.tau = 5

        self.cost = per_cost
        self.blur = 0.01
        self.scaling = 0.75
        self.reach = 0.5
        self.p = 1
        self.uot = SamplesLoss(blur=self.blur, scaling=self.scaling, debias=False, backend='tensorized', cost=self.cost, reach=self.reach, p=self.p)
        self.pointLoss = nn.L1Loss(reduction=reduction)
        self.pixelLoss = nn.MSELoss(reduction=reduction)

        self.down = 1

    def forward(self, dens, dots, box_size=None):
        bs = dens.size(0)
        point_loss, pixel_loss, emd_loss = 0, 0, 0
        entropy = 0
        for i in range(bs):
            den = dens[i, 0]
            seq = torch.nonzero(dots[i, 0])
            if box_size is not None:
                self.cost.box_scale = torch.sigmoid((box_size[i] - 64) / 32) * 2 + 1
            else:
                self.cost.box_scale = [2, 2]

            if seq.size(0) < 1 or den.sum() < eps:
                point_loss += torch.abs(den).mean()
                pixel_loss += torch.abs(den).mean()
                emd_loss += torch.abs(den).mean()
            else:
                A, A_coord = self.den2coord(den)
                A_coord = A_coord.reshape(1, -1, 2)
                A = A.reshape(1, -1, 1)

                B_coord = seq[None, :, :]
                B = torch.ones(seq.size(0)).float().cuda().view(1, -1, 1) * self.factor
                
                oploss, F, G = self.uot(A, A_coord, B, B_coord)
                
                C = self.cost(A_coord, B_coord)
                PI = torch.exp((F.view(1, -1, 1) + G.view(1, 1, -1) - C).detach() / (self.blur ** self.p)) * A * B.view(1, 1, -1)
                entropy += torch.mean((1e-20+PI) * torch.log(1e-20+PI))
                emd_loss += torch.mean(oploss)
                point_loss += self.pointLoss(PI.sum(dim=1).view(1, -1, 1), B)
                pixel_loss += self.pixelLoss(PI.sum(dim=2).detach().view(1, -1, 1), A)
                
        loss = (emd_loss + self.tau * (point_loss + pixel_loss) + self.blur * entropy) / bs
        return loss
    
    def den2coord(self, denmap):
        assert denmap.dim() == 2, f"denmap.shape = {denmap.shape}, whose dim is not 2"
        coord = torch.nonzero(denmap)
        denval = denmap[coord[:, 0], coord[:, 1]]
        return denval, coord
    
