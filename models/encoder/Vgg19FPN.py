import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from mmcv.ops import DeformConv2d as dfconv


def nc2dc(nconv):
    w = nconv.weight
    # b = nconv.bias
    outc, inc, _, _ = w.shape
    dconv = dfconv(in_channels=inc, out_channels=outc, kernel_size=3, stride=1, padding=1)
    dconv.weight = nn.Parameter(w.clone())
    return dconv, inc

class SizeBlock(nn.Module):
    def __init__(self, conv):
        super(SizeBlock, self).__init__()
        self.conv, inc = nc2dc(conv)
        self.glob = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32)
        )
        self.local = nn.Sequential(
            nn.Conv2d(inc, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1)
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3 * 3 * 2, 3, padding=1)
        )
        self.relu = nn.ReLU()


    def forward(self, x, bsize):
        b, c, h, w = x.shape
        g_offset = self.glob(bsize)
        g_offset = g_offset.view(b, -1, 1, 1).repeat(1, 1, h, w).contiguous()
        l_offset = self.local(x)
        offset = self.fuse(torch.cat((g_offset, l_offset), dim=1))
        fea = self.conv(x, offset)
        return  self.relu(fea)

class NormBlock(nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.conv = nn.Sequential(conv, nn.ReLU(inplace=True))
    
    def forward(self, x, size=None):
        return self.conv(x)

class Vgg19FPN(nn.Module):
    def __init__(self, pretrained=True):
        super(Vgg19FPN, self).__init__()
        vgg = models.vgg19(pretrained=pretrained)
        mods = list(vgg.features.children())[:28]

        self.modlist = nn.ModuleList()
        self.selist = nn.ModuleList()
        last = 0
        for i, mod in enumerate(mods):
            if 'MaxPool2d' in str(mod):
                self.modlist.append(nn.Sequential(*mods[last:i-2]))
                self.selist.append(SizeBlock(mods[i-2]))
                last = i
        
        self.clsfc = nn.Conv2d(512, 512, 1, padding=0)
        self.denfc = nn.Conv2d(512, 512, 1, padding=0)
        
    def forward(self, x, msize):
        for mod, sem in zip(self.modlist, self.selist):
            x = mod(x) # b c h w
            x = sem(x, msize)
        fea = x 
        return self.clsfc(fea), self.denfc(fea)
    
    def outdim(self):
        return 512
