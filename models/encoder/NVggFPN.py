import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import copy

class NVggFPN(nn.Module):
    def __init__(self, pretrained=True):
        super(NVggFPN, self).__init__()
        vgg = models.vgg19(pretrained=pretrained)
        mods = list(vgg.features.children())[:27]

        self.encoder = nn.Sequential(*mods)

        outc = 512
        self.clsfc = nn.Conv2d(outc, outc, 1)
        self.denfc = nn.Conv2d(outc, outc, 1)
        
    def forward(self, x, msize):
        x = self.encoder(x)
        return self.clsfc(x), self.denfc(x)
    
    def outdim(self):
        return 512
