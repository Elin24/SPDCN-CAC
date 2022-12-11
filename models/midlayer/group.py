import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupOp(nn.Module):
    def __init__(self):
        super().__init__()
        

    def forward(self, feature, anchor, mask=None, boxes=None):
        sim = F.normalize(feature, 2, dim=1) * F.normalize(anchor, 2, dim=1)
        sim = F.relu(sim.sum(dim=1, keepdim=True), inplace=True)
        return sim
