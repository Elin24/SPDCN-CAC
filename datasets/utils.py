# -*- coding: utf-8 -*-

import torch
from torchvision import transforms

class NormalSample(object):
    def __init__(self):

        self.totensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
                mean=[0.56347245, 0.50660025, 0.45908741],
                std=[0.28393339, 0.2804536 , 0.30424776]
            )

    def __call__(self, image, dotmap=None):
        image = self.totensor(image)
        image = self.normalize(image)

        if dotmap is None:
            return image
        else:
            dotmap = torch.from_numpy(dotmap).float()
            return image, dotmap

jpg2id = lambda x: x.replace('.jpg', '')
