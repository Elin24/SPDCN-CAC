# -*- coding: utf-8 -*-

from .spdcn import SPDCN
from .loss.genloss import GeneralizedLoss
from .loss.mseloss import MSELoss
        

def build_model(config):
    model = SPDCN(config)
    # return model, MSELoss(config.FACTOR)
    return model, GeneralizedLoss(config.FACTOR)
