# -*- coding: utf-8 -*-

from torch.utils.data import DataLoader
from .dataset import FSC147

def build_loader(config, mode):
    data_path = config.DATA_PATH
    batch_size = config.BATCH_SIZE
    num_workers = config.NUM_WORKERS
    train_set = FSC147(data_path, mode)


    return DataLoader(
        train_set,
        batch_size = batch_size,
        num_workers = num_workers,
        shuffle = (mode=='train'),
        collate_fn=FSC147.collate_fn
    )