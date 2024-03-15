import glob
import logging
import os
import random

import cv2
import numpy as np
from sympy import true
import torch
import torch.nn.functional as F

from omegaconf import open_dict, OmegaConf

from torch.utils.data import Dataset
from .masks import MixedMaskGenerator
import albumentations as A

class LAMAVAEDataset(Dataset):
    def __init__(self, indir, mask_gen_kwargs, out_size, use_transform = True):
        self.in_files = list(glob.glob(os.path.join(indir, '**', '*.jpg'), recursive=True))
        self.mask_generator = MixedMaskGenerator(**mask_gen_kwargs)
        self.transform = A.Compose([
            A.RandomScale(scale_limit=0.2),  # +/- 20%
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
        self.transform_val = A.Compose([
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.CenterCrop(height=out_size, width=out_size),
            A.ToFloat()
        ])
        self.iter_i = 0
        self.use_transform = use_transform

    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, item):
        path = self.in_files[item]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.use_transform:
            img = self.transform(image=img)['image']
        else:
            img = self.transform_val(image=img)['image']
        img = np.transpose(img, (2, 0, 1))
        
        # TODO: maybe generate mask before augmentations? slower, but better for segmentation-based masks
        mask = self.mask_generator(img, iter_i=self.iter_i)
        img = np.transpose(img, (1, 2, 0))
        img = torch.tensor(img, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        self.iter_i += 1
        return dict(image=img,
                    mask=mask)