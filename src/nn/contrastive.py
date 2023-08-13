import random
from typing import Tuple

import torch
import numpy as np
import random
import torchvision.transforms as transforms
from PIL import ImageFilter


class SingleInstanceTwoView:
    '''
    This class is adapted from BarlowTwins and SimSiam in our external_src folder.
    '''

    def __init__(self, imsize: int, mean: Tuple[float], std: Tuple[float]):
        self.is_odd = True
        self.seed1 = 0
        self.seed2 = 1

        self.augmentation = transforms.Compose([
            transforms.Resize(
                imsize, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomResizedCrop(
                imsize,
                scale=(0.6, 1.6),
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
            ],
                                   p=0.4),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __call__(self, x):
        '''
        The purpose is to apply the same two augmentations
        for each two consecutive incoming images.
        '''
        if self.is_odd:
            # Re-randomize the seeds.
            self.seed1 = np.random.randint(2147483647)
            self.seed2 = np.random.randint(2147483647)
            random.seed(self.seed1)
            torch.manual_seed(self.seed1)
            aug1 = self.augmentation(x)
            random.seed(self.seed2)
            torch.manual_seed(self.seed2)
            aug2 = self.augmentation(x)
            self.is_odd = False
        else:
            # Apply the same random seeds.
            random.seed(self.seed1)
            torch.manual_seed(self.seed1)
            aug1 = self.augmentation(x)
            random.seed(self.seed2)
            torch.manual_seed(self.seed2)
            aug2 = self.augmentation(x)
            self.is_odd = True
        return aug1, aug2


class GaussianBlur(object):

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
