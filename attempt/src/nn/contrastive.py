import random
from typing import Tuple

import torch
import numpy as np
import random
import torchvision.transforms as transforms
from PIL import ImageOps, ImageFilter


class SingleInstanceTwoView:
    '''
    This class is adapted from BarlowTwins and SimSiam in our external_src folder.
    '''

    def __init__(self, imsize: int, mean: Tuple[float], std: Tuple[float]):
        self.is_odd = True
        self.seed1 = 0
        self.seed2 = 1

        self.augmentation1 = transforms.Compose([
            transforms.Resize(
                imsize, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomResizedCrop(
                imsize,
                scale=(0.6, 1.6),
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
                ],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        self.augmentation2 = transforms.Compose([
            transforms.Resize(
                imsize, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomResizedCrop(
                imsize,
                scale=(0.6, 1.6),
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
                ],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
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
            aug1 = self.augmentation1(x)
            random.seed(self.seed2)
            torch.manual_seed(self.seed2)
            aug2 = self.augmentation2(x)
            self.is_odd = False
        else:
            # Apply the same random seeds.
            random.seed(self.seed1)
            torch.manual_seed(self.seed1)
            aug1 = self.augmentation1(x)
            random.seed(self.seed2)
            torch.manual_seed(self.seed2)
            aug2 = self.augmentation2(x)
            self.is_odd = True
        return aug1, aug2


class GaussianBlur(object):

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        else:
            return img