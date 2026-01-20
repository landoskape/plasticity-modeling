from __future__ import annotations
from math import ceil
import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.io import loadmat
from ..files import get_natural_images_dir


def load_natural_images():
    """
    Load natural images from the .mat file

    These are the natural images provided by Olshausen & Field on their website.
    """
    fpath = get_natural_images_dir()
    X = loadmat(fpath)
    return X["IMAGES"]


class NatPatchDataset(Dataset):

    def __init__(self, N: int, L: int, border: int = 4):
        super().__init__()
        self.N = N  # number of images
        self.L = L  # length of image (image dimension = L**2)
        self.border = border  # border around the image
        self.images = None  # holder for the images
        self._extract_patches()  # initialize patches

    def __len__(self):
        """Return the number of images"""
        return self.images.shape[0]

    def __getitem__(self, idx):
        """Return the image at the given index"""
        return self.images[idx]

    def _extract_patches(self):
        """
        Retrieve the images, extract patches, prepare the dataloader.
        """
        images = torch.tensor(load_natural_images()).float()
        img_size = images.shape[0]
        n_img = images.shape[2]
        self.images = torch.zeros((self.N, self.L, self.L))
        num_per_image = ceil(self.N / n_img)
        image_index = torch.arange(n_img).repeat(num_per_image)[: self.N]
        xs = torch.randint(self.border, img_size - self.L - self.border, (self.N,))
        ys = torch.randint(self.border, img_size - self.L - self.border, (self.N,))
        for ii, (imidx, x, y) in enumerate(zip(image_index, xs, ys)):
            self.images[ii] = images[x : x + self.L, y : y + self.L, imidx]
