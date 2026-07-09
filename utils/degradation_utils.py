import torch
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor, Grayscale

from PIL import Image
import random
import numpy as np

from utils.image_utils import crop_img


class Degradation(object):
    def __init__(self, args):
        super(Degradation, self).__init__()
        self.args = args
        self.toTensor = ToTensor()
        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])




    # def _add_poisson_noise(self, clean_patch):
    #     noisy_patch = np.random.poisson(clean_patch).astype(np.uint8)
    #     noisy_patch = np.clip(noisy_patch, 0, 255)
    #     return noisy_patch, clean_patch


    # def _add_salt_and_pepper_noise(self, clean_patch, amount=0.01, salt_vs_pepper=0.5):
    #     noisy_patch = clean_patch.copy()
    #     num_pixels = clean_patch.size
    #     num_salt = int(amount * num_pixels * salt_vs_pepper)
    #     num_pepper = int(amount * num_pixels * (1.0 - salt_vs_pepper))

    #     # Add salt (white) noise
    #     coords = [np.random.randint(0, i, num_salt) for i in clean_patch.shape]
    #     noisy_patch[tuple(coords)] = 255

    #     # Add pepper (black) noise
    #     coords = [np.random.randint(0, i, num_pepper) for i in clean_patch.shape]
    #     noisy_patch[tuple(coords)] = 0

    #     return noisy_patch.astype(np.uint8), clean_patch


    def _add_gaussian_noise(self, clean_patch, sigma):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch



    def _add_poisson_noise(self, clean_patch):
        noisy_patch = np.random.poisson(clean_patch).astype(np.uint8)
        noisy_patch = np.clip(noisy_patch, 0, 255)
        return noisy_patch, clean_patch




    def _degrade_by_type(self, clean_patch, degrade_type):
        if degrade_type == 0:
            # denoise sigma=15
            degraded_patch, clean_patch = self._add_gaussian_noise(clean_patch, sigma=15)
        elif degrade_type == 1:
            # denoise sigma=25
            degraded_patch, clean_patch = self._add_gaussian_noise(clean_patch, sigma=25)
        elif degrade_type == 2:
            # denoise sigma=50
            degraded_patch, clean_patch = self._add_gaussian_noise(clean_patch, sigma=50)
        elif degrade_type == -1:
            degraded_patch, clean_patch = self._add_poisson_noise(clean_patch)
        return degraded_patch, clean_patch

    def degrade(self, clean_patch_1, clean_patch_2, degrade_type=None):
        if degrade_type == None:
            degrade_type = random.randint(0, 3)
        else:
            degrade_type = degrade_type

        degrad_patch_1, _ = self._degrade_by_type(clean_patch_1, degrade_type)
        degrad_patch_2, _ = self._degrade_by_type(clean_patch_2, degrade_type)
        return degrad_patch_1, degrad_patch_2

    def single_degrade(self,clean_patch,degrade_type = None):
        if degrade_type == None:
            degrade_type = random.randint(0, 3)
        else:
            degrade_type = degrade_type

        degrad_patch_1, _ = self._degrade_by_type(clean_patch, degrade_type)
        return degrad_patch_1
