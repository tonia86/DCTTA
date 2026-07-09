import glob
import random
import os
from PIL import Image
import cv2
import math
import numpy as np
import torch
import torch.utils.data as data
from torchvision.transforms.functional import normalize
from .utils import pad_tensor, hiseq_color_cv2_img, generate_position_encoding


class Dataset(data.Dataset):
    def __init__(
        self,
        folder,
        image_size,
        name
    ):
        super().__init__()


        self.gt_root = folder[0]
        self.input_root = folder[1]


        self.gt_paths = sorted(glob.glob(os.path.join(self.gt_root, '*.png')) + \
                glob.glob(os.path.join(self.gt_root, '*.jpg')))

        self.input_paths = sorted(glob.glob(os.path.join(self.input_root, '*.png')) + \
                glob.glob(os.path.join(self.input_root, '*.jpg')))


        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]

        if  name == "train":
            self.input_mode  = "crop"
            self.crop_size = [256, 256]
            self.concat_with_position_encoding = True
            self.use_flip = True
        else:
            self.crop_size = [256, 256]
            self.concat_with_hiseq = True
            self.input_mode = "crop"
            self.divide = 32
            self.concat_with_position_encoding = True






    def __len__(self):
        return len(self.gt_paths)

    def __getitem__(self, index):
        # condition

        gt_path = self.gt_paths[index]
        gt_name = os.path.split(gt_path)[-1]
        input_path = self.input_paths[index]

        gt_img = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB) / 255.
        input_img = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB) / 255. 


        if hasattr(self, 'use_flip') and self.use_flip and np.random.uniform() < 0.5:
            gt_img = cv2.flip(gt_img, 1, gt_img)
            input_img = cv2.flip(input_img, 1, input_img)



        if self.input_mode == 'crop':
            crop_size = self.crop_size
            H, W, _ = input_img.shape
            assert input_img.shape[:2] == gt_img.shape[:2], f"{input_img.shape}, {gt_img.shape}, {gt_path}"
            h = np.random.randint(0, H - crop_size[0] + 1)
            w = np.random.randint(0, W - crop_size[1] + 1)
            gt_img = gt_img[h: h + crop_size[0], w: w + crop_size[1], :]
            input_img = input_img[h: h + crop_size[0], w: w + crop_size[1], :]



        # if self.input_mode == 'pad':
        #     divide = self.divide
        #     gt_img_pt = torch.from_numpy(gt_img.transpose((2, 0, 1)))
        #     input_img_pt = torch.from_numpy(input_img.transpose((2, 0, 1)))
        #     gt_img_pt = torch.unsqueeze(gt_img_pt, 0)
        #     input_img_pt = torch.unsqueeze(input_img_pt, 0)
        #     gt_img_pt, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(gt_img_pt, divide)
        #     input_img_pt, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(input_img_pt, divide)
        #     gt_img_pt = gt_img_pt[0, ...]
        #     input_img_pt = input_img_pt[0, ...]
        #     gt_img = gt_img_pt.numpy().transpose((1, 2, 0))
        #     input_img = input_img_pt.numpy().transpose((1, 2, 0))

        gt_img_pt = torch.from_numpy(gt_img.transpose((2, 0, 1)))
        input_img_pt = torch.from_numpy(input_img.transpose((2, 0, 1)))

        input_img_pt = input_img_pt.float()
        gt_img_pt = gt_img_pt.float()


        # self.return_dict = {}
        # if self.input_mode == 'pad':
        #     self.return_dict["pad_left"] = pad_left
        #     self.return_dict["pad_right"] = pad_right
        #     self.return_dict["pad_top"] = pad_top
        #     self.return_dict["pad_bottom"] = pad_bottom
        return [gt_img_pt,input_img_pt]





    def load_name(self, index, sub_dir=False):
        name = self.gt_paths[index]
        return os.path.basename(name)

    def to_tensor(self, img):
        img = Image.fromarray(img)  # returns an image object.
        img_t = TF.to_tensor(img).float()
        return img_t


    def get_pad_size(self):

        return self.return_dict

