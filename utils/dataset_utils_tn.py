import os
import random
import copy
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
import torch

from utils.image_utils import random_augmentation, crop_img
from utils.degradation_utils import Degradation
from torchvision.transforms import functional as F
import cv2


# def crop_patch(img_1,patch_size=256):
#     H = img_1.shape[1]
#     W = img_1.shape[2]
#     ind_H = random.randint(0, H - patch_size)
#     ind_W = random.randint(0, W - patch_size)

#     patch_1 = img_1[:,ind_H:ind_H + patch_size, ind_W:ind_W + patch_size]
#     return patch_1



################################

def crop_patch(img_1, patch):
    if isinstance(img_1, torch.Tensor):
        img_1 = img_1.permute(1, 2, 0).cpu().numpy()
    patch_1 = cv2.resize(img_1, (patch, patch), interpolation=cv2.INTER_AREA)
    patch_1 = torch.from_numpy(patch_1).permute(2, 0, 1).float()
    return patch_1


class PromptTrainDataset(Dataset):
    def __init__(self, args,seed=42):
        super(PromptTrainDataset, self).__init__()
        self.args = args
        self.rs_ids = []
        self.hazy_ids = []
        self.D = Degradation(args)
        self.de_temp = 0
        self.de_type = self.args.de_type
        self.crop = self.args.crop
        print(self.de_type)

        self.de_dict = {'denoise_15': 0, 'denoise_25': 1, 'denoise_50': 2, 'derain': 3, 'dehaze': 4, 'deblur' : 5}


        # **固定随机种子**
        self.seed = seed
        self._set_seed()

        self._init_ids()
        self._merge_ids()

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])

        self.toTensor = ToTensor()



    def _set_seed(self):
        """固定随机种子"""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)



    def _init_ids(self):
        if 'derain' in self.de_type:
            self._init_rs_ids()
        if 'dehaze' in self.de_type:
            self._init_hazy_ids()
        if 'denoise_15' in self.de_type or 'denoise_25' in self.de_type or 'denoise_50' in self.de_type:
            self._init_clean_ids()

        random.shuffle(self.de_type)

    def _init_hazy_ids(self):

        self.hazy_ids = []

        if 'LOL' in self.args.dehaze_dir:
            input_dir = self.args.dehaze_dir
        else:
            input_dir = os.path.join(self.args.dehaze_dir, "LQ")

        for input_name in sorted(os.listdir(input_dir)):
            input_path = os.path.join(input_dir, input_name)
            if os.path.exists(input_path):
                self.hazy_ids.append({"input_path": input_path, "de_type": 4})



        # self.hazy_ids = self.hazy_ids * 4
        # self.hazy_counter = 0


        self.num_hazy = len(self.hazy_ids)

        print("Total Hazy Ids : {}".format(self.hazy_ids))
        print("Total Hazy Ids : {}".format(self.num_hazy))
                
    def _init_rs_ids(self):
        self.rs_ids = []

        if "raindrop" or "rainstreak" in self.args.derain_dir:
            input_dir = self.args.derain_dir
        else:
            input_dir = os.path.join(self.args.derain_dir, "LQ")



        for input_name in sorted(os.listdir(input_dir)):
            input_path = os.path.join(input_dir, input_name)
            if os.path.exists(input_path):
                self.rs_ids.append({"input_path": input_path, "de_type": 3})

        # ###数据扩充
        # self.rs_ids = self.rs_ids * 120
        # self.rl_counter = 0
        # self.num_rl = len(self.rs_ids)
        # print("Total Rainy Ids : {}".format(self.num_rl))
        # print("Total Rainy Ids : {}".format(self.rs_ids))



    def _init_clean_ids(self):


        self.s15_ids = []
        self.s25_ids = []
        self.s50_ids = []
        self.poisson_ids = []
        if 'Kodak24' or 'McMaster' in self.args.denoise_dir:
            input_dir = os.path.join(self.args.denoise_dir)
        else:
            input_dir = os.path.join(self.args.denoise_dir, "GT")

        for input_name in sorted(os.listdir(input_dir)):
            input_path = os.path.join(input_dir, input_name)

            if os.path.exists(input_path):
                self.s15_ids.append({"gt_path": input_path, "de_type": 0})
                self.s25_ids.append({"gt_path": input_path, "de_type": 1})
                self.s50_ids.append({"gt_path": input_path, "de_type": 2})
                self.poisson_ids.append({"gt_path": input_path, "de_type": -1})


        self.num_clean = len(self.s15_ids)
        print("Total Denoise Ids : {}".format(self.num_clean))


    ###################
    # def _crop_patch(self, img_1):
    #     H = img_1.shape[0]
    #     W = img_1.shape[1]
    #     ind_H = random.randint(0, H - self.args.patch_size)
    #     ind_W = random.randint(0, W - self.args.patch_size)

    #     patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
    #     return patch_1


    #####################resize
    def _crop_patch(self, img_1):
        patch_1 = cv2.resize(
            img_1, 
            (self.args.patch_size, self.args.patch_size), 
            interpolation=cv2.INTER_AREA  # 更适合缩小图像
        )
        return patch_1





    def _get_gt_name(self, rainy_name):
        if 'rainy' in rainy_name:
            gt_name = rainy_name.split("rainy")[0] + 'rainy1/GT/norain-' + rainy_name.split('rain-')[-1]
        if 'Rain100H' in rainy_name:
            gt_name = rainy_name.split("Rain100H")[0] + 'Rain100H/GT' + rainy_name.split('LQ')[-1]
        if 'Rain100L' in rainy_name:
            gt_name = rainy_name.split("Rain100L")[0] + 'Rain100L/GT' + rainy_name.split('LQ')[-1]

        if 'rainstreak_raindrop' in rainy_name:
            gt_name = rainy_name.split("RainDS_real")[0] + 'RainDS_real/gt' + rainy_name.split('rainstreak_raindrop')[-1]            
        # if 'raindrop' in rainy_name:
        #     gt_name = rainy_name.split("RainDS_real")[0] + 'RainDS_real/gt' + rainy_name.split('raindrop')[-1]
        # if 'rainstreak' in rainy_name:
        #     gt_name = rainy_name.split("RainDS_real")[0] + 'RainDS_real/gt' + rainy_name.split('rainstreak')[-1]

        return gt_name

    def _get_nonhazy_name(self, hazy_name):
        if 'hazy' in hazy_name:    
            nonhazy_name = hazy_name.split("hazy")[0] + 'hazy1/GT/' + hazy_name.split('/')[-1].split('_')[0] + ".png"
        if 'HSTS' in hazy_name:   
            nonhazy_name = hazy_name.split("HSTS")[0] + 'HSTS/GT/' + hazy_name.split('/')[-1].split('_')[0]            
        if 'Dense_Haze' in hazy_name:   
            nonhazy_name = hazy_name.split("Dense_Haze")[0] + 'Dense_Haze/GT/' + hazy_name.split('/')[-1].split('_')[0] 
        if 'NH-HAZE' in hazy_name:   
            nonhazy_name = hazy_name.split("NH-HAZE")[0] + 'NH-HAZE/GT/' + hazy_name.split('/')[-1].split('_')[0] 
        if 'indoor' in hazy_name:   
            nonhazy_name = hazy_name.split("indoor")[0] + 'indoor/GT/' + hazy_name.split('/')[-1].split('_')[0] 
        if 'O-HAZY_resize' in hazy_name:   
            nonhazy_name = hazy_name.split("O-HAZY_resize")[0] + 'O-HAZY_resize/GT/' + hazy_name.split('/')[-1].split('_')[0] 
        if 'O-HAZY_crop' in hazy_name:   
            nonhazy_name = hazy_name.split("O-HAZY_crop")[0] + 'O-HAZY_crop/GT/' + hazy_name.split('/')[-1].split('_')[0] 
        if 'eval15' in hazy_name:   
            nonhazy_name = hazy_name.split("eval15")[0] + 'eval15/high/' + hazy_name.split('/')[-1].split('_')[0] 

        if 'Real_captured' in hazy_name:  
            if "Low_resize" in hazy_name:
                nonhazy_name = hazy_name.split("Real_captured")[0] + 'Real_captured/Test/Normal_resize/' + hazy_name.split('/')[-1].split('_')[0].replace('low', 'normal', 1)
            else:
                nonhazy_name = hazy_name.split("Real_captured")[0] + 'Real_captured/Test/Normal/' + hazy_name.split('/')[-1].split('_')[0].replace('low', 'normal', 1)

        return nonhazy_name

    def _merge_ids(self):
        self.sample_ids = []

        # if "denoise_15" in self.de_type:
        #     self.sample_ids += self.s15_ids
        #     self.sample_ids += self.s25_ids
            # self.sample_ids += self.s50_ids
            # self.sample_ids +=self.poisson_ids

        if "derain" in self.de_type:
            self.sample_ids+= self.rs_ids
        
        # if "dehaze" in self.de_type:
            # self.sample_ids+= self.hazy_ids
        print(len(self.sample_ids))

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        de_id = sample["de_type"]

        if de_id < 3:
            # if de_id == 0:
            #     clean_id = sample["gt_path"]
            # elif de_id == 1:
            #     clean_id = sample["gt_path"]
            # elif de_id == 2:
            clean_id = sample["gt_path"]


            clean_img = crop_img(np.array(Image.open(clean_id).convert('RGB')), base=16)

            #####
            # clean_patch = self.crop_transform(clean_img)
            # clean_patch= np.array(clean_patch)
            #####
            clean_name = clean_id.split("/")[-1].split('.')[0]

            # clean_patch = random_augmentation(clean_patch)[0]
            clean_patch = self._crop_patch(clean_img)

            # clean_patch = clean_img


            degrad_patch = self.D.single_degrade(clean_patch, de_id)
        else:
            if de_id == 3:
                # Rain Streak Removal
                degrad_img = np.array(Image.open(sample["input_path"]).convert('RGB'))
                clean_name = self._get_gt_name(sample["input_path"])
                clean_img = np.array(Image.open(clean_name).convert('RGB'))
            elif de_id == 4:
                # Dehazing with SOTS outdoor training set
                degrad_img = np.array(Image.open(sample["input_path"]).convert('RGB'))
                clean_name = self._get_nonhazy_name(sample["input_path"])
                clean_img = np.array(Image.open(clean_name).convert('RGB'))


            ##裁剪
            if self.crop:
                degrad_patch = self._crop_patch(degrad_img)
                clean_patch = self._crop_patch(clean_img)
            else:
                degrad_patch = degrad_img
                clean_patch = clean_img


        degrad_patch  = self.toTensor(degrad_patch)
        clean_patch = self.toTensor(clean_patch)

        filename = os.path.basename(clean_name)

        return filename,degrad_patch,clean_patch


    def __len__(self):
        return len(self.sample_ids)





class DenoiseTestDataset(Dataset):
    def __init__(self, args,patch=None):
        super(DenoiseTestDataset, self).__init__()
        self.args = args
        self.clean_ids = []
        self.sigma = 15
        self.patch = patch
        self._init_clean_ids()

        self.toTensor = ToTensor()

    def _init_clean_ids(self):
        name_list = os.listdir(self.args.denoise_path)
        self.clean_ids += [self.args.denoise_path + id_ for id_ in name_list]

        self.num_clean = len(self.clean_ids)

    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch



    def _add_poisson_noise(self, clean_patch):
        noisy_patch = np.random.poisson(clean_patch).astype(np.uint8)
        noisy_patch = np.clip(noisy_patch, 0, 255)
        return noisy_patch, clean_patch




    def set_sigma(self, sigma):
        self.sigma = sigma

    def __getitem__(self, clean_id):
        clean_img = crop_img(np.array(Image.open(self.clean_ids[clean_id]).convert('RGB')), base=16)
        clean_name = self.clean_ids[clean_id].split("/")[-1].split('.')[0]
        if self.sigma == -1:
            noisy_img, _ = self._add_poisson_noise(clean_img)
        else:
            noisy_img, _ = self._add_gaussian_noise(clean_img)

        clean_img, noisy_img = self.toTensor(clean_img), self.toTensor(noisy_img)

        if self.patch:
            noisy_img = crop_patch(noisy_img,self.patch)
            clean_patch = crop_patch(clean_img,self.patch)

        return [clean_name], noisy_img, clean_img
    def tile_degrad(input_,tile=128,tile_overlap =0):
        sigma_dict = {0:0,1:15,2:25,3:50}
        b, c, h, w = input_.shape
        tile = min(tile, h, w)
        assert tile % 8 == 0, "tile size should be multiple of 8"

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h, w).type_as(input_)
        W = torch.zeros_like(E)
        s = 0
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = input_[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = in_patch
                # out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(in_patch)

                E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
                W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
        # restored = E.div_(W)

        restored = torch.clamp(restored, 0, 1)
        return restored
    def __len__(self):
        return self.num_clean


class DerainDehazeDataset(Dataset):
    def __init__(self, args, task="derain",addnoise = False,sigma = None,patch=None):
        super(DerainDehazeDataset, self).__init__()
        self.ids = []
        self.task_idx = 1
        self.args = args

        self.task_dict = {'derain': 0, 'dehaze': 1}
        self.toTensor = ToTensor()
        self.addnoise = addnoise
        self.sigma = sigma
        self.patch = patch

        self.set_dataset(task)


    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch





    def _init_input_ids(self):
        if self.task_idx == 0:
            
            if "RainDS" in self.args.derain_path:
                print("!!!!!!!!!!!!385")
                self.ids = []
                name_list = os.listdir(self.args.derain_path)
                self.ids += [self.args.derain_path + id_ for id_ in name_list]
            else:
                self.ids = []
                name_list = os.listdir(self.args.derain_path + 'LQ/')
                # print(name_list)
                # print(self.args.derain_path)
                self.ids += [self.args.derain_path + 'LQ/' + id_ for id_ in name_list]
        elif self.task_idx == 1:

            if 'LOL' in self.args.dehaze_path:
                self.ids = []
                name_list = os.listdir(self.args.dehaze_path )
                self.ids += [self.args.dehaze_path + id_ for id_ in name_list]

            else:
                self.ids = []
                name_list = os.listdir(self.args.dehaze_path + 'LQ/')
                self.ids += [self.args.dehaze_path + 'LQ/' + id_ for id_ in name_list]


        self.length = len(self.ids)

    def _get_gt_path(self, degraded_name):
        # if self.task_idx == 0 and 'Rain100H' not in degraded_name:
        #     dir_name = degraded_name.split("LQ")[0] + 'GT/'
        #     name = "no" + degraded_name.split('/')[-1].split('_')[0]
        #     gt_name = dir_name + name
        # elif self.task_idx == 0 and 'Rain100H' in degraded_name:
        #     dir_name = degraded_name.split("LQ")[0] + 'GT/'
        #     name = degraded_name.split('/')[-1].split('_')[0]
        #     gt_name = dir_name + name

        if self.task_idx == 0:

            ########rainds
            if  "rainstreak_raindrop" in degraded_name:
                dir_name = degraded_name.split("rainstreak_raindrop")[0] + 'gt/'
                name = degraded_name.split('/')[-1].split('_')[0]
                gt_name = dir_name + name
            if  "raindrop" in degraded_name:
                dir_name = degraded_name.split("raindrop")[0] + 'gt/'
                name = degraded_name.split('/')[-1].split('_')[0]
                gt_name = dir_name + name
            if "rainstreak" in degraded_name:
                dir_name = degraded_name.split("rainstreak")[0] + 'gt/'
                name = degraded_name.split('/')[-1].split('_')[0]
                gt_name = dir_name + name
            else:
                dir_name = degraded_name.split("LQ")[0] + 'GT/'
                name = degraded_name.split('/')[-1].split('_')[0]
                gt_name = dir_name + name



        elif self.task_idx == 1 and 'LQ' in degraded_name:
            dir_name = degraded_name.split("LQ")[0] + 'GT/'
            name = degraded_name.split('/')[-1].split('_')[0]
            # if 'Dense_Haze' in dir_name:
            #     name = name +"_GT"
            if 'png' not in name and 'jpg' not in name:
                name = name +".png"
            gt_name = dir_name + name
        elif self.task_idx == 1 and 'eval15' in degraded_name:          
            dir_name = degraded_name.split("low")[0] + 'high/'
            name = degraded_name.split('/')[-1].split('_')[0]
            if 'png' not in name:
                name = name +".png"
            gt_name = dir_name + name
        #### lolV2 
        else:

            if "Low_resize" in degraded_name:
                modified_path = degraded_name.replace('Low_resize', 'Normal_resize')
                head, sep, tail = modified_path.partition("Normal_resize")
                dir_name = head + sep +'/'
            else:
                modified_path = degraded_name.replace('/Low/', '/Normal/', 1).replace('/low', '/normal', 1)
                dir_name = modified_path.split("normal")[0]

            if "low" in degraded_name.split('/')[-1].split('_')[0]:
                name = degraded_name.split('/')[-1].split('_')[0].replace('low', 'normal', 1)
                if 'png' not in name:
                    name += ".png"
                gt_name = dir_name + name
            else:
                gt_name = dir_name



        return gt_name


    def set_dataset(self, task):
        self.task_idx = self.task_dict[task]
        self._init_input_ids()

    def __getitem__(self, idx):
        degraded_path = self.ids[idx]
        clean_path = self._get_gt_path(degraded_path)

        degraded_img = crop_img(np.array(Image.open(degraded_path).convert('RGB')), base=16)
        if self.addnoise:
            degraded_img,_ = self._add_gaussian_noise(degraded_img)
        clean_img = crop_img(np.array(Image.open(clean_path).convert('RGB')), base=16)

        clean_img, degraded_img = self.toTensor(clean_img), self.toTensor(degraded_img)
        degraded_name = degraded_path.split('/')[-1][:-4]

        ####裁剪#########################################################################

        if self.patch:
            degraded_img = crop_patch(degraded_img,self.patch)
            clean_img = crop_patch(clean_img,self.patch)


        return [degraded_name], degraded_img, clean_img

    def __len__(self):
        return self.length


class TestSpecificDataset(Dataset):
    def __init__(self, args):
        super(TestSpecificDataset, self).__init__()
        self.args = args
        self.degraded_ids = []
        self._init_clean_ids(args.test_path)

        self.toTensor = ToTensor()

    def _init_clean_ids(self, root):
        extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']
        if os.path.isdir(root):
            name_list = []
            for image_file in os.listdir(root):
                if any([image_file.endswith(ext) for ext in extensions]):
                    name_list.append(image_file)
            if len(name_list) == 0:
                raise Exception('The input directory does not contain any image files')
            self.degraded_ids += [root + id_ for id_ in name_list]
        else:
            if any([root.endswith(ext) for ext in extensions]):
                name_list = [root]
            else:
                raise Exception('Please pass an Image file')
            self.degraded_ids = name_list
        print("Total Images : {}".format(name_list))

        self.num_img = len(self.degraded_ids)

    def __getitem__(self, idx):
        degraded_img = crop_img(np.array(Image.open(self.degraded_ids[idx]).convert('RGB')), base=16)
        name = self.degraded_ids[idx].split('/')[-1][:-4]

        degraded_img = self.toTensor(degraded_img)

        return [name], degraded_img

    def __len__(self):
        return self.num_img
    















