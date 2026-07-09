import subprocess
from tqdm import tqdm
from torch.nn import init
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from utils.dataset_utils import PromptTrainDataset, PairedImageDataset, PromptTrainDataset_Simple
from net.model import PromptIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
from utils.image_io import save_image_tensor
import numpy as np
import wandb
from options import options as opt
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger
import tta
from copy import deepcopy
import utils.utils_tta as util
from utils.val_utils import AverageMeter, compute_psnr_ssim
import os
import torch
import logging
import random
from RDDM.net import ResidualDiffusionModel

from net.model import StudentModel



log_filename = 'train.log'
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode='w'), 
        logging.StreamHandler()  
    ]
)
logger = logging.getLogger(__name__) 



def set_seed(seed=23):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  



########损失函数

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input, target):
        return ((input - target) ** 2).mean()

class TVLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

    def forward(self, x):
        b_s, _, h_x, w_x = x.size()
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:] - x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:] - x[:,:,:,:w_x-1]),2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / b_s

class VGGLoss(nn.Module):
    def __init__(self, layers=9):
        super().__init__()
        self.mse_loss = MSELoss()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.model = models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_V1).features[:layers]
        self.model.requires_grad_(False)
        self.model.eval()
    
    def forward(self, input, target):
        batch = torch.cat([input, target], dim=0)
        feats = self.model(self.normalize(batch))
        input_feats, target_feats = feats.chunk(2, dim=0)
        return self.mse_loss(input_feats, target_feats)




class GANLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCELoss()
        self.disc_net = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1), nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.LeakyReLU(),
            nn.Conv2d(256, 1, 1), nn.Sigmoid())
        self.optimizer = torch.optim.Adam(self.disc_net.parameters(), 5e-5)
    
    def init_disc_net(self):
        for m in self.disc_net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                if m.bias is not None: m.bias.data.fill_(0)
    
    def get_loss_value(self, input, is_real):
        target_value = 1.0 if is_real else 0.0
        target_label = torch.ones_like(input) * target_value
        return self.loss(input, target_label)
    
    def forward(self, input, target):
        self.disc_net.train()
        self.optimizer.zero_grad()
        logits = self.disc_net(torch.cat((input,target)).detach())
        fake_loss = self.get_loss_value(logits[:input.size(0)], False)
        real_loss = self.get_loss_value(logits[input.size(0):], True)
        (fake_loss + real_loss).backward()
        self.optimizer.step()
        self.disc_net.eval()
        return self.get_loss_value(self.disc_net(input), True)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.LeakyReLU()
        )
        self.resblocks = nn.Sequential(
            *[ResBlock(256) for _ in range(6)]  # 6个ResBlock
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.LeakyReLU(),
            nn.Conv2d(64, 3, 3, 1, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.resblocks(x)
        x = self.decoder(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)  # 残差连接


class Degradation(nn.Module):
    def __init__(self,opt):
        super().__init__()
        self.gene_net = ResidualDiffusionModel(opt, debug=False)

    
    def train(self,input,target,name):
        return self.gene_net.train(input,target,name)



    def sample(self,input,name):
        return self.gene_net.sample(input,name)



def setup_tta(opt, model,logger):
    model = tta.configure_model(opt, model)
    params, _ = tta.collect_params(model)
    optimizer = optim.Adam(params, lr=opt.lr, betas=opt.betas)
    return tta.SRTTA(opt, model, optimizer,logger)


def main(opt):
    set_seed(23)

    os.environ["CUDA_VISIBLE_DEVICES"] = "4"

    device = torch.device("cuda")

    lq_dir = '/data2/tn/code/work_tta/DCTTA/testdata/Rain100H/LQ'
    gt_dir = '/data2/tn/code/work_tta/DCTTA/testdata/Rain100H/GT'

    trainset = PromptTrainDataset_Simple(lq_dir, gt_dir, patch_size=320)

    
    train_loader = DataLoader(
        trainset, batch_size=opt.batch_size, pin_memory=True,
        shuffle=True, drop_last=True, num_workers=opt.num_workers
    )
    logger.info(f"训练集大小: {len(train_loader)}")

    ckpt_path = "./pretrain/model.ckpt"

    # ckpt_path = "./pretrain/epoch=80.ckpt" #5task



    model = PromptIR(decoder=True)

    checkpoint = torch.load(ckpt_path, map_location="cpu")  # 加载 checkpoint
    state_dict = checkpoint["state_dict"]

    new_state_dict = {key.replace("net.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)

    origin_model = deepcopy(model)

    model = model.cuda()
    origin_model = origin_model.cuda()
    tta_model = setup_tta(opt, model,logger)
    
    
    logger.info(f"-------------保存模型--------------")
    # tta_model.save(batch_idx, ckpt_name="promptir_5task_tta.ckpt")

    logger.info(f"-------------开始推理与评估--------------")
    testset = PairedImageDataset(lq_dir, gt_dir)   
    test_loader = DataLoader(
        testset, batch_size=1, pin_memory=True,
        shuffle=False, num_workers=opt.num_workers
    )
    logger.info(f"测试集大小: {len(test_loader)}")




    tta_model.model.eval()   # 冻结模型参数
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    # 这里可以用相同的 train_loader，也可以加载新的 test_loader
    
    for batch_idx, (de_id, degrad_patch, clean_patch) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Evaluation Progress"):
        degrad_patch = degrad_patch.cuda()
        clean_patch = clean_patch.cuda()

        # 前向推理（恢复图像）
        with torch.no_grad():
            restored = tta_model.model(degrad_patch)  # 如果模型输出为tuple
        # 计算PSNR / SSIM
        psnr, ssim = compute_psnr_ssim(restored, clean_patch)
        psnr_meter.update(psnr)
        ssim_meter.update(ssim)

        # 保存恢复图像
        output_path = '/data2/tn/code/work_tta/DCTTA/results/Rain100H_PromptIR_oral/'
        subprocess.check_output(['mkdir', '-p', output_path])
        save_image_tensor(restored, output_path + de_id[0].split('.')[0] + ".png")

    logger.info(f"平均PSNR: {psnr_meter.avg:.3f}, 平均SSIM: {ssim_meter.avg:.4f}")
    print(f"平均PSNR: {psnr_meter.avg:.3f}, 平均SSIM: {ssim_meter.avg:.4f}")
    logger.info(f"-------------推理完成，结果已保存--------------")

    logger.info(f"-------------TTA完成--------------")


    

if __name__ == '__main__':
    main(opt)
