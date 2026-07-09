import os
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from copy import deepcopy
import utils.utils_tta as utils_tta
import utils.utils_image as util

from utils.loss_utils import PerceptualLoss,WaveletTransform

from tqdm import tqdm
import matplotlib.pyplot as plt
# import utils.utils_blindsr_plus as blindsr_plus
# import utils.basicsr_degradations as degradations
# from utils.diffjpeg import DiffJPEG
import numpy as np
from utils.image_io import save_image_tensor
import logging
import subprocess
from itertools import cycle
from torchvision.transforms import RandomHorizontalFlip, RandomRotation
import torch.nn as nn













logger = logging.getLogger(__name__)

def configure_model(opt, model,airnet=False):
    """Configure model for use with TTA."""
    model.train()

    # 仅更新 Prompt 还有encoder 相关模块
    # train_params =['prompt']

    # train_params = ['decoder','encoder']

    if airnet==0:

        train_params = ['prompt','encoder','decoder']

        # train_params = ['prompt','decoder']

        # train_params = ['prompt']

        for k, v in model.named_parameters():
            prefix = k.split('.')[0]  # 获取参数的前缀部分
            if any(tp in prefix for tp in train_params):
                logger.info('train params: {}'.format(k))
                v.requires_grad = True
            else:
                logger.info('freezing params: {}'.format(k))
                v.requires_grad = False  # 冻结其他层


    return model





# def configure_model(opt, model):
#     """Configure model for use with TTA."""
#     model.train()

#     # 需要冻结的参数
#     freeze_params = ['prompt','reduce','refinement','output']

#     for k, v in model.named_parameters():
#         prefix = k.split('.')[0]  # 获取参数的前缀部分
#         if any(fp in prefix for fp in freeze_params):
#             logger.info('freezing params: {}'.format(k))
#             v.requires_grad = False  # 冻结 `prompt` 和 `chnl_reduce`
#         else:
#             logger.info('train params: {}'.format(k))
#             v.requires_grad = True  # 训练其他层

#     return model




def collect_params(model):
    """Collect all trainable parameters.

    Walk the model's modules and collect all parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if True: #isinstance(m, nn.BatchNorm2d): collect all 
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
                    # print(nm, np)
    return params, names



# def collect_params(model):
#     params = []
#     names = []
#     for name, p in model.named_parameters():
#         if name in ['weight', 'bias'] or name.endswith('.weight') or name.endswith('.bias'):
#             if p.requires_grad:
#                 params.append(p)
#                 names.append(name)
#     return params, names







def compute_loss(pred, target, eps=1e-3):
    """ L1 Charbonnier loss """
    return torch.sqrt(((pred - target)**2) + eps).mean()









def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

def create_ema_model(model):
    """Copy the model and optimizer states for resetting after adaptation."""
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return ema_model





class SRTTA():
    def __init__(self, opt, model, optimizer, fisher=None, logger=None,airnet=False):
        self.opt = opt
        self.model = model

        # self.model_teacher = teacher_model


        self.fisher = fisher
        self.optimizer = optimizer
        ####################
        self.G_lossfn_weight = 1.0
        self.P_lossfn_weight = 0.01
        self.RL_lossfn_weight = 0.1
        self.device = torch.device("cuda")
        self.airnet = airnet


        # self.text_code = text_code #####################


        self.E_decay = 0.45 #oral=0.45
        # print('=======================')
        # print(self.E_decay)
        self.wt1 = WaveletTransform(1,True)
        self.wt3 = WaveletTransform(3,True)

        ###################

        self.compute_loss = compute_loss

        ###用于fisher_restoration
        self.model_state = deepcopy(model.state_dict())
        self.optimizer_state = deepcopy(optimizer.state_dict())
        
        self.fishers = {}
        self.logger = logger
        self.model_teacher = create_ema_model(self.model)

    def __call__(self, train_loader, input_img, img_name,degeneration_model,text_code=None):

        # print(f"Performing TTA on {img_name}")
        
        for iteration in range(self.opt.iterations):
            # self.test_time_adaptation(train_loader,degeneration_model,iteration)

            self.test_time_adaptation(train_loader,input_img,img_name,degeneration_model,iteration,text_code)

            input_img = input_img.cuda()
            
            ##################EMA
            if self.E_decay > 0:
                teacher_params = dict(self.model_teacher.named_parameters())
                student_params = dict(self.model.named_parameters())
                for k in teacher_params.keys():
                    teacher_params[k].data.mul_(self.E_decay).add_((student_params[k]), 
                                                                              alpha=1-self.E_decay)
            ##################EMA

            self.model.eval()
            with torch.no_grad():
                restored = self.model(input_img,text_code)
            
            # with torch.no_grad():
            #     psnr, ssim = utils_tta.test_one(self.args, self.model, input_img, img_gt, sr_img=sr_img)
            
            # if iteration == self.args.iterations - 1:
            #     logger.info(f"Adapted PSNR/SSIM: {psnr:.3f}/{ssim:.4f} on {img_name} with {iteration} iterations")
        
        return restored



    
    def test_time_adaptation(self,train_loader, degrad_patch,de_id, degeneration_model,iteration,text_code):

        total_loss = 0
        self.model.train()
        self.optimizer.zero_grad()
        
        RPF = RandomHorizontalFlip(p=1)
        R90 = RandomRotation((90,90))
        R180 = RandomRotation((180,180))
        R270 = RandomRotation((270,270))
        Rn90 = RandomRotation((-90,-90))
        Rn180 = RandomRotation((-180,-180))
        Rn270 = RandomRotation((-270,-270))


        if self.airnet==1:
            ###############################原始
            print(1111111111111111111111111111111111111111111111111111111111111111)
            with torch.no_grad():
                pslabel_list = []
                for _ in range(1):
                    pslabel1 = RPF(self.model_teacher(RPF(degrad_patch))[0])
                    pslabel2 = RPF(Rn90(self.model_teacher(R90(RPF(degrad_patch)))[0]))
                    pslabel3 = RPF(Rn180(self.model_teacher(R180(RPF(degrad_patch)))[0]))
                    pslabel4 = RPF(Rn270(self.model_teacher(R270(RPF(degrad_patch)))[0]))
                    pslabel5 = self.model_teacher(degrad_patch)[0]
                    pslabel6 = Rn90(self.model_teacher(R90(degrad_patch))[0])
                    pslabel7 = Rn180(self.model_teacher(R180(degrad_patch))[0])
                    pslabel8 = Rn270(self.model_teacher(R270(degrad_patch))[0])
                    pslabel_list.append((pslabel1+pslabel2+pslabel3+pslabel4+pslabel5+pslabel6+pslabel7+pslabel8) / 8.)

                stacked_pslabel =torch.stack(pslabel_list)
                out_tea_gt = torch.mean(stacked_pslabel, dim=0) 
                var = torch.var(stacked_pslabel, unbiased=False, dim=0)
                confidence = 3/2 - torch.sigmoid(var/ 0.0004)

            second_lq = degeneration_model.train(degrad_patch,out_tea_gt,de_id)
            with torch.no_grad():
                out_student_gt = self.model(degrad_patch)[0]
            out_student = self.model(second_lq)[0]


        elif self.airnet==0:
            ###############################原始
            with torch.no_grad():
                pslabel_list = []
                for _ in range(4):
                    # print("DEBUG degrad_patch type:", type(degrad_patch))
                    tmp = self.model_teacher(RPF(degrad_patch))
                    # print("DEBUG after RPF:", type(tmp))
                    # print(tmp.size())

                    pslabel1 = RPF(self.model_teacher(RPF(degrad_patch)))
                    pslabel2 = RPF(Rn90(self.model_teacher(R90(RPF(degrad_patch)))))
                    pslabel3 = RPF(Rn180(self.model_teacher(R180(RPF(degrad_patch)))))
                    pslabel4 = RPF(Rn270(self.model_teacher(R270(RPF(degrad_patch)))))
                    pslabel5 = self.model_teacher(degrad_patch)
                    pslabel6 = Rn90(self.model_teacher(R90(degrad_patch)))
                    pslabel7 = Rn180(self.model_teacher(R180(degrad_patch)))
                    pslabel8 = Rn270(self.model_teacher(R270(degrad_patch)))
                    pslabel_list.append((pslabel1+pslabel2+pslabel3+pslabel4+pslabel5+pslabel6+pslabel7+pslabel8) / 8.)

                stacked_pslabel =torch.stack(pslabel_list)
                out_tea_gt = torch.mean(stacked_pslabel, dim=0) 
                var = torch.var(stacked_pslabel, unbiased=False, dim=0)
                confidence = 3/2 - torch.sigmoid(var/ 0.0004)
                # print("!!!!!!!!296",confidence)

            second_lq = degeneration_model.train(degrad_patch,out_tea_gt,de_id)


            with torch.no_grad():
                out_student_gt = self.model(degrad_patch)
            out_student = self.model(second_lq)

        else:

            with torch.no_grad():
                pslabel_list = []
                for _ in range(1):
                    # print("DEBUG degrad_patch type:", type(degrad_patch))
                    tmp = self.model_teacher(RPF(degrad_patch),text_code)
                    # print("DEBUG after RPF:", type(tmp))
                    # print(tmp.size())

                    pslabel1 = RPF(self.model_teacher(RPF(degrad_patch),text_code))
                    pslabel2 = RPF(Rn90(self.model_teacher(R90(RPF(degrad_patch)),text_code)))
                    pslabel3 = RPF(Rn180(self.model_teacher(R180(RPF(degrad_patch)),text_code)))
                    pslabel4 = RPF(Rn270(self.model_teacher(R270(RPF(degrad_patch)),text_code)))
                    pslabel5 = self.model_teacher(degrad_patch,text_code)
                    pslabel6 = Rn90(self.model_teacher(R90(degrad_patch),text_code))
                    pslabel7 = Rn180(self.model_teacher(R180(degrad_patch),text_code))
                    pslabel8 = Rn270(self.model_teacher(R270(degrad_patch),text_code))
                    pslabel_list.append((pslabel1+pslabel2+pslabel3+pslabel4+pslabel5+pslabel6+pslabel7+pslabel8) / 8.)

                stacked_pslabel =torch.stack(pslabel_list)
                out_tea_gt = torch.mean(stacked_pslabel, dim=0) 
                var = torch.var(stacked_pslabel, unbiased=False, dim=0)
                confidence = 3/2 - torch.sigmoid(var/ 0.0004)
                # print("!!!!!!!!296",confidence)

            second_lq = degeneration_model.train(degrad_patch,out_tea_gt,de_id)


            with torch.no_grad():
                out_student_gt = self.model(degrad_patch,text_code)
            out_student = self.model(second_lq,text_code)            


        out_student_e = self.wt1(out_student.permute(0,2,3,1))[:,0,...]
        out_student_gt_e = self.wt1(out_student_gt.permute(0,2,3,1))[:,0,...]
        out_tea_gt_e = self.wt1(out_tea_gt.permute(0,2,3,1))[:,0,...]  #torch.Size([1, 128, 128, 3])


        loss_s = self.compute_loss2(out_student, out_student_gt,out_student_e,out_student_gt_e,confidence) 


        # #################compute teacher loss
        if self.opt.teacher_weight > 0:
            loss_t = self.opt.teacher_weight * self.compute_loss2(out_student, out_tea_gt,out_student_e,out_tea_gt_e,confidence)
            # loss_t = self.opt.teacher_weight * self.compute_loss2(out_student, out_tea_gt,confidence)

        loss = loss_s + loss_t 
        logger.info(f"loss_s:{loss_s},loss_t:{loss_t}")


        self.optimizer.zero_grad()
        loss.backward()

        total_loss += loss

        self.optimizer.step()

        if self.opt.compute_fisher:
            self.fisher_restoration()
        return total_loss
    


    def fisher_restoration(self):
        """使用 Fisher 重要性恢复模型关键参数"""
        for nm, m in self.model.named_modules():
            for npp, p in m.named_parameters():
                if npp in ['weight', 'bias'] and p.requires_grad:
                    key = f"{nm}.{npp}"
                    if key not in self.fishers:
                        continue  # 跳过没有Fisher信息的参数
                    mask = self.fishers[key][-1]
                    with torch.no_grad():
                        p.data = self.model_state[key] * mask + p * (1. - mask)



    def compute_loss2(self,pred, target,E_LPF,L_LPF,cof):
    # def compute_loss2(self,pred, target,cof):
        G_lossfn = nn.L1Loss().to(self.device)

        P_lossfn = PerceptualLoss().to(self.device)

        RL_lossfn = nn.L1Loss().to(self.device)

        loss = self.G_lossfn_weight * G_lossfn(cof*pred, cof*target) + self.P_lossfn_weight * P_lossfn(pred, target) +self.RL_lossfn_weight * RL_lossfn(E_LPF,L_LPF)
        
        return loss


    def compute_fisher(self, test_loaders,airnet = 0,text_code =None):
        if len(self.fishers) > 0: 
            return self.fishers

        fishers = {}
        fisher_optimizer = optim.Adam(self.model.parameters())

        for idx, (filename,img_lr,clean_patch) in tqdm(enumerate(test_loaders,  start=1), total=len(test_loaders)):

            img_lr = img_lr.cuda()
            clean_patch = clean_patch.cuda()
            fisher_optimizer.zero_grad()

            tran_imgs, tran_ops = utils_tta.augment_transform(img_lr)
            tran_imgs.reverse() # the last item is img_lr
            tran_ops.reverse() 
            
            # compute consistent loss
            sr_imgs = []
            for idx, (tran_img, op) in enumerate(zip(tran_imgs, tran_ops), start=1):
                if idx < len(tran_imgs):
                    with torch.no_grad():
                        if airnet ==1:
                            sr_img,_,_ = self.model(tran_img)#################
                        elif airnet ==0:
                            sr_img= self.model(tran_img)#################
                        else:
                            sr_img= self.model(tran_img,text_code)#################
                        sr_img = utils_tta.transform_tensor(sr_img, op, undo=True)
                        sr_imgs.append(sr_img)
                else:
                    if airnet == 1:
                        sr_img,_,_ = self.model(tran_img)#################
                    elif airnet ==0:
                        sr_img= self.model(tran_img)#################
                    else:
                        sr_img= self.model(tran_img,text_code)#################


                    sr_imgs.append(sr_img)
            # with torch.no_grad():
            sr_pseudo = torch.cat(sr_imgs, dim=0).mean(dim=0, keepdim=True).detach()
            loss = self.compute_loss(sr_imgs[-1], sr_pseudo)
            loss.backward()

            # # computer fisher
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher = param.grad.data.clone().detach() ** 2
                    if name in fishers:
                        fishers[name] += fisher 
                    else:
                        fishers[name] = fisher 

        # computer mask based on the fisher
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                fisher = fishers[name].flatten() # TODO: check whether flatten is reverse
                _, mask_idx = torch.topk(fisher, k=int(len(fisher) * self.opt.fisher_ratio))
                mask = param.new_zeros(param.shape).flatten() # ensure the mask and p are in the save devide 
                mask[mask_idx] = 1
                mask = mask.view(param.shape)
                self.fishers.update({name: [fisher, mask]})

        # self.fishers = fishers
        fisher_optimizer.zero_grad()

        # for key in self.fishers:
        #     logger.info(f"Fisher Key: {key}")
        fisher_values = []
        # 获取 Fisher 重要性
        for name, fisher_data in self.fishers.items():
            fisher_value = fisher_data[0]  
            logger.info(f"Fisher Key: {name}, Fisher Importance: {fisher_value.mean().item():.6f}")

        return fisher




    def reset_parameters(self):
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)
    
    def resume(self, resume_path):
        if resume_path is not None:
            resume_state = torch.load(resume_path)
            load_model_and_optimizer(self.model, self.optimizer, resume_state['model'], resume_state['optimizer'])
            self.model_state = resume_state['ori_model']
            self.optimizer_state = resume_state['ori_optimizer']
            corruption = resume_state['corruption']
            iter_idx = resume_state['iter_idx']
        
        return corruption, iter_idx
    
    def save(self, iter_idx=0, ckpt_name="model_last.ckpt",dfpir = False):
        """
        保存模型权重与优化器状态。
        
        参数:
            iter_idx (int): 当前迭代步数。
            ckpt_name (str): 要保存的 ckpt 文件名，例如 'model_airnet.pt' 或 'model_adair.ckpt'。
        """


        if not dfpir:


            state = {
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'ori_model': getattr(self, 'model_state', None),
                'ori_optimizer': getattr(self, 'optimizer_state', None),
                'iter_idx': iter_idx
            }

            save_dir = self.opt.save_dir
            os.makedirs(save_dir, exist_ok=True)

            # 自动补上后缀（如果没写）
            if not ckpt_name.endswith(('.ckpt', '.pt', '.pth')):
                ckpt_name = f"{ckpt_name}.ckpt"

            save_path = os.path.join(save_dir, ckpt_name)
            torch.save(state, save_path)

            print(f" Model saved to: {save_path}")                  

        else:


            torch.save({
                'epoch': 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'ori_model': getattr(self, 'model_state', None),
                'ori_optimizer': getattr(self, 'optimizer_state', None)
                },
                os.path.join(self.opt.save_dir,'checkpoint_epoch_V1.pth.tar'))


