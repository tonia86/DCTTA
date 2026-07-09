import os
import sys

from src.denoising_diffusion_pytorch import GaussianDiffusion
from src.residual_denoising_diffusion_pytorch import (ResidualDiffusion,
                                                      Trainer, Unet, UnetRes,
                                                      set_seed)

# init
os.environ['CUDA_VISIBLE_DEVICES'] = '6'


sys.stdout.flush()
set_seed(10)
debug = False

if debug:
    save_and_sample_every = 2
    sampling_timesteps = 10
    sampling_timesteps_original_ddim_ddpm = 10
    train_num_steps = 200
else:
    save_and_sample_every = 50##########

    sampling_timesteps = 5  #############
    sampling_timesteps_original_ddim_ddpm = 250
    train_num_steps = 1000 ########

original_ddim_ddpm = False
if original_ddim_ddpm:
    condition = False
    input_condition = False
    input_condition_mask = False
else:
    condition = True
    input_condition = False
    input_condition_mask = False



#########
folder = ["../../../DATA/HSTS/LQ",
            "../../../DATA/HSTS/GT",
            "../../../DATA/HSTS/LQ",
            "../../../DATA/HSTS/GT"]
train_batch_size = 1
num_samples = 1
sum_scale = 0.01
image_size = 256

#####

num_unet = 1
objective = 'pred_res'
test_res_or_noise = "res"



model = UnetRes(
    # dim=64,
    dim=64,
    dim_mults=(1, 2, 4, 8),
    # dim_mults=(1, 2, 4, 8),
    num_unet=num_unet,
    condition=condition,
    input_condition=input_condition,
    objective=objective,
    test_res_or_noise = test_res_or_noise
)
diffusion = ResidualDiffusion(
    model,
    image_size=image_size,
    timesteps=1000,           # number of steps
    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    sampling_timesteps=sampling_timesteps,
    objective=objective,
    loss_type='l2',            # L1 or L2
    condition=condition,
    sum_scale=sum_scale,
    input_condition=input_condition,
    input_condition_mask=input_condition_mask,
    test_res_or_noise = test_res_or_noise
)

trainer = Trainer(
    diffusion,
    folder,
    train_batch_size=train_batch_size,
    num_samples=num_samples,
    train_lr=2e-4,
    train_num_steps=train_num_steps,         # total training steps
    gradient_accumulate_every=2,    # gradient accumulation steps
    ema_decay=0.995,                # exponential moving average decay
    amp=False,                        # turn on mixed precision
    convert_image_to="RGB",
    condition=condition,
    save_and_sample_every=save_and_sample_every,
    equalizeHist=False,
    crop_patch=False,
    generation=True,
    num_unet=num_unet,
)


trainer.load('./results/sample/model.pt')
trainer.set_results_folder(
    './results/test_timestep_'+str(sampling_timesteps))
trainer.test(last=True)


