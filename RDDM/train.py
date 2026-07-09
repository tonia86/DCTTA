# import os
# import sys

# from src.denoising_diffusion_pytorch import GaussianDiffusion
# from src.residual_denoising_diffusion_pytorch import (ResidualDiffusion,
#                                                       Trainer, Unet, UnetRes,
#                                                       set_seed)


# def rddm():
#     # init
#     os.environ['CUDA_VISIBLE_DEVICES'] = '6'


#     sys.stdout.flush()
#     set_seed(10)
#     debug = False

#     if debug:
#         save_and_sample_every = 2
#         sampling_timesteps = 10
#         sampling_timesteps_original_ddim_ddpm = 10
#         train_num_steps = 200
#     else:
#         save_and_sample_every = 50##########

#         sampling_timesteps = 5  #############
#         sampling_timesteps_original_ddim_ddpm = 250
#         train_num_steps = 1000 ########

#     original_ddim_ddpm = False
#     if original_ddim_ddpm:
#         condition = False
#         input_condition = False
#         input_condition_mask = False
#     else:
#         condition = True
#         input_condition = False
#         input_condition_mask = False



#     #########
#     folder = ["../../../DATA/HSTS/LQ",
#                 "../../../DATA/HSTS/GT",
#                 "../../../DATA/HSTS/LQ",
#                 "../../../DATA/HSTS/GT"]
#     train_batch_size = 1
#     num_samples = 1
#     sum_scale = 0.01
#     image_size = 256

#     #####

#     num_unet = 1
#     objective = 'pred_res'
#     test_res_or_noise = "res"



#     model = UnetRes(
#         # dim=64,
#         dim=64,
#         dim_mults=(1, 2, 4, 8),
#         # dim_mults=(1, 2, 4, 8),
#         num_unet=num_unet,
#         condition=condition,
#         input_condition=input_condition,
#         objective=objective,
#         test_res_or_noise = test_res_or_noise
#     )
#     diffusion = ResidualDiffusion(
#         model,
#         image_size=image_size,
#         timesteps=1000,           # number of steps
#         # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
#         sampling_timesteps=sampling_timesteps,
#         objective=objective,
#         loss_type='l2',            # L1 or L2
#         condition=condition,
#         sum_scale=sum_scale,
#         input_condition=input_condition,
#         input_condition_mask=input_condition_mask,
#         test_res_or_noise = test_res_or_noise
#     )

#     trainer = Trainer(
#         diffusion,
#         folder,
#         train_batch_size=train_batch_size,
#         num_samples=num_samples,
#         train_lr=2e-4,
#         train_num_steps=train_num_steps,         # total training steps
#         gradient_accumulate_every=2,    # gradient accumulation steps
#         ema_decay=0.995,                # exponential moving average decay
#         amp=False,                        # turn on mixed precision
#         convert_image_to="RGB",
#         condition=condition,
#         save_and_sample_every=save_and_sample_every,
#         equalizeHist=False,
#         crop_patch=False,
#         generation=True,
#         num_unet=num_unet,
#     )


#     # train
#     trainer.train()

#     # test
#     if not trainer.accelerator.is_local_main_process:
#         pass
#     else:
#         trainer.load(trainer.train_num_steps//save_and_sample_every)
#         trainer.set_results_folder(
#             './results/test_timestep_'+str(sampling_timesteps))
#         trainer.test(last=True)

#     # trainer.set_results_folder('./results/test_sample')
#     # trainer.test(sample=True)


# rddm()







import os
import sys

from src.denoising_diffusion_pytorch import GaussianDiffusion
from src.residual_denoising_diffusion_pytorch import (ResidualDiffusion,
                                                      Trainer, Unet, UnetRes,
                                                      set_seed)


class ResidualDiffusionModel:
    def __init__(self, data_folder, image_size=256, train_batch_size=1, num_samples=1,
                 sum_scale=0.01, num_unet=1, objective='pred_res', test_res_or_noise="res",
                 debug=False, cuda_device='6'):
        """
        初始化残差扩散模型类

        :param data_folder: 数据文件夹路径，包含LQ和GT图像
        :param image_size: 输入图像大小
        :param train_batch_size: 训练批次大小
        :param num_samples: 采样数量
        :param sum_scale: 残差缩放比例
        :param num_unet: U-Net模型数量
        :param objective: 目标类型，'pred_res'表示预测残差
        :param test_res_or_noise: 测试时预测残差或噪声
        :param debug: 调试模式，控制训练步数和采样步数
        :param cuda_device: 使用的CUDA设备编号
        """
        # 设置CUDA设备
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device

        # 设置随机种子
        set_seed(10)

        # 调试模式
        self.debug = debug

        # 设置训练参数
        if self.debug:
            self.save_and_sample_every = 2
            self.sampling_timesteps = 10
            self.sampling_timesteps_original_ddim_ddpm = 10
            self.train_num_steps = 200
        else:
            self.save_and_sample_every = 50
            self.sampling_timesteps = 5
            self.sampling_timesteps_original_ddim_ddpm = 250
            self.train_num_steps = 1000

        # 设置是否使用原始DDIM/DDPM
        self.original_ddim_ddpm = False
        if self.original_ddim_ddpm:
            self.condition = False
            self.input_condition = False
            self.input_condition_mask = False
        else:
            self.condition = True
            self.input_condition = False
            self.input_condition_mask = False

        # 数据文件夹
        self.folder = data_folder

        # 模型参数
        self.image_size = image_size
        self.train_batch_size = train_batch_size
        self.num_samples = num_samples
        self.sum_scale = sum_scale
        self.num_unet = num_unet
        self.objective = objective
        self.test_res_or_noise = test_res_or_noise

        # 初始化模型和训练器
        self.model = None
        self.diffusion = None
        self.trainer = None

    def _init_model(self):
        """
        初始化模型和扩散过程
        """
        # 定义U-Net模型
        self.model = UnetRes(
            dim=64,
            dim_mults=(1, 2, 4, 8),
            num_unet=self.num_unet,
            condition=self.condition,
            input_condition=self.input_condition,
            objective=self.objective,
            test_res_or_noise=self.test_res_or_noise
        )

        # 定义扩散过程
        self.diffusion = ResidualDiffusion(
            self.model,
            image_size=self.image_size,
            timesteps=1000,
            sampling_timesteps=self.sampling_timesteps,
            objective=self.objective,
            loss_type='l2',
            condition=self.condition,
            sum_scale=self.sum_scale,
            input_condition=self.input_condition,
            input_condition_mask=self.input_condition_mask,
            test_res_or_noise=self.test_res_or_noise
        )

        # 定义训练器
        self.trainer = Trainer(
            self.diffusion,
            self.folder,
            train_batch_size=self.train_batch_size,
            num_samples=self.num_samples,
            train_lr=2e-4,
            train_num_steps=self.train_num_steps,
            gradient_accumulate_every=2,
            ema_decay=0.995,
            amp=False,
            convert_image_to="RGB",
            condition=self.condition,
            save_and_sample_every=self.save_and_sample_every,
            equalizeHist=False,
            crop_patch=False,
            generation=True,
            num_unet=self.num_unet,
        )

    def train(self):
        """
        训练模型
        """
        self._init_model()
        self.trainer.train()

    def test(self, results_folder='./results/test'):
        """
        测试模型

        :param results_folder: 测试结果保存文件夹
        """
        if not self.trainer.accelerator.is_local_main_process:
            pass
        else:
            # 加载最新模型
            self.trainer.load(self.train_num_steps // self.save_and_sample_every)
            # 设置结果文件夹
            self.trainer.set_results_folder(results_folder)
            # 进行测试
            self.trainer.test(last=True)

    def sample(self, results_folder='./results/sample'):
        """
        采样生成图像

        :param results_folder: 采样结果保存文件夹
        """
        if not self.trainer.accelerator.is_local_main_process:
            pass
        else:
            # 设置结果文件夹
            self.trainer.set_results_folder(results_folder)
            # 进行采样
            self.trainer.test(sample=True)


# 使用示例
if __name__ == "__main__":
    # 数据文件夹路径
    data_folder = ["../../../DATA/HSTS/LQ",
                   "../../../DATA/HSTS/GT",
                   "../../../DATA/HSTS/LQ",
                   "../../../DATA/HSTS/GT"]

    # 创建模型实例
    model = ResidualDiffusionModel(data_folder, debug=False)

    # 训练模型
    model.train()

    # 测试模型
    model.test(results_folder='./results/test_timestep_5')

