
import time
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skvideo.measure import niqe



class AverageMeter():
    """ Computes and stores the average and current value """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res


def compute_psnr_ssim(recoverd, clean):
    assert recoverd.shape == clean.shape
    recoverd = np.clip(recoverd.detach().cpu().numpy(), 0, 1)
    clean = np.clip(clean.detach().cpu().numpy(), 0, 1)

    recoverd = recoverd.transpose(0, 2, 3, 1)
    clean = clean.transpose(0, 2, 3, 1)
    psnr = 0
    ssim = 0

    for i in range(recoverd.shape[0]):
        # psnr_val += compare_psnr(clean[i], recoverd[i])
        # ssim += compare_ssim(clean[i], recoverd[i], multichannel=True)
        psnr += peak_signal_noise_ratio(clean[i], recoverd[i], data_range=1)
        ssim += structural_similarity(clean[i], recoverd[i], data_range=1, channel_axis=-1)

    return psnr / recoverd.shape[0], ssim / recoverd.shape[0]


def compute_psnr_ssim_single(recovered, clean):
    """
    计算单张图像的 PSNR 和 SSIM
    Args:
        recovered (torch.Tensor or np.ndarray): 模型恢复图像 (C, H, W)
        clean (torch.Tensor or np.ndarray): 对应的干净图像 (C, H, W)
    Returns:
        psnr (float), ssim (float)
    """

    # 转 numpy 并归一化到 [0, 1]
    if not isinstance(recovered, np.ndarray):
        recovered = recovered.detach().cpu().numpy()
    if not isinstance(clean, np.ndarray):
        clean = clean.detach().cpu().numpy()

    recovered = np.clip(recovered, 0, 1)
    clean = np.clip(clean, 0, 1)

    # 调整维度到 (H, W, C)
    if recovered.shape[0] == 3:  # (C, H, W)
        recovered = recovered.transpose(1, 2, 0)
        clean = clean.transpose(1, 2, 0)

    # 计算指标
    psnr = peak_signal_noise_ratio(clean, recovered, data_range=1)
    ssim = structural_similarity(clean, recovered, data_range=1, multichannel=True)

    return psnr, ssim







def compute_niqe(image):
    image = np.clip(image.detach().cpu().numpy(), 0, 1)
    image = image.transpose(0, 2, 3, 1)
    niqe_val = niqe(image)

    return niqe_val.mean()

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0