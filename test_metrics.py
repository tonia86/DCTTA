# ======================================
# Image Quality Assessment Script (with FID and Multi-Dataset Support)
# ======================================

import os
import sys
import glob
import argparse
import logging
from datetime import datetime
import time

import numpy as np
import torch
from PIL import Image
import pyiqa
from basicsr.utils import img2tensor
from natsort import natsorted


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        fmt='%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
        datefmt='%y-%m-%d %H:%M:%S'
    )
    logger.setLevel(level)

    if tofile:
        log_file = os.path.join(root, f"{phase}_{get_timestamp()}.log")
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)


def dict2str(opt, indent=1):
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent * 2) + f"{k}:[\n"
            msg += dict2str(v, indent + 1)
            msg += ' ' * (indent * 2) + "]\n"
        else:
            msg += ' ' * (indent * 2) + f"{k}: {v}\n"
    return msg


def crop_img(image, base=64):
    h, w = image.shape[:2]
    crop_h = h % base
    crop_w = w % base
    return image[crop_h // 2:h - crop_h + crop_h // 2, crop_w // 2:w - crop_w + crop_w // 2, :]


def main():
    parser = argparse.ArgumentParser(description="IQA Evaluation with FID and Multi-Dataset Support")
    parser.add_argument("--log", type=str, default="./logs", help="Directory to save logs")
    parser.add_argument("--log_name", type=str, default='DFPIR_METRICS', help="Base log name")
    args = parser.parse_args()

    # 固定多数据集路径
    args.inp_imgs = [
        "/data2/tn/code/work_tta/DCTTA/results"
    ]
    args.gt_imgs = [
        "/data2/tn/data/test/Rain100H_100/GT"
    ]


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.log, exist_ok=True)

    setup_logger('base', args.log, f'test_{args.log_name}', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info("===== Configuration =====")
    logger.info(dict2str(vars(args)))
    logger.info("==========================\n")

    # 初始化指标
    logger.info("Initializing IQA metrics...")
    iqa_metrics = {
        'PSNR': pyiqa.create_metric('psnr', test_y_channel=False, color_space='rgb').to(device),
        'SSIM': pyiqa.create_metric('ssim', test_y_channel=False, color_space='rgb').to(device),
        'LPIPS': pyiqa.create_metric('lpips', device=device),
        'DISTS': pyiqa.create_metric('dists', device=device),
        'CLIPIQA': pyiqa.create_metric('clipiqa', device=device),
        'NIQE': pyiqa.create_metric('niqe', device=device),
        'MUSIQ': pyiqa.create_metric('musiq', device=device),
        'MANIQA': pyiqa.create_metric('maniqa-pipal', device=device),
    }
    fid_metric = pyiqa.create_metric('fid', device=device)
    logger.info("Metrics initialized.\n")

    overall_metrics = {k: [] for k in list(iqa_metrics.keys()) + ['FID']}

    # ===== 逐数据集评估 =====
    for dir_idx, init_dir in enumerate(args.inp_imgs):
        gt_dir = args.gt_imgs[dir_idx]
        dir_name = os.path.basename(os.path.normpath(init_dir))

        img_gt_list = natsorted(glob.glob(os.path.join(gt_dir, '*.[pj]*[np]*[g]*')))
        img_sr_list = natsorted(glob.glob(os.path.join(init_dir, '*.[pj]*[np]*[g]*')))

        if len(img_gt_list) == 0 or len(img_sr_list) == 0:
            logger.warning(f"[{dir_name}] 没有有效图像，跳过。")
            continue

        logger.info(f"\n===== Evaluating [{dir_name}] ({len(img_gt_list)} pairs) =====")
        metrics_accum = {metric: 0.0 for metric in iqa_metrics.keys()}

        for img_idx, (sr_path, gt_path) in enumerate(zip(img_sr_list, img_gt_list)):
            degraded_img = crop_img(np.array(Image.open(sr_path).convert('RGB')), base=16)
            clean_img = crop_img(np.array(Image.open(gt_path).convert('RGB')), base=16)

            sr_tensor = img2tensor(degraded_img, bgr2rgb=True, float32=True).unsqueeze(0).to(device) / 255.0
            gt_tensor = img2tensor(clean_img, bgr2rgb=True, float32=True).unsqueeze(0).to(device) / 255.0

            with torch.no_grad():
                metrics = {}
                for name, metric in iqa_metrics.items():
                    if name in ['CLIPIQA', 'NIQE', 'MUSIQ', 'MANIQA']:
                        metrics[name] = metric(sr_tensor).item()
                    else:
                        metrics[name] = metric(sr_tensor, gt_tensor).item()

            for name in metrics_accum:
                metrics_accum[name] += metrics[name]

        # 平均值
        num_images = len(img_sr_list)
        avg_metrics = {k: round(v / num_images, 4) for k, v in metrics_accum.items()}

        # FID
        # fid_value = fid_metric(gt_dir, init_dir).item()
        # avg_metrics['FID'] = round(fid_value, 4)

        # 保存结果
        for k, v in avg_metrics.items():
            overall_metrics[k].append(v)

        # 日志输出
        metrics_output = " | ".join([f"/{v:.4f}" for k, v in avg_metrics.items()])
        logger.info(f"Average for [{dir_name}] → {metrics_output}")

    # ===== 全局平均 =====
    overall_avg = {k: round(sum(v) / len(v), 4) for k, v in overall_metrics.items() if len(v) > 0}
    logger.info("\n===== Overall Average Across All Datasets =====")
    for k, v in overall_avg.items():
        logger.info(f"{k}: {v:.4f}")
    logger.info("===============================================")


if __name__ == "__main__":
    main()
