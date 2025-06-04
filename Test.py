# test.py
import argparse
import os
import glob
import cv2
import math
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import utils, transforms

from sec.co_lut_arch import CoNet
from batch_transformers import BatchTestResolution, BatchToTensor
from metrics import compute_psnr_ssim    # 如果没有可自行替换为你的实现

EPS = 1e-8
IMG_EXT = ('*.jpg', '*.png', '*.jpeg', '*.bmp', '*.tiff')


# --------------------- 工具函数 ---------------------
def list_images(folder):
    files = []
    for ext in IMG_EXT:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files)


def save_tensor_as_img(tensor, path):
    """tensor: (1,3,H,W) 0–1"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tensor = tensor.clamp_(0, 1)
    utils.save_image(tensor, path)

def imread_resize_to_tensor(path, max_side=2048):
    """
    读图、按最长边 resize、归一化并转 Tensor
    返回 (1,3,H,W) float32 in [0,1]
    """
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    h, w = rgb.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        new_size = (int(w * scale + 0.5), int(h * scale + 0.5))  # (w,h)
        rgb = cv2.resize(rgb, new_size, interpolation=cv2.INTER_AREA)

    rgb = rgb.astype(np.float32) / 255.0          # H,W,C 0-1
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1))  # C,H,W
    return tensor.unsqueeze(0)                    # 1,C,H,W

def mertens_fusion(img_tensor_list):
    """img_tensor_list: list of (1,3,H,W)"""
    imgs = [t.squeeze(0).cpu().numpy() for t in img_tensor_list]   # to C,H,W
    imgs = [(i * 255).astype(np.uint8).transpose(1, 2, 0) for i in imgs]
    bgr_imgs = [cv2.cvtColor(i, cv2.COLOR_RGB2BGR) for i in imgs]
    fused = cv2.createMergeMertens().process(bgr_imgs) * 255
    fused = cv2.cvtColor(fused, cv2.COLOR_BGR2RGB).transpose(2, 0, 1) / 255.
    return torch.from_numpy(fused).unsqueeze(0)    # (1,3,H,W)


# --------------------- 主流程 ---------------------
def main(cfg):
    device = 'cuda' if (cfg.use_cuda and torch.cuda.is_available()) else 'cpu'
    # 1. 模型
    model = CoNet().to(device)
    print(f'Load checkpoint: {cfg.ckpt}')
    state = torch.load(cfg.ckpt, map_location=device)
    model.load_state_dict(state['state_dict'] if 'state_dict' in state else state)
    model.eval()

    # 2. 变换
    tfm = transforms.Compose([
        BatchTestResolution(cfg.resize, interpolation=2)
    ])

    # 3. 模式分支
    if cfg.mode == 'correction':
        run_correction(model, tfm, cfg, device)
    else:  # fusion
        run_fusion(model, tfm, cfg, device)


# --------------------- correction 模式 ---------------------
def run_correction(model, tfm, cfg, device):
    img_files = list_images(cfg.input_dir)
    assert img_files, f'No images found in {cfg.input_dir}'
    out_dir = os.path.join(cfg.output_dir, 'corrected')
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad(), open(os.path.join(cfg.output_dir, 'metrics.txt'), 'w') as log:
        for img_path in tqdm(img_files, desc='Correction'):
            name = os.path.splitext(os.path.basename(img_path))[0]
            inp = imread_resize_to_tensor(img_path, cfg.resize).to(device)
            # inp = tfm(inp).to(device)

            pred, *_ = model(inp)
            save_tensor_as_img(pred.cpu(), os.path.join(out_dir, f'{name}.jpg'))

            # optional metrics
            if cfg.gt_dir:
                gt_path = os.path.join(cfg.gt_dir, os.path.relpath(img_path, cfg.input_dir))
                if os.path.exists(gt_path):
                    gt = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB) / 255.
                    gt = torch.from_numpy(gt.transpose(2, 0, 1)).float().unsqueeze(0)
                    psnr, ssim = compute_psnr_ssim(pred.cpu(), gt)
                    log.write(f'{name}: PSNR={psnr[0]:.4f}, SSIM={ssim[0]:.4f}\n')


# --------------------- fusion 模式 ---------------------
def run_fusion(model, tfm, cfg, device):
    scene_dirs = [d for d in sorted(os.listdir(cfg.input_dir))
                  if os.path.isdir(os.path.join(cfg.input_dir, d))]
    assert scene_dirs, f'No sub-folders found in {cfg.input_dir}'

    corrected_root = os.path.join(cfg.output_dir, 'corrected')
    fused_root = os.path.join(cfg.output_dir, 'fused')
    os.makedirs(corrected_root, exist_ok=True)
    os.makedirs(fused_root, exist_ok=True)

    with torch.no_grad(), open(os.path.join(cfg.output_dir, 'metrics.txt'), 'w') as log:
        for scene in tqdm(scene_dirs, desc='Fusion scenes'):
            scene_path = os.path.join(cfg.input_dir, scene)
            img_files = list_images(scene_path)
            if not img_files:
                continue

            corrected_tensors = []
            scene_corr_dir = os.path.join(corrected_root, scene)
            os.makedirs(scene_corr_dir, exist_ok=True)

            for img_path in img_files:
                name = os.path.splitext(os.path.basename(img_path))[0]
                inp = imread_resize_to_tensor(img_path, cfg.resize).to(device)

                pred, *_ = model(inp)
                corrected_tensors.append(pred.cpu())

                save_tensor_as_img(pred.cpu(), os.path.join(scene_corr_dir, f'{name}.jpg'))

            # 多曝光融合
            fused = mertens_fusion(corrected_tensors)
            save_tensor_as_img(fused, os.path.join(fused_root, f'{scene}.jpg'))

            # optional metrics
            if cfg.gt_dir:
                gt_scene_dir = os.path.join(cfg.gt_dir, scene)
                if os.path.isdir(gt_scene_dir):
                    gt_files = list_images(gt_scene_dir)
                    if len(gt_files) == 1:                     # 场景真值为单张
                        gt = cv2.cvtColor(cv2.imread(gt_files[0]), cv2.COLOR_BGR2RGB) / 255.
                        gt = torch.from_numpy(gt.transpose(2, 0, 1)).float().unsqueeze(0)
                        psnr, ssim = compute_psnr_ssim(fused, gt)
                        log.write(f'{scene}: PSNR={psnr[0]:.4f}, SSIM={ssim[0]:.4f}\n')


# -------------------------- CLI -------------------------- #
# Default configuration:
#   --mode        correction
#   --input_dir   D:\ECprojects\ME\MSEC\testing\INPUT_IMAGES
#   --output_dir  ./results
#   --ckpt        ./ckpts/sec-best-epoch0028.pt
#   --use_cuda    enabled

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Run exposure-correction (single-image) or exposure-fusion (multi-image) inference')

    # Processing mode: single-image correction or multi-image fusion
    parser.add_argument(
        '--mode',
        choices=['correction', 'fusion'],
        default='correction',
        help="Choose 'correction' for single-image exposure correction or 'fusion' for multi-image exposure fusion "
             "(default: correction)")

    # Input directory
    parser.add_argument(
        '--input_dir',
        default=r'D:\ECprojects\ME\MSEC\lyq\lr',
        help='Root folder containing images (default: D:\\ECprojects\\ME\\MSEC\\testing\\INPUT_IMAGES)')

    # Output directory
    parser.add_argument(
        '--output_dir',
        default='./results',
        help='Folder where results will be saved (default: ./results)')

    # Model checkpoint
    parser.add_argument(
        '--ckpt',
        default='./ckpts/sec-best-epoch0028.pt',
        help='Path to a trained CoNet checkpoint (.pt) (default: ./ckpts/sec-best-epoch0028.pt)')

    # Longest side after resize
    parser.add_argument(
        '--resize',
        type=int,
        default=2048,
        help='Resize the longest side of each input image to this value; set to 0 to disable resizing (default: 2048)')

    # CUDA flags: --use_cuda / --no_cuda
    parser.add_argument(
        '--use_cuda',
        dest='use_cuda',
        action='store_true',
        help='Enable GPU inference (default: enabled)')
    parser.add_argument(
        '--no_cuda',
        dest='use_cuda',
        action='store_false',
        help='Disable GPU inference')
    parser.set_defaults(use_cuda=True)

    # Optional ground-truth directory for metrics
    parser.add_argument(
        '--gt_dir',
        default='',
        help='Folder with ground-truth images for PSNR/SSIM evaluation (optional)')

    cfg = parser.parse_args()
    main(cfg)
