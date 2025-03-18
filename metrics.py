import os
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim
from PIL import Image


def compute_psnr_ssim(tensor1, tensor2):
    """
    计算两个 BCHW Tensor 的 PSNR 和 SSIM。

    Args:
        tensor1 (torch.Tensor): BCHW 格式的第一个 Tensor。
        tensor2 (torch.Tensor): BCHW 格式的第二个 Tensor。

    Returns:
        dict: 包含每张图片 PSNR 和 SSIM 的结果，以及平均 PSNR 和 SSIM。
    """
    assert tensor1.shape == tensor2.shape, "Input tensors must have the same shape."
    assert tensor1.ndim == 4, "Input tensors must be in BCHW format."

    psnr_values = []
    ssim_values = []

    # 遍历 Batch 维度
    for i in range(tensor1.shape[0]):  # B 次
        img1 = tensor1[i].permute(1, 2, 0).detach().cpu().numpy()  # 转换为 HWC 格式
        img2 = tensor2[i].permute(1, 2, 0).detach().cpu().numpy()  # 转换为 HWC 格式

        # 计算 PSNR 和 SSIM
        psnr = calculate_psnr(img1, img2, data_range=1.0)
        ssim = calculate_ssim(img1, img2, channel_axis=-1, data_range=1.0)

        psnr_values.append(psnr)
        ssim_values.append(ssim)

    return psnr_values, ssim_values


def psnr_ssim_for_folder(corrected_folder, gt_folder):
    """
    计算文件夹中所有序列的所有图像与 GT 图像的 PSNR 和 SSIM。

    Args:
        corrected_folder (str): 矫正结果保存的文件夹路径（包含所有序列）。
        gt_folder (str): GT 图像文件夹路径。

    Returns:
        dict: 每张图像的 PSNR 和 SSIM 结果，结构为 {image_name: (psnr, ssim)}。
    """
    psnr_dict = []
    ssim_dict = []
    output_file = os.path.join(corrected_folder, 'psnr_ssim.txt')

    # 遍历矫正结果文件夹中的每个序列文件夹
    with open(output_file, 'w') as f:
        for seq_folder in os.listdir(corrected_folder):
            seq_folder_path = os.path.join(corrected_folder, seq_folder)

            # 确保是文件夹
            if os.path.isdir(seq_folder_path):
                # 获取该序列的所有图片文件
                corrected_images = sorted(os.listdir(seq_folder_path))

                # 遍历该序列中的所有矫正结果与GT图像进行比较
                for corrected_img in corrected_images:
                    # 获取对应的 GT 图像路径
                    gt_img_path = os.path.join(gt_folder, f"{seq_folder}.jpg")  # 生成对应GT图像的路径

                    # 检查是否存在对应的GT图像
                    if os.path.exists(gt_img_path):
                        # 获取图片路径
                        corrected_img_path = os.path.join(seq_folder_path, corrected_img)

                        # 读取图片
                        corrected_img_data = np.array(Image.open(corrected_img_path))
                        gt_img_data = np.array(Image.open(gt_img_path))

                        # 计算 PSNR 和 SSIM
                        psnr = calculate_psnr(corrected_img_data, gt_img_data, data_range=1.0)
                        ssim = calculate_ssim(corrected_img_data, gt_img_data, channel_axis=-1, data_range=1.0)
                        f.write(f"{corrected_img}, {psnr:.4f}, {ssim:.4f}\n")
                        # 保存结果
                        psnr_dict.append(psnr)
                        ssim_dict.append(ssim)
                    else:
                        print(f"GT image for {corrected_img} not found!")

    avg_psnr = np.mean(psnr_dict)
    avg_ssim = np.mean(ssim_dict)
    return avg_psnr, avg_ssim


def psnr_ssim(saved_folder, gt_folder):
    """
    计算保存文件夹和GT文件夹中图像的 PSNR 和 SSIM 值。

    Args:
        saved_folder (str): 模型生成结果保存的文件夹路径（包含矫正和融合结果）。
        gt_folder (str): GT 图像文件夹路径。

    Returns:
        dict: 每张图像的 PSNR 和 SSIM 结果。
    """
    psnr_dict = []
    ssim_dict = []
    output_file = os.path.join(saved_folder, 'psnr_ssim.txt')

    # 遍历矫正结果文件夹中的每个序列文件夹
    with open(output_file, 'w') as f:
        # 获取保存结果和GT文件夹中的所有图像文件
        saved_files = os.listdir(saved_folder)

        # 遍历每一对图片，计算PSNR和SSIM
        for saved_file in saved_files:

            saved_img_path = os.path.join(saved_folder, saved_file)
            gt_img_path = os.path.join(gt_folder, saved_file)

            if not os.path.exists(gt_img_path):
                continue

            # 读取图片
            saved_img = np.array(Image.open(saved_img_path))
            gt_img = np.array(Image.open(gt_img_path))

            # 计算PSNR和SSIM
            psnr = calculate_psnr(saved_img, gt_img, data_range=1.0)
            ssim = calculate_ssim(saved_img, gt_img, channel_axis=-1, data_range=1.0)
            f.write(f"{saved_file}, {psnr:.4f}, {ssim:.4f}\n")
            # 保存结果
            psnr_dict.append(psnr)
            ssim_dict.append(ssim)

    avg_psnr = np.mean(psnr_dict)
    avg_ssim = np.mean(ssim_dict)
    return avg_psnr, avg_ssim
