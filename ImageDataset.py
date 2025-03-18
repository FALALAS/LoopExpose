import os
import functools
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import re

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
	Args:
		filename (string): path to a file
		extensions (iterable of strings): extensions to consider (lowercase)
	Returns:
		bool: True if the filename ends with one of given extensions
	"""
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def sort_filenames_by_brightness(filenames):
    """
    按照亮度级别（N1.5, N1, 0, P1, P1.5）排序文件名
    :param filenames: 未排序的文件名列表
    :return: 按正确亮度顺序排列的文件名列表
    """
    brightness_order = ["N1.5", "N1", "0", "P1", "P1.5"]
    file_list = [None] * 5  # 初始化一个长度为5的列表

    for fname in filenames:
        name, ext = os.path.splitext(fname)
        parts = name.split("_")  # 以 "_" 分割文件名
        for i, b in enumerate(brightness_order):
            if b in parts:  # 如果文件名中包含这个亮度标识
                file_list[i] = fname
                break

    if None in file_list:
        raise ValueError("排序失败，某些亮度级别的图片缺失！")

    return file_list


def image_seq_loader(img_seq_dir):
    img_seq_dir = os.path.expanduser(img_seq_dir)
    cases = []
    img_seq = []
    for root, _, fnames in sorted(os.walk(img_seq_dir)):
        fnames = sort_filenames_by_brightness(fnames)
        for fname in fnames:
            if has_file_allowed_extension(fname, IMG_EXTENSIONS):
                image_name = os.path.join(root, fname)
                cases.append(fname)
                img_seq.append(Image.open(image_name))

    return img_seq, cases


def get_default_img_seq_loader():
    return functools.partial(image_seq_loader)


class ImageSeqDataset(Dataset):
    def __init__(self, csv_file,
                 hr_img_seq_dir,
                 rgb_transform=None,
                 get_loader=get_default_img_seq_loader):
        """
        Args:
        csv_file (string): Path to the csv file with annotations.
        hr_img_seq_dir (string): Directory with all the high resolution image sequences.
        transform (callable, optional): transform to be applied on a sample.
        """
        self.seqs = pd.read_csv(csv_file, sep='/n', header=None)
        self.hr_root = hr_img_seq_dir
        self.rgb_transform = rgb_transform
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
        index (int): Index
        Returns:
        samples: a Tensor that represents a video segment.
        """
        hr_seq_dir = os.path.join(self.hr_root, self.seqs.iloc[index, 0])
        I, _ = self.loader(hr_seq_dir)
        I_rgb = self.rgb_transform(I)
        I_rgb = torch.stack(I_rgb, 0).contiguous()
        # I_hr = self._reorderBylum(I_hr)
        # I_lr = self._reorderBylum(I_lr)
        # I_hr = torch.cat([I_hr[0, :], I_hr[I_hr.shape[0] - 1, :]], 0).view(2, 3, I_hr.shape[2], I_hr.shape[3])
        # I_lr = torch.cat([I_lr[0, :], I_lr[I_lr.shape[0] - 1, :]], 0).view(2, 3, I_lr.shape[2], I_lr.shape[3])
        sample = {'I_rgb': I_rgb}
        return sample

    def __len__(self):
        return len(self.seqs)

    @staticmethod
    def _reorderBylum(seq):
        I = torch.sum(torch.sum(torch.sum(seq, 1), 1), 1)
        _, index = torch.sort(I)
        result = seq[index, :]
        return result


class ImageSeqDatasetEval(Dataset):
    def __init__(self, csv_file,
                 hr_img_seq_dir,
                 gt_img_dir,
                 rgb_transform=None,
                 get_loader=get_default_img_seq_loader):
        """
        Args:
        csv_file (string): Path to the csv file with annotations.
        hr_img_seq_dir (string): Directory with all the high resolution image sequences.
        transform (callable, optional): transform to be applied on a sample.
        """
        self.seqs = pd.read_csv(csv_file, sep='/n', header=None)
        self.hr_root = hr_img_seq_dir
        self.gt_root = gt_img_dir
        self.rgb_transform = rgb_transform
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
        index (int): Index
        Returns:
        samples: a Tensor that represents a video segment.
        """
        hr_seq_dir = os.path.join(self.hr_root, self.seqs.iloc[index, 0])
        I, cases = self.loader(hr_seq_dir)
        I_rgb = self.rgb_transform(I)
        I_rgb = torch.stack(I_rgb, 0).contiguous()

        gt_seq_dir = os.path.join(self.gt_root, f"{self.seqs.iloc[index, 0]}.jpg")
        img = Image.open(gt_seq_dir).convert('RGB')  # 确保图像为 RGB 格式
        img_tensor = transforms.ToTensor()(img)  # 转换为 Tensor，形状为 (C, H, W)
        gt_rgb = img_tensor.unsqueeze(0)  # 在第 0 维添加批量维度

        # I_hr = self._reorderBylum(I_hr)
        # I_lr = self._reorderBylum(I_lr)
        # I_hr = torch.cat([I_hr[0, :], I_hr[I_hr.shape[0] - 1, :]], 0).view(2, 3, I_hr.shape[2], I_hr.shape[3])
        # I_lr = torch.cat([I_lr[0, :], I_lr[I_lr.shape[0] - 1, :]], 0).view(2, 3, I_lr.shape[2], I_lr.shape[3])
        sample = {'I_rgb': I_rgb, 'I_gt': gt_rgb, 'cases': cases, 'case': str(self.seqs.iloc[index, 0])}
        return sample

    def __len__(self):
        return len(self.seqs)

class ImageSeqDatasetMultiEval(Dataset):
    def __init__(self, csv_file,
                 hr_img_seq_dir,
                 gt_img_dirs,
                 rgb_transform=None,
                 get_loader=get_default_img_seq_loader):
        """
        Args:
        csv_file (string): Path to the csv file with annotations.
        hr_img_seq_dir (string): Directory with all the high resolution image sequences.
        gt_img_dirs (list): List of directories containing ground truth images.
        transform (callable, optional): transform to be applied on a sample.
        """
        self.seqs = pd.read_csv(csv_file, sep='/n', header=None)
        self.hr_root = hr_img_seq_dir
        self.gt_roots = [os.path.join("D:\\ECprojects\\ME\\seqMSEC\\GT_IMAGES", gt_dir) for gt_dir in gt_img_dirs]  # List of GT directories with prefix
        self.rgb_transform = rgb_transform
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
        index (int): Index
        Returns:
        samples: a Tensor that represents a video segment.
        """
        hr_seq_dir = os.path.join(self.hr_root, self.seqs.iloc[index, 0])
        I, cases = self.loader(hr_seq_dir)
        I_rgb = self.rgb_transform(I)
        I_rgb = torch.stack(I_rgb, 0).contiguous()

        # Load ground truth images from multiple directories
        gt_images = []
        for gt_root in self.gt_roots:
            gt_seq_dir = os.path.join(gt_root, f"{self.seqs.iloc[index, 0]}.jpg")
            if os.path.exists(gt_seq_dir):
                img = Image.open(gt_seq_dir).convert('RGB')  # Ensure RGB format
                img_tensor = transforms.ToTensor()(img)  # Convert to tensor (C, H, W)
                gt_images.append(img_tensor.unsqueeze(0))  # Add batch dimension

        if len(gt_images) > 0:
            I_gt = torch.cat(gt_images, dim=0)  # Stack along batch dimension
        else:
            I_gt = torch.zeros(1, 3, 256, 256)  # Placeholder (ensure proper handling)

        sample = {'I_rgb': I_rgb, 'I_gt': I_gt, 'cases': cases, 'case': str(self.seqs.iloc[index, 0])}
        return sample

    def __len__(self):
        return len(self.seqs)
