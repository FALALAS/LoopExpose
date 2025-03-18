import collections

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

RANDOM_RESOLUTIONS = [512, 768, 1024, 1280, 1536]


class BatchRandomResolution(object):
    def __init__(self, size=None, interpolation=Image.BILINEAR):

        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2) or (size is None)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, imgs):

        if self.size is None:
            h, w = imgs[0].size
            max_idx = 0
            for i in range(len(RANDOM_RESOLUTIONS)):
                if h > RANDOM_RESOLUTIONS[i] and w > RANDOM_RESOLUTIONS[i]:
                    max_idx += 1
            idx = np.random.randint(max_idx)
            self.size = RANDOM_RESOLUTIONS[idx]
        return [transforms.Resize(self.size, self.interpolation)(img) for img in imgs]


class BatchTestResolution(object):
    def __init__(self, size=None, interpolation=Image.BILINEAR):

        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2) or (size is None)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, imgs):

        h, w = imgs[0].size
        if h > self.size and w > self.size:
            return [transforms.Resize(self.size, self.interpolation)(img) for img in imgs]
        else:
            return imgs


class BatchToTensor(object):
    def __call__(self, imgs):
        return [transforms.ToTensor()(img) for img in imgs]


class BatchRGBToGray(object):
    def __call__(self, imgs):
        return [img[0, :, :] * 0.299 + img[1, :, :] * 0.587 + img[2:, :, :] * 0.114 for img in imgs]


class BatchRGBToYCbCr(object):
    def __call__(self, imgs):
        return [torch.stack((0. / 256. + img[0, :, :] * 0.299000 + img[1, :, :] * 0.587000 + img[2, :, :] * 0.114000,
                             128. / 256. - img[0, :, :] * 0.168736 - img[1, :, :] * 0.331264 + img[2, :, :] * 0.500000,
                             128. / 256. + img[0, :, :] * 0.500000 - img[1, :, :] * 0.418688 - img[2, :, :] * 0.081312),
                            dim=0) for img in imgs]


class YCbCrToRGB(object):
    def __call__(self, img):
        return torch.stack((img[:, 0, :, :] + (img[:, 2, :, :] - 128 / 256.) * 1.402,
                            img[:, 0, :, :] - (img[:, 1, :, :] - 128 / 256.) * 0.344136 - (
                                    img[:, 2, :, :] - 128 / 256.) * 0.714136,
                            img[:, 0, :, :] + (img[:, 1, :, :] - 128 / 256.) * 1.772),
                           dim=1)


class RGBToYCbCr(object):
    def __call__(self, img):
        return torch.stack(
            (0. / 256. + img[:, 0, :, :] * 0.299000 + img[:, 1, :, :] * 0.587000 + img[:, 2, :, :] * 0.114000,
             128. / 256. - img[:, 0, :, :] * 0.168736 - img[:, 1, :, :] * 0.331264 + img[:, 2, :, :] * 0.500000,
             128. / 256. + img[:, 0, :, :] * 0.500000 - img[:, 1, :, :] * 0.418688 - img[:, 2, :, :] * 0.081312),
            dim=1)


class TensorRandom(object):
    def __init__(self, size=None, interpolation="bilinear"):
        assert isinstance(size, int) or (isinstance(size, (tuple, list)) and len(size) == 2) or (size is None)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, imgs):
        """
        Args:
            imgs: List of tensors of shape (C, H, W) or a single tensor of shape (B, C, H, W).
        Returns:
            List of resized tensors or a resized batch tensor.
        """

        return self._resize_batch(imgs)

    def _resize_batch(self, batch):
        if self.size is None:
            h, w = batch.shape[-2:]
            max_idx = 0
            for i in range(len(RANDOM_RESOLUTIONS)):
                if h > RANDOM_RESOLUTIONS[i] and w > RANDOM_RESOLUTIONS[i]:
                    max_idx += 1
            idx = np.random.randint(max_idx)
            self.size = RANDOM_RESOLUTIONS[idx]

        return F.interpolate(batch, size=self.size, mode=self.interpolation, align_corners=False)


class TensorTest(object):
    def __init__(self, size=None, interpolation="bilinear"):
        assert isinstance(size, int) or (isinstance(size, (tuple, list)) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, imgs):
        """
        Args:
            imgs: List of tensors of shape (C, H, W) or a single tensor of shape (B, C, H, W).
        Returns:
            List of resized tensors or a resized batch tensor.
        """

        return self._resize_batch(imgs)

    def _resize_batch(self, batch):
        h, w = batch.shape[-2:]
        if h > self.size and w > self.size:
            return F.interpolate(batch, size=self.size, mode=self.interpolation, align_corners=False)
        return batch


class TensorRGBToYCbCr:
    def __call__(self, rgb):
        """
        将 RGB 张量转换为 YCbCr 张量
        Args:
            rgb: 输入张量，形状为 (B, 3, H, W)，值范围为 [0, 1]
        Returns:
            ycbcr: 转换后的 YCbCr 张量，形状为 (B, 3, H, W)
        """
        assert rgb.ndim == 4 and rgb.size(1) == 3, "Input must be a 4D tensor with shape (B, 3, H, W)"

        # 提取 R, G, B 通道
        R, G, B = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]

        # Y, Cb, Cr 的计算公式
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 0.5  # 标准化到 [0, 1]
        Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 0.5  # 标准化到 [0, 1]

        # 合并通道
        ycbcr = torch.cat([Y, Cb, Cr], dim=1)

        # 保证范围在 [0, 1]
        ycbcr = torch.clamp(ycbcr, 0.0, 1.0)

        return ycbcr
