import torch
import torch.nn.functional as F

import torch.nn as nn


class ProportionalDownsampler:
    """
    等比例下采样类，保证所有图像的短边对齐到目标值，长边按比例缩放，并且最终尺寸为4的倍数。
    """

    def __init__(self, target_short_side=256):
        """
        初始化采样器。
        :param target_short_side: int, 目标短边长度
        """
        self.target_short_side = target_short_side

    def downsample(self, image_tensor):
        """
        等比例下采样一个 BCHW 格式的张量序列，并调整尺寸为4的倍数。
        :param image_tensor: torch.Tensor, 形状为 (B, C, H, W)
        :return: torch.Tensor, 下采样后的张量
        """
        B, C, H, W = image_tensor.shape

        # 计算等比例缩放的目标尺寸
        scale = self.target_short_side / min(H, W)
        new_H = int(H * scale)
        new_W = int(W * scale)

        # 调整尺寸为4的倍数
        new_H = (new_H + 3) // 4 * 4  # 向上取整到最近的4倍数
        new_W = (new_W + 3) // 4 * 4  # 向上取整到最近的4倍数

        # 使用双线性插值进行缩放
        downsampled_tensor = F.interpolate(image_tensor, size=(new_H, new_W), mode='bilinear', align_corners=False)

        return downsampled_tensor


class BasicBlock(nn.Sequential):
    r"""The basic block module (Conv+LeakyReLU[+InstanceNorm]).
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, norm=False):
        body = [nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=1), nn.LeakyReLU(0.2)]
        if norm:
            body.append(nn.InstanceNorm2d(out_channels, affine=True))
        super(BasicBlock, self).__init__(*body)


class LightBackbone(nn.Sequential):
    r"""The 5-layer CNN backbone module in [TPAMI 3D-LUT]
        (https://github.com/HuiZeng/Image-Adaptive-3DLUT).
    Args:
        input_resolution (int, optional): Resolution for pre-downsampling. Default: 256.
        extra_pooling (bool, optional): Whether to use an extra pooling layer at the end
            of the backbone. Default: False.
        n_base_feats (int, optional): Channel multiplier. Default: 8.
    """

    def __init__(self, input_resolution=256, extra_pooling=True, n_base_feats=8, **kwargs) -> None:
        body = [BasicBlock(3, n_base_feats, stride=2, norm=True)]
        n_feats = n_base_feats
        for _ in range(3):
            body.append(BasicBlock(n_feats, n_feats * 2, stride=2, norm=True))
            n_feats = n_feats * 2
        body.append(BasicBlock(n_feats, n_feats, stride=2))
        body.append(nn.Dropout(p=0.5))
        if extra_pooling:
            body.append(nn.AdaptiveAvgPool2d(2))
        super().__init__(*body)
        self.input_resolution = input_resolution
        self.out_channels = n_feats * (4 if extra_pooling else (input_resolution // 32) ** 2)

    def forward(self, imgs):
        imgs = F.interpolate(imgs, size=(self.input_resolution,) * 2, mode='bilinear', align_corners=False)
        return super().forward(imgs).view(imgs.shape[0], -1)


class MySampler(nn.Module):

    def __init__(self, sampler_input_resolution=256, sampler_output_resolution=256):
        super().__init__()
        self.n_vertices = sampler_output_resolution

        self.generator = LightBackbone(input_resolution=sampler_input_resolution, extra_pooling=True, n_base_feats=8)
        self.intervals_generator = nn.Linear(256, (self.n_vertices - 1) * 2)
        self.init_weights()

    def init_weights(self):

        def special_initilization(m):
            classname = m.__class__.__name__
            if 'Conv' in classname:
                nn.init.xavier_normal_(m.weight.data)
            elif 'InstanceNorm' in classname:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

        self.generator.apply(special_initilization)
        nn.init.zeros_(self.intervals_generator.weight)
        nn.init.ones_(self.intervals_generator.bias)

    def forward(self, img):
        code = self.generator(img)
        code = code.view(code.shape[0], -1)
        intervals = self.intervals_generator(code).view(code.shape[0], -1, self.n_vertices - 1)
        intervals = intervals.softmax(-1)
        vertieces = F.pad(intervals.cumsum(-1), (1, 0), 'constant', 0)
        grid_batch = []
        for b in range(code.shape[0]):
            xx, yy = torch.meshgrid(vertieces[b, 0, :], vertieces[b, 1, :])
            xx = xx.unsqueeze(0).unsqueeze(0)
            yy = yy.unsqueeze(0).unsqueeze(0)
            xx = xx * 2 - 1
            yy = yy * 2 - 1
            grid_batch.append(torch.cat([yy, xx], dim=1).permute(0, 2, 3, 1))
        self.vertieces = vertieces
        grid = torch.cat(grid_batch, dim=0)
        img_sampled = F.grid_sample(img, grid, mode='bilinear', align_corners=False, padding_mode='zeros')
        return img_sampled, grid


# 示例
'''
if __name__ == "__main__":
    # 假设 image_tensor 是一个形状为 (B, C, H, W) 的张量
    image_tensor = torch.randn(5, 3, 963, 600)  # 生成示例张量
    downsampler = ProportionalDownsampler(target_short_side=256)
    downsampled_tensor = downsampler.downsample(image_tensor)

    # 显示下采样后的张量尺寸
    print(f"Downsampled tensor shape: {downsampled_tensor.shape}")
'''
