import torch
import torch.nn as nn


class ColorLoss(nn.Module):
    """
    同时包含:
    1) Single-Image Color Constancy (针对 single_images)
    2) Cross-Image Color Consistency (针对 cross_images)

    输入:
      - single_images: (B1, 3, H1, W1)，只做单图像内部约束
      - cross_images:  (B2, 3, H2, W2)，只做多图像(跨图像)一致性约束

    参数:
      - w_single: 单图损失的权重
      - w_cross:  跨图损失的权重
    """

    def __init__(self, w_single=1.0, w_cross=1.0):
        super(ColorLoss, self).__init__()
        self.w_single = w_single
        self.w_cross = w_cross

    def _single_image_cc_loss(self, images: torch.Tensor) -> torch.Tensor:
        """
        对单批次 (B,3,H,W) 图像做单图像色彩恒常性:
        在 (H,W) 上求三通道平均 -> 做 (R,G), (R,B), (G,B) 的差平方和
        """
        B, C, H, W = images.shape
        if C != 3:
            raise ValueError(f"[single_image_cc_loss] 只支持RGB 3通道, 但当前C={C}")

        # 在 (H, W) 上做均值 => (B,3)
        means = images.mean(dim=[2, 3])  # (B, 3)

        # 对 (R,G), (R,B), (G,B) 三对通道做差
        pair_indices = [(0, 1), (0, 2), (1, 2)]
        loss_val = 0.0
        for (p, q) in pair_indices:
            diff = means[:, p] - means[:, q]  # shape (B,)
            loss_val += (diff ** 2).mean()  # 在 B 上做均值

        return loss_val

    def _cross_image_cc_loss(self, images: torch.Tensor) -> torch.Tensor:
        """
        对一批图像 (B,3,H,W) 做多图像(跨图)一致性约束:
        - 假设 B 张图像是同一场景或需要风格保持一致
        - 两两比较它们的通道均值差，让其相近

        若 B < 2，则无法两两比较，返回0.
        """
        B, C, H, W = images.shape
        if C != 3:
            raise ValueError(f"[cross_image_consistency_loss] 只支持RGB 3通道, 但当前C={C}")
        if B < 2:
            return torch.tensor(0.0, device=images.device)

        # 在 (H,W) 做均值 => (B,3)
        means = images.mean(dim=[2, 3])  # shape (B,3)

        cross_loss = 0.0
        num_pairs = 0
        # 遍历 i<j
        for i in range(B):
            for j in range(i + 1, B):
                diff_ij = means[i, :] - means[j, :]  # shape (3,)
                cross_loss += (diff_ij ** 2).mean()  # 通道上做平均
                num_pairs += 1
        
        
        if num_pairs > 0:
            cross_loss = cross_loss / num_pairs
        
        return cross_loss

    def forward(self, single_images: torch.Tensor, cross_images: torch.Tensor) -> torch.Tensor:
        """
        :param single_images: (B1, 3, H1, W1)，只做单图像CC
        :param cross_images:  (B2, 3, H2, W2)，只做多图像一致性
        :return: scalar，总损失
        """
        # 1) 单图像色彩恒常性
        loss_single = self._single_image_cc_loss(single_images) if single_images.numel() > 0 else 0.0

        # 2) 多图像(跨图)一致性
        loss_cross = self._cross_image_cc_loss(cross_images) if cross_images.numel() > 0 else 0.0

        # 3) 合并
        loss = self.w_single * loss_single + self.w_cross * loss_cross
        return loss
