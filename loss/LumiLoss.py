import torch
import torch.nn as nn
import torch.nn.functional as F


class BrightnessOrderLoss(nn.Module):
    def __init__(self, margin=0.0, pairwise=True):
        """
        亮度排序损失，确保亮度特征按照原始亮度顺序排列
        :param margin: 保证相邻（或任意成对）特征至少有 margin 的差异
        :param pairwise: 如果为 True，则对所有成对样本进行排序约束；否则只对相邻样本进行约束
        """
        super(BrightnessOrderLoss, self).__init__()
        self.margin = margin
        self.pairwise = pairwise

    def forward(self, features):
        """
        计算亮度排序损失
        :param features: 输入特征张量，形状为 (N, )，N 表示图像数量（例如5）
        :return: 排序损失（标量）
        """

        brightness = features
        if self.pairwise:
            # 对所有成对样本施加约束：对于任意 i<j，要求 brightness[i] + margin <= brightness[j]
            N = brightness.shape[0]
            loss = 0.0
            count = 0
            for i in range(N):
                for j in range(i + 1, N):
                    diff = brightness[i] + self.margin - brightness[j]
                    loss += F.relu(diff)
                    count += 1
            # loss = loss / count if count > 0 else loss
        else:
            # 只对相邻样本施加约束：brightness[i] + margin <= brightness[i+1]
            diffs = brightness[:-1] + self.margin - brightness[1:]
            loss = F.relu(diffs).mean()

        return loss
