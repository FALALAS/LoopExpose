import torch
import torch.nn as nn


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size, _, h_x, w_x = x.size()

        count_h = max(x[:, :, 1:, :].numel(), 1)  # 避免除零
        count_w = max(x[:, :, :, 1:].numel(), 1)

        h_tv = torch.norm(x[:, :, 1:, :] - x[:, :, :-1, :], p=2) ** 2
        w_tv = torch.norm(x[:, :, :, 1:] - x[:, :, :, :-1], p=2) ** 2

        return self.TVLoss_weight * (h_tv / count_h + w_tv / count_w)

'''
B, C, H, W = 4, 3, 256, 256  # Batch=4, 3通道, 256x256
input_tensor = torch.rand(B, C, H, W, requires_grad=True)

# 计算 TV Loss
tv_loss_fn = TVLoss(TVLoss_weight=1)
loss = tv_loss_fn(input_tensor)
print("Total Variation Loss:", loss.item())
'''

