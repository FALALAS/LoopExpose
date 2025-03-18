import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models import VGG19_Weights


class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False


class PerceptualLoss(torch.nn.Module):
    def __init__(self, rank=0, downscale_factor=2):  # 默认缩小2倍
        super(PerceptualLoss, self).__init__()
        self.vgg_pretrained_features = models.vgg19(weights=VGG19_Weights.DEFAULT).features
        self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).cuda()
        self.downscale_factor = downscale_factor  # 记录缩放因子

        # 配置的感知损失层及其权重
        self.indices = [2, 7, 16, 25, 34]  # 对应 conv1_2, conv2_2, conv3_4, conv4_4, conv5_4
        self.weights = [0.1, 0.1, 1.0, 1.0, 1.0]  # 与 perceptual_opt 里的一致

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X, Y):
        # 降低分辨率，减少显存占用
        X = F.interpolate(X, scale_factor=1 / self.downscale_factor, mode='bilinear', align_corners=False)
        Y = F.interpolate(Y, scale_factor=1 / self.downscale_factor, mode='bilinear', align_corners=False)

        X = self.normalize(X)
        Y = self.normalize(Y)

        loss = 0
        k = 0  # 用于索引 `weights`

        for i in range(self.indices[-1] + 1):
            X = self.vgg_pretrained_features[i](X)
            Y = self.vgg_pretrained_features[i](Y)
            if i in self.indices:
                loss += self.weights[k] * (X - Y.detach()).abs().mean() * 0.1  # 乘以 perceptual_weight = 0.1
                k += 1

        return loss
