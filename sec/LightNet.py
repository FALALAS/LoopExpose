import torch
import torch.nn as nn


class AttING(nn.Module):

    def __init__(self, in_channels, channels):
        super(AttING, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2_1 = nn.Conv2d(channels // 2, channels // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_2 = nn.Conv2d(channels // 2, channels // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.instance = nn.InstanceNorm2d(channels // 2, affine=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.process = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=True), nn.LeakyReLU(0.1),
            nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1, bias=True), nn.Sigmoid())

    def forward(self, x):
        x1 = self.conv1(x)
        x1, x2 = torch.chunk(x1, 2, dim=1)
        out_instance = self.instance(x1)
        out_identity = x2
        out1 = self.conv2_1(out_instance)
        out2 = self.conv2_2(out_identity)
        xp = torch.cat((out1, out2), 1)
        xp = self.process(self.avgpool(xp)) * xp
        xout = xp
        return xout


class LightNet(nn.Module):

    def __init__(self, dim=8, expand=2):
        super().__init__()
        self.dim = dim
        self.stage = 2
        self.in_proj = nn.Conv2d(3, dim, 1, 1, 0, bias=False)
        self.enc = AttING(dim, dim)
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(2):
            self.encoder_layers.append(
                nn.ModuleList([
                    nn.Conv2d(dim_stage, dim_stage * expand, 1, 1, 0, bias=False),
                    nn.Conv2d(dim_stage * expand, dim_stage * expand, 3, 2, 1, bias=False, groups=dim_stage * expand),
                    nn.Conv2d(dim_stage * expand, dim_stage * expand, 1, 1, 0, bias=False),
                ]))
            dim_stage *= 2

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.luminance_fc = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 1, kernel_size=1)
        )

        self.decoder_layers = nn.ModuleList([])
        for i in range(2):
            self.decoder_layers.append(
                nn.ModuleList([
                    nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                    nn.Conv2d(dim_stage // 2, dim_stage, 1, 1, 0, bias=False),
                    nn.Conv2d(dim_stage, dim_stage, 3, 1, 1, bias=False, groups=dim_stage),
                    nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, 0, bias=False),
                ]))
            dim_stage //= 2
        self.out_conv2 = nn.Conv2d(self.dim, 3, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """

        fea = self.lrelu(self.in_proj(x))
        fea = self.enc(fea)
        fea_encoder = []
        for (Conv1, Conv2, Conv3) in self.encoder_layers:
            fea_encoder.append(fea)
            fea = Conv3(self.lrelu(Conv2(self.lrelu(Conv1(fea)))))

        luminance_feature = self.global_avg_pool(fea)
        luminance_feature = self.luminance_fc(luminance_feature).squeeze()

        for i, (FeaUpSample, Conv1, Conv2, Conv3) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Conv3(self.lrelu(Conv2(self.lrelu(Conv1(fea)))))
            fea = fea + fea_encoder[self.stage - 1 - i]

        out_feature = fea
        return out_feature, luminance_feature


'''
model = LightNet()
x = torch.randn(5, 3, 256, 222)  # Example input
out_feature, luminance_feature = model(x)
print("param_map shape:", luminance_feature.shape)
print("out_feature shape:", out_feature.shape)

'''
