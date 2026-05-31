# ============================================================
# 标准 U-Net 模型架构
# 论文: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
# ============================================================

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """双卷积块：Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """编码器下采样块：MaxPool -> DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """解码器上采样块：转置卷积上采样 -> 拼接跳跃连接 -> DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 处理尺寸不匹配的情况（当输入尺寸不是 2 的幂次时）
        diff_h = x2.size(2) - x1.size(2)
        diff_w = x2.size(3) - x1.size(3)
        x1 = nn.functional.pad(x1, [diff_w // 2, diff_w - diff_w // 2,
                                     diff_h // 2, diff_h - diff_h // 2])
        # 拼接跳跃连接（编码器特征）与上采样特征
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    标准 U-Net 模型
    输入: (B, in_channels, H, W) 的图像张量
    输出: (B, out_channels, H, W) 的分割概率图
    """

    def __init__(self, in_channels=3, out_channels=1, num_filters=64):
        super().__init__()

        f = num_filters  # 64

        # 编码器（收缩路径）
        self.inc = DoubleConv(in_channels, f)         # 3  -> 64
        self.down1 = Down(f, f * 2)                   # 64 -> 128
        self.down2 = Down(f * 2, f * 4)               # 128 -> 256
        self.down3 = Down(f * 4, f * 8)               # 256 -> 512
        self.down4 = Down(f * 8, f * 16)              # 512 -> 1024

        # 解码器（扩张路径）
        self.up1 = Up(f * 16, f * 8)                  # 1024 -> 512
        self.up2 = Up(f * 8, f * 4)                   # 512  -> 256
        self.up3 = Up(f * 4, f * 2)                   # 256  -> 128
        self.up4 = Up(f * 2, f)                       # 128  -> 64

        # 输出层
        self.outc = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x):
        # 编码器
        x1 = self.inc(x)       # (B, 64,  H, W)
        x2 = self.down1(x1)    # (B, 128, H/2,  W/2)
        x3 = self.down2(x2)    # (B, 256, H/4,  W/4)
        x4 = self.down3(x3)    # (B, 512, H/8,  W/8)
        x5 = self.down4(x4)    # (B, 1024,H/16, W/16)

        # 解码器 + 跳跃连接
        x = self.up1(x5, x4)   # (B, 512, H/8,  W/8)
        x = self.up2(x, x3)    # (B, 256, H/4,  W/4)
        x = self.up3(x, x2)    # (B, 128, H/2,  W/2)
        x = self.up4(x, x1)    # (B, 64,  H,    W)

        return self.outc(x)     # (B, 1,   H,    W)
