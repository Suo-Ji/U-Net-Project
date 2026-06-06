# ============================================================
# 语义分割模型定义
# 支持: U-Net (含边缘注意力变体)
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# U-Net 模型
# 论文: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
# ============================================================

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


# ============================================================
# 边缘注意力模块 + U-Net 边缘注意力变体
# ============================================================

class EdgeSE(nn.Module):
    """
    边缘注意力模块（SE-Block 风格）
    用边缘图引导编码器特征的通道权重

    edge_map (B,1,H,W) → GAP → FC → ReLU → FC → Sigmoid → channel_weights (B,C,1,1)
    feature_map (B,C,H,W) × channel_weights → weighted_feature_map
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, feature_map, edge_map):
        """
        参数:
            feature_map: (B, C, H, W) 编码器特征
            edge_map: (B, 1, H, W) 边缘图
        """
        # 边缘图全局平均池化 → (B, 1)
        edge_pooled = edge_map.mean(dim=[2, 3])  # (B, 1)
        # 通过 FC 生成通道权重
        weights = self.fc(edge_pooled)            # (B, C)
        weights = weights.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return feature_map * weights


class UNetEdgeAttention(UNet):
    """
    U-Net 边缘注意力变体
    在编码器 inc 层之后插入 EdgeSE 模块

    forward(x, edge_map) 接收边缘图作为额外输入
    """

    def __init__(self, in_channels=3, out_channels=1, num_filters=64):
        super().__init__(in_channels, out_channels, num_filters)
        f = num_filters
        self.edge_se = EdgeSE(f)
        self.use_edge = True  # 标记，供 train.py 判断

    def forward(self, x, edge_map):
        # 编码器（inc 后插入边缘注意力）
        x1 = self.inc(x)              # (B, 64, H, W)
        x1 = self.edge_se(x1, edge_map)  # 边缘注意力加权

        x2 = self.down1(x1)           # (B, 128, H/2, W/2)
        x3 = self.down2(x2)           # (B, 256, H/4, W/4)
        x4 = self.down3(x3)           # (B, 512, H/8, W/8)
        x5 = self.down4(x4)           # (B, 1024, H/16, W/16)

        # 解码器 + 跳跃连接
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)


# ============================================================
# 模型工厂函数
# ============================================================

VALID_MODELS = ("unet", "unet_canny", "unet_sobel", "unet_laplacian")


def get_model_class(model_name):
    """
    根据模型名称返回对应的模型类

    参数:
        model_name: "unet" / "unet_canny" / "unet_sobel" / "unet_laplacian"

    返回:
        对应的模型类
    """
    model_name = model_name.lower()
    if model_name == "unet":
        return UNet
    elif model_name in ("unet_canny", "unet_sobel", "unet_laplacian"):
        return UNetEdgeAttention
    else:
        raise ValueError(f"不支持的模型: {model_name}，可选: {VALID_MODELS}")
