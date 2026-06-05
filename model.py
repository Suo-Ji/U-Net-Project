# ============================================================
# 语义分割模型定义
# 支持: U-Net, SegNet, DeepLabV3+
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
# SegNet 模型
# 论文: "SegNet: A Deep Convolutional Encoder-Decoder Architecture
#        for Image Segmentation"
# 核心特点：解码器使用编码器 MaxPool 的索引进行上采样（池化索引上采样）
# ============================================================

class SegNetDown(nn.Module):
    """SegNet 编码器块：Conv*2 -> BN -> ReLU -> MaxPool（记录索引）"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x):
        x = self.conv(x)
        x, indices = self.pool(x)
        return x, indices


class SegNetUp(nn.Module):
    """SegNet 解码器块：MaxUnpool（使用编码器索引）-> Conv*2 -> BN -> ReLU"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, indices, output_size):
        x = self.unpool(x, indices, output_size=output_size)
        x = self.conv(x)
        return x


class SegNet(nn.Module):
    """
    SegNet 模型
    输入: (B, in_channels, H, W) 的图像张量
    输出: (B, out_channels, H, W) 的分割概率图

    与 U-Net 的区别：
    - 解码器不使用转置卷积，而是用 MaxUnpool2d + 编码器池化索引恢复分辨率
    - 不传递编码器特征图（仅传索引），参数量更少、显存更省
    """

    def __init__(self, in_channels=3, out_channels=1, num_filters=64):
        super().__init__()

        f = num_filters  # 64

        # 编码器
        self.enc1 = SegNetDown(in_channels, f)         # 3   -> 64
        self.enc2 = SegNetDown(f, f * 2)               # 64  -> 128
        self.enc3 = SegNetDown(f * 2, f * 4)           # 128 -> 256
        self.enc4 = SegNetDown(f * 4, f * 8)           # 256 -> 512
        self.enc5 = SegNetDown(f * 8, f * 16)          # 512 -> 1024

        # 解码器
        self.dec5 = SegNetUp(f * 16, f * 8)            # 1024 -> 512
        self.dec4 = SegNetUp(f * 8, f * 4)             # 512  -> 256
        self.dec3 = SegNetUp(f * 4, f * 2)             # 256  -> 128
        self.dec2 = SegNetUp(f * 2, f)                 # 128  -> 64
        self.dec1 = SegNetUp(f, f)                     # 64   -> 64

        # 输出层
        self.outc = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x):
        # 编码器（记录每层池化索引和输出尺寸）
        e1, idx1 = self.enc1(x)    # (B, 64,  H/2,  W/2)
        e2, idx2 = self.enc2(e1)   # (B, 128, H/4,  W/4)
        e3, idx3 = self.enc3(e2)   # (B, 256, H/8,  W/8)
        e4, idx4 = self.enc4(e3)   # (B, 512, H/16, W/16)
        e5, idx5 = self.enc5(e4)   # (B, 1024,H/32, W/32)

        # 解码器（使用编码器池化索引上采样）
        d5 = self.dec5(e5, idx5, output_size=e4.size())   # (B, 512, H/16, W/16)
        d4 = self.dec4(d5, idx4, output_size=e3.size())   # (B, 256, H/8,  W/8)
        d3 = self.dec3(d4, idx3, output_size=e2.size())   # (B, 128, H/4,  W/4)
        d2 = self.dec2(d3, idx2, output_size=e1.size())   # (B, 64,  H/2,  W/2)
        d1 = self.dec1(d2, idx1, output_size=x.size())    # (B, 64,  H,    W)

        return self.outc(d1)    # (B, 1,   H,    W)


# ============================================================
# DeepLabV3+ 模型
# 论文: "Encoder-Decoder with Atrous Separable Convolution
#        for Semantic Image Segmentation"
# Backbone: Xception (modified for DeepLab, output_stride=16)
# ============================================================

class SeparableConv2d(nn.Module):
    """深度可分离卷积：Depthwise Conv + Pointwise Conv + BN + ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class XceptionBlock(nn.Module):
    """
    Xception 残差块
    reps 个 SeparableConv2d + skip connection
    """

    def __init__(self, in_channels, out_channels, reps=3, stride=1,
                 dilation=1, grow_first=True):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)

        # 构建 reps 个 SeparableConv2d
        layers = []
        if grow_first:
            # 第一个 rep: in_channels -> out_channels，后续保持 out_channels
            layers.append(SeparableConv2d(in_channels, out_channels, dilation=dilation))
            for _ in range(reps - 1):
                layers.append(SeparableConv2d(out_channels, out_channels, dilation=dilation))
        else:
            # 前 reps-1 个用 in_channels，最后一个用 out_channels
            for _ in range(reps - 1):
                layers.append(SeparableConv2d(in_channels, in_channels, dilation=dilation))
            layers.append(SeparableConv2d(in_channels, out_channels, dilation=dilation))
        self.sep_layers = nn.Sequential(*layers)

        # Skip connection
        self.skip = None
        if stride > 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()

        # 当 stride > 1 时，残差路径使用 MaxPool 下采样
        self.pool = nn.MaxPool2d(3, stride=stride, padding=1) if stride > 1 else None

    def forward(self, x):
        residual = self.sep_layers(x)
        if self.pool is not None:
            residual = self.pool(residual)
        skip = self.skip(x)
        return self.relu(residual + skip)


class XceptionBackbone(nn.Module):
    """
    Modified Xception backbone for DeepLabV3+
    output_stride=16: exit flow 使用 atrous conv 保持分辨率

    返回:
        low_level_features: Entry stage2 输出 (B, 128, H/4, W/4)
        high_level_features: Exit flow 输出 (B, 2048, H/16, W/16)
    """

    def __init__(self, in_channels=3):
        super().__init__()

        # ---- Entry Flow ----
        self.entry0 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.entry1 = XceptionBlock(64, 128, reps=2, stride=2, grow_first=True)
        self.entry2 = XceptionBlock(128, 256, reps=2, stride=2, grow_first=True)
        self.entry3 = XceptionBlock(256, 728, reps=2, stride=2, grow_first=True)

        # ---- Middle Flow (×16) ----
        self.middle = nn.Sequential(
            *[XceptionBlock(728, 728, reps=3, stride=1, grow_first=True) for _ in range(16)]
        )

        # ---- Exit Flow (output_stride=16, 使用 atrous conv) ----
        self.exit0 = XceptionBlock(728, 728, reps=2, stride=1, grow_first=True)
        self.exit1 = nn.Sequential(
            SeparableConv2d(728, 728, dilation=1),
            SeparableConv2d(728, 2048, dilation=2, padding=2),
        )
        self.exit1_skip = nn.Sequential(
            nn.Conv2d(728, 2048, 1, bias=False),
            nn.BatchNorm2d(2048)
        )
        self.exit1_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Entry flow
        x = self.entry0(x)       # (B, 64, H/2, W/2)
        x = self.entry1(x)       # (B, 128, H/4, W/4)
        low_level = x            # ← 保存 low-level features
        x = self.entry2(x)       # (B, 256, H/8, W/8)
        x = self.entry3(x)       # (B, 728, H/16, W/16)

        # Middle flow
        x = self.middle(x)       # (B, 728, H/16, W/16)

        # Exit flow
        x = self.exit0(x)        # (B, 728, H/16, W/16)
        residual = self.exit1(x)
        skip = self.exit1_skip(x)
        x = self.exit1_relu(residual + skip)  # (B, 2048, H/16, W/16)

        return low_level, x


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling
    5 个并行分支：1×1 conv + 3×3 atrous conv (rate=6/12/18) + global avg pool
    """

    def __init__(self, in_channels=2048, out_channels=256):
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.branch2 = SeparableConv2d(in_channels, out_channels, dilation=6, padding=6)
        self.branch3 = SeparableConv2d(in_channels, out_channels, dilation=12, padding=12)
        self.branch4 = SeparableConv2d(in_channels, out_channels, dilation=18, padding=18)

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # 投影层：1280 -> 256
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h, w = x.shape[2:]

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        b5 = self.global_pool(x)
        b5 = F.interpolate(b5, size=(h, w), mode='bilinear', align_corners=False)

        x = torch.cat([b1, b2, b3, b4, b5], dim=1)
        x = self.project(x)
        return x


class DeepLabV3PlusDecoder(nn.Module):
    """
    DeepLabV3+ 解码器
    将 ASPP 输出与 low-level features 融合
    """

    def __init__(self, low_level_channels=128, aspp_channels=256, out_channels=1):
        super().__init__()

        # 压缩 low-level features 通道
        self.reduce_low = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        # 融合后的卷积
        self.block = nn.Sequential(
            nn.Conv2d(aspp_channels + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # 输出层
        self.output_conv = nn.Conv2d(256, out_channels, 1)

    def forward(self, low_level, aspp_out):
        # 压缩 low-level features
        low = self.reduce_low(low_level)

        # 上采样 ASPP 输出到 low-level 分辨率 (×4)
        aspp_up = F.interpolate(aspp_out, size=low.shape[2:],
                                mode='bilinear', align_corners=False)

        # 拼接 + 卷积
        x = torch.cat([low, aspp_up], dim=1)
        x = self.block(x)

        # 上采样到原始分辨率 (×4)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)

        return self.output_conv(x)


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ 模型 (Xception backbone, output_stride=16)

    输入: (B, in_channels, H, W) 的图像张量
    输出: (B, out_channels, H, W) 的分割 logits

    注: num_filters 参数为兼容接口保留，实际不使用（Xception 通道数固定）
    """

    def __init__(self, in_channels=3, out_channels=1, num_filters=64):
        super().__init__()

        self.backbone = XceptionBackbone(in_channels)
        self.aspp = ASPP(in_channels=2048, out_channels=256)
        self.decoder = DeepLabV3PlusDecoder(
            low_level_channels=128, aspp_channels=256, out_channels=out_channels
        )

    def forward(self, x):
        low_level, high_level = self.backbone(x)
        x = self.aspp(high_level)
        x = self.decoder(low_level, x)
        return x


# ============================================================
# 模型工厂函数
# ============================================================

VALID_MODELS = ("unet", "segnet", "deeplabv3plus")


def get_model_class(model_name):
    """
    根据模型名称返回对应的模型类

    参数:
        model_name: "unet" 或 "segnet"

    返回:
        对应的模型类（UNet 或 SegNet）
    """
    model_name = model_name.lower()
    if model_name == "unet":
        return UNet
    elif model_name == "segnet":
        return SegNet
    elif model_name == "deeplabv3plus":
        return DeepLabV3Plus
    else:
        raise ValueError(f"不支持的模型: {model_name}，可选: {VALID_MODELS}")
