# ============================================================
# CamVid 数据集加载器
# 功能：加载 CamVid 数据集，将灰度索引掩膜转换为二值道路掩膜，
#       并应用 Albumentations 数据增强
#
# 本数据集的标注为单通道灰度索引图（非 RGB 彩色掩膜）：
#   类别索引: 0=Void, 1=Building, 2=Column_Pole, 3=Road,
#             4=Sidewalk, 5=Tree, 6=Sign, 7=Fence,
#             8=Car, 9=Pedestrian, 10=Cyclist, 11=Void
# ============================================================

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config

# CamVid 类别索引中，Road = 3
ROAD_CLASS_INDEX = 3

# 标注目录的后缀映射：train -> trainannot, val -> valannot, test -> testannot
ANNOT_DIR_SUFFIX = "annot"


class CamVidDataset(Dataset):
    """
    CamVid 数据集类

    目录结构要求：
        data_dir/
            {split}/        - 图像目录（如 train/, val/, test/）
            {split}annot/   - 标签目录（如 trainannot/, valannot/, testannot/）

    掩膜处理：将灰度索引掩膜转换为二值掩膜（道路=1，非道路=0）
    """

    def __init__(self, data_dir, split="train", img_size=(256, 256), augment=False):
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        self.augment = augment

        # 构建图像和标签目录路径
        img_dir = os.path.join(data_dir, split)
        mask_dir = os.path.join(data_dir, f"{split}{ANNOT_DIR_SUFFIX}")

        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"图像目录不存在: {img_dir}")
        if not os.path.isdir(mask_dir):
            raise FileNotFoundError(f"标签目录不存在: {mask_dir}")

        # 获取所有图像文件并排序
        self.images = sorted([
            os.path.join(img_dir, f) for f in os.listdir(img_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])

        # 匹配对应的标签文件（图像与标签同名，存放在 {split}annot/ 目录中）
        self.masks = []
        for img_path in self.images:
            img_name = os.path.basename(img_path)
            mask_path = os.path.join(mask_dir, img_name)

            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"找不到标签文件: {mask_path}")
            self.masks.append(mask_path)

        assert len(self.images) == len(self.masks), \
            f"图像数量 ({len(self.images)}) 与标签数量 ({len(self.masks)}) 不匹配"

        print(f"[{split}] 加载了 {len(self.images)} 个样本，道路类别索引 = {ROAD_CLASS_INDEX}")

        # 构建数据增强管道
        self.transform = self._build_transform()

    def _build_transform(self):
        """构建数据增强与预处理管道"""
        transform_list = [
            A.Resize(self.img_size[0], self.img_size[1]),
        ]

        # 仅在训练阶段应用数据增强
        if self.augment:
            transform_list.extend([
                A.HorizontalFlip(p=config.AUGMENT_PROB),
                A.RandomRotate90(p=config.AUGMENT_PROB),
                A.RandomCrop(height=self.img_size[0], width=self.img_size[1], p=0.3),
                A.Affine(
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    scale=(0.85, 1.15), rotate=(-15, 15),
                    border_mode=0, p=config.AUGMENT_PROB
                ),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1,
                    p=config.AUGMENT_PROB
                ),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.GaussNoise(p=0.2),
            ])

        # 归一化并转为 Tensor
        transform_list.extend([
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])

        return A.Compose(transform_list)

    def _mask_to_binary(self, mask_gray):
        """
        将灰度索引掩膜转换为二值掩膜
        道路像素 (class index = ROAD_CLASS_INDEX) -> 1.0，其余 -> 0.0
        """
        return (mask_gray == ROAD_CLASS_INDEX).astype(np.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 加载 RGB 图像
        image = np.array(Image.open(self.images[idx]).convert("RGB"))

        # 加载灰度索引标签（单通道，每个像素值为类别索引）
        mask = np.array(Image.open(self.masks[idx]))

        # 应用数据增强与预处理
        augmented = self.transform(image=image, mask=mask)
        image_tensor = augmented["image"]        # (3, H, W), 归一化后的 Tensor

        # ToTensorV2 会将 mask 转为 Tensor，需要转回 numpy 以便二值化处理
        mask_tensor = augmented["mask"]           # (H, W) Tensor
        if isinstance(mask_tensor, torch.Tensor):
            mask_arr = mask_tensor.cpu().numpy()
        else:
            mask_arr = mask_tensor

        # 将索引掩膜转换为二值道路掩膜
        binary_mask = self._mask_to_binary(mask_arr)
        mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0)  # (1, H, W)

        return image_tensor, mask_tensor
