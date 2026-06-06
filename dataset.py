# ============================================================
# 数据集加载器
# 支持: CamVid, Cityscapes
# ============================================================

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

import config


# ============================================================
# 边缘检测工具函数
# ============================================================

def compute_edge_map(image_rgb, edge_type):
    """
    从 RGB 图像计算边缘图

    参数:
        image_rgb: numpy 数组 (H, W, 3), uint8
        edge_type: "canny" / "sobel" / "laplacian" / None

    返回:
        numpy 数组 (H, W), float32, 归一化到 [0, 1]；edge_type=None 时返回全零
    """
    if edge_type is None:
        return np.zeros(image_rgb.shape[:2], dtype=np.float32)

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    if edge_type == "canny":
        edge = cv2.Canny(gray, 50, 150)
    elif edge_type == "sobel":
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    elif edge_type == "laplacian":
        edge = cv2.Laplacian(gray, cv2.CV_64F)
        edge = np.abs(edge)
    else:
        raise ValueError(f"不支持的边缘类型: {edge_type}")

    # 归一化到 [0, 1]
    edge = edge.astype(np.float32)
    max_val = edge.max()
    if max_val > 0:
        edge = edge / max_val
    return edge


def get_edge_type_from_model(model_name):
    """
    从模型名称推导边缘类型

    参数:
        model_name: "unet" / "unet_canny" / "unet_sobel" / "unet_laplacian"

    返回:
        edge_type: "canny" / "sobel" / "laplacian" / None
    """
    edge_models = {
        "unet_canny": "canny",
        "unet_sobel": "sobel",
        "unet_laplacian": "laplacian",
    }
    return edge_models.get(model_name)

# ============================================================
# CamVid 数据集
# ============================================================
# 标注为单通道灰度索引图（非 RGB 彩色掩膜）：
#   类别索引: 0=Void, 1=Building, 2=Column_Pole, 3=Road,
#             4=Sidewalk, 5=Tree, 6=Sign, 7=Fence,
#             8=Car, 9=Pedestrian, 10=Cyclist, 11=Void

CAMVID_ROAD_CLASS_INDEX = 3
CAMVID_ANNOT_DIR_SUFFIX = "annot"


class CamVidDataset(Dataset):
    """
    CamVid 数据集类

    目录结构要求：
        data_dir/
            {split}/        - 图像目录（如 train/, val/, test/）
            {split}annot/   - 标签目录（如 trainannot/, valannot/, testannot/）

    掩膜处理：将灰度索引掩膜转换为二值掩膜（道路=1，非道路=0）
    """

    def __init__(self, data_dir, split="train", img_size=(256, 256), augment=False, edge_type=None):
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        self.augment = augment
        self.edge_type = edge_type

        # 构建图像和标签目录路径
        img_dir = os.path.join(data_dir, split)
        mask_dir = os.path.join(data_dir, f"{split}{CAMVID_ANNOT_DIR_SUFFIX}")

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

        print(f"[{split}] 加载了 {len(self.images)} 个样本，道路类别索引 = {CAMVID_ROAD_CLASS_INDEX}")

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

        # edge_map 作为额外的 mask 类型，同步进行空间变换
        return A.Compose(transform_list, additional_targets={'edge': 'mask'})

    def _mask_to_binary(self, mask_gray):
        """
        将灰度索引掩膜转换为二值掩膜
        道路像素 (class index = CAMVID_ROAD_CLASS_INDEX) -> 1.0，其余 -> 0.0
        """
        return (mask_gray == CAMVID_ROAD_CLASS_INDEX).astype(np.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 加载 RGB 图像
        image = np.array(Image.open(self.images[idx]).convert("RGB"))

        # 加载灰度索引标签（单通道，每个像素值为类别索引）
        mask = np.array(Image.open(self.masks[idx]))

        # 在 transform 之前计算边缘图（边缘算子需要 uint8 输入）
        edge = compute_edge_map(image, self.edge_type)

        # 应用数据增强与预处理（edge 同步进行空间变换）
        augmented = self.transform(image=image, mask=mask, edge=edge)
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

        # 边缘图转为 Tensor
        edge_tensor = augmented["edge"].float().unsqueeze(0)  # (1, H, W)

        return image_tensor, mask_tensor, edge_tensor


# ============================================================
# Cityscapes 数据集
# ============================================================
# 标注为 _gtFine_labelIds.png（单通道灰度索引图）：
#   常用类别索引: 0=Void, 1=Flat/Road, 2=Human, 3=Vehicle, ...
#   在 labelIds 格式中，Road = 7

CITYSCAPES_ROAD_CLASS_INDEX = 7
CITYSCAPES_IMG_DIR_NAME = "leftImg8bit"
CITYSCAPES_MASK_DIR_NAME = "gtFine"
CITYSCAPES_IMG_SUFFIX = "_leftImg8bit.png"
CITYSCAPES_MASK_SUFFIX = "_gtFine_labelIds.png"


class CityscapesDataset(Dataset):
    """
    Cityscapes 数据集类

    目录结构要求：
        data_dir/
            leftImg8bit/
                {split}/{city}/*.png     - 图像目录（含城市子目录）
            gtFine/
                {split}/{city}/*.png     - 标签目录（含城市子目录）

    掩膜处理：将 _gtFine_labelIds.png 转换为二值掩膜（道路=1，非道路=0）
    """

    def __init__(self, data_dir, split="train", img_size=(256, 256), augment=False, edge_type=None):
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        self.augment = augment
        self.edge_type = edge_type

        # 构建图像根目录：leftImg8bit/{split}/
        img_split_dir = os.path.join(data_dir, CITYSCAPES_IMG_DIR_NAME, split)
        if not os.path.isdir(img_split_dir):
            raise FileNotFoundError(f"图像目录不存在: {img_split_dir}")

        # 递归遍历所有城市子目录，收集图像文件
        self.images = []
        for city_dir in sorted(os.listdir(img_split_dir)):
            city_path = os.path.join(img_split_dir, city_dir)
            if not os.path.isdir(city_path):
                continue
            for fname in sorted(os.listdir(city_path)):
                if fname.endswith(CITYSCAPES_IMG_SUFFIX):
                    self.images.append(os.path.join(city_path, fname))

        # 构建对应的标签文件路径
        self.masks = []
        for img_path in self.images:
            # 替换目录名：leftImg8bit -> gtFine
            mask_path = img_path.replace(
                CITYSCAPES_IMG_DIR_NAME, CITYSCAPES_MASK_DIR_NAME, 1
            )
            # 替换文件名后缀：_leftImg8bit.png -> _gtFine_labelIds.png
            mask_path = mask_path.replace(CITYSCAPES_IMG_SUFFIX, CITYSCAPES_MASK_SUFFIX)

            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"找不到标签文件: {mask_path}")
            self.masks.append(mask_path)

        assert len(self.images) == len(self.masks), \
            f"图像数量 ({len(self.images)}) 与标签数量 ({len(self.masks)}) 不匹配"

        print(f"[{split}] 加载了 {len(self.images)} 个样本 (Cityscapes)，道路类别索引 = {CITYSCAPES_ROAD_CLASS_INDEX}")

        # 构建数据增强管道
        self.transform = self._build_transform()

    def _build_transform(self):
        """构建数据增强与预处理管道（与 CamVid 一致）"""
        transform_list = [
            A.Resize(self.img_size[0], self.img_size[1]),
        ]

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

        transform_list.extend([
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])

        # edge_map 作为额外的 mask 类型，同步进行空间变换
        return A.Compose(transform_list, additional_targets={'edge': 'mask'})

    def _mask_to_binary(self, mask_gray):
        """
        将灰度索引掩膜转换为二值掩膜
        道路像素 (class index = CITYSCAPES_ROAD_CLASS_INDEX) -> 1.0，其余 -> 0.0
        """
        return (mask_gray == CITYSCAPES_ROAD_CLASS_INDEX).astype(np.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 加载 RGB 图像
        image = np.array(Image.open(self.images[idx]).convert("RGB"))

        # 加载灰度索引标签
        mask = np.array(Image.open(self.masks[idx]))

        # 在 transform 之前计算边缘图（边缘算子需要 uint8 输入）
        edge = compute_edge_map(image, self.edge_type)

        # 应用数据增强与预处理（edge 同步进行空间变换）
        augmented = self.transform(image=image, mask=mask, edge=edge)
        image_tensor = augmented["image"]

        # ToTensorV2 会将 mask 转为 Tensor，需要转回 numpy 以便二值化处理
        mask_tensor = augmented["mask"]
        if isinstance(mask_tensor, torch.Tensor):
            mask_arr = mask_tensor.cpu().numpy()
        else:
            mask_arr = mask_tensor

        # 将索引掩膜转换为二值道路掩膜
        binary_mask = self._mask_to_binary(mask_arr)
        mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0)  # (1, H, W)

        # 边缘图转为 Tensor
        edge_tensor = augmented["edge"].float().unsqueeze(0)  # (1, H, W)

        return image_tensor, mask_tensor, edge_tensor


# ============================================================
# 数据集工厂函数
# ============================================================

def get_dataset_class(dataset_name):
    """
    根据数据集名称返回对应的 Dataset 类

    参数:
        dataset_name: "camvid" 或 "cityscapes"

    返回:
        对应的 Dataset 类（CamVidDataset 或 CityscapesDataset）
    """
    dataset_name = dataset_name.lower()
    if dataset_name == "camvid":
        return CamVidDataset
    elif dataset_name == "cityscapes":
        return CityscapesDataset
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}，可选: camvid, cityscapes")
