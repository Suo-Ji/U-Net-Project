# ============================================================
# 可视化模块
# 在测试集上随机选取图像，并排展示：输入图像、真实标签、预测结果
# ============================================================

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

import config

# 图片统一保存到项目根目录的 figure/ 文件夹
FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figure")


def denormalize(tensor):
    """
    反归一化：将标准化后的图像张量还原为 [0, 255] 的 numpy 数组
    输入: (3, H, W) tensor，经过 Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    输出: (H, W, 3) numpy uint8
    """
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * 0.5 + 0.5          # 反归一化到 [0, 1]
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def visualize_prediction(model, test_loader, device, num_samples=3):
    """
    在测试集上随机选取图像进行推理预测并可视化

    参数:
        model: 训练好的 U-Net 模型
        test_loader: 测试集 DataLoader
        device: 计算设备
        num_samples: 展示的样本数量
    """
    model.eval()

    # 收集所有测试样本
    samples = list(test_loader)

    # 随机选取样本
    selected = random.sample(range(len(samples)), min(num_samples, len(samples)))

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]  # 确保 axes 始终为 2D

    for row, idx in enumerate(selected):
        image, mask = samples[idx]
        image = image.to(device)
        mask = mask.to(device)

        # 推理预测
        with torch.no_grad():
            output = model(image)
            pred = torch.sigmoid(output)
            pred_mask = (pred > 0.5).float()

        # 转换为可显示的格式
        img_np = denormalize(image[0])
        mask_np = mask[0, 0].cpu().numpy()
        pred_np = pred_mask[0, 0].cpu().numpy()

        # 绘制三张图
        axes[row, 0].imshow(img_np)
        axes[row, 0].set_title("Input Image", fontsize=14)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("Ground Truth Mask", fontsize=14)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(pred_np, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].set_title("Predicted Segmentation Mask", fontsize=14)
        axes[row, 2].axis("off")

    plt.tight_layout()
    save_path = os.path.join(FIGURE_DIR, "prediction_result.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n预测结果已保存至 {save_path}")
    plt.show()


def plot_training_history(history):
    """绘制训练历史曲线（损失、IoU、像素准确率）"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # 损失曲线
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker="o", markersize=3)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", marker="s", markersize=3)
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # IoU 曲线
    axes[1].plot(epochs, history["val_iou"], label="Val IoU", color="green", marker="o", markersize=3)
    axes[1].set_title("IoU Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("IoU")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 像素准确率曲线
    axes[2].plot(epochs, history["val_acc"], label="Val Pixel Accuracy", color="orange", marker="o", markersize=3)
    axes[2].set_title("Pixel Accuracy Curve")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(FIGURE_DIR, "training_history.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"训练曲线已保存至 {save_path}")
    plt.show()


def predict_custom_images(model, image_paths, device):
    """
    对自定义图片进行道路分割预测并可视化
    如果图片所在的同级目录存在同名标签文件，则额外展示 Ground Truth（三列布局）

    参数:
        model: 训练好的 U-Net 模型
        image_paths: 图片路径列表（支持 .png/.jpg/.jpeg/.bmp）
        device: 计算设备
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    model.eval()

    # 预处理管道（与测试时一致：Resize + Normalize + ToTensor）
    transform = A.Compose([
        A.Resize(config.IMG_HEIGHT, config.IMG_WIDTH),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])

    # 检测哪些图片有对应的 GT 标签
    has_gt = []
    gt_arrays = []
    for path in image_paths:
        found = None
        img_dir = os.path.dirname(path)
        if img_dir:
            parent_dir = os.path.dirname(img_dir)
            # 直接拼接 "annot" 后缀：CamVid/test → CamVid/testannot
            annot_dir = os.path.join(parent_dir, os.path.basename(img_dir) + "annot")
            gt_path = os.path.join(annot_dir, os.path.basename(path))
            if os.path.isdir(annot_dir) and os.path.isfile(gt_path):
                found = gt_path
        has_gt.append(found is not None)
        if found:
            gt = np.array(Image.open(found))
            # 确保是单通道灰度索引图，排除误匹配到 RGB 原图
            if len(gt.shape) == 2:
                gt_binary = (gt == 3).astype(np.float32)  # Road class index = 3
            else:
                has_gt[-1] = False
                gt_binary = None
            gt_arrays.append(gt_binary)
        else:
            gt_arrays.append(None)

    # 有 GT 则三列，无 GT 则两列
    num_cols = 3 if any(has_gt) else 2
    n = len(image_paths)
    fig, axes = plt.subplots(n, num_cols, figsize=(5 * num_cols, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, path in enumerate(image_paths):
        # 加载并预处理
        image = np.array(Image.open(path).convert("RGB"))
        augmented = transform(image=image)
        input_tensor = augmented["image"].unsqueeze(0).to(device)

        # 推理
        with torch.no_grad():
            pred = torch.sigmoid(model(input_tensor))
            pred_mask = (pred > 0.5).float()

        # 转换为可显示格式
        img_np = denormalize(input_tensor[0])
        pred_np = pred_mask[0, 0].cpu().numpy()

        col = 0
        # 原图
        axes[row, col].imshow(img_np)
        axes[row, col].set_title(f"Input: {os.path.basename(path)}", fontsize=12)
        axes[row, col].axis("off")
        col += 1

        # Ground Truth（如果有）
        if has_gt[row]:
            gt_resized = np.array(Image.fromarray((gt_arrays[row] * 255).astype(np.uint8)).resize(
                (config.IMG_WIDTH, config.IMG_HEIGHT), Image.NEAREST)) / 255.0
            axes[row, col].imshow(gt_resized, cmap="gray", vmin=0, vmax=1)
            axes[row, col].set_title("Ground Truth Mask", fontsize=12)
            axes[row, col].axis("off")
            col += 1

        # 预测掩膜
        axes[row, col].imshow(pred_np, cmap="gray", vmin=0, vmax=1)
        axes[row, col].set_title("Predicted Road Mask", fontsize=12)
        axes[row, col].axis("off")

    plt.tight_layout()
    save_path = os.path.join(FIGURE_DIR, "custom_prediction.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n自定义预测结果已保存至 {save_path}")
    plt.show()
