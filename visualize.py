# ============================================================
# 可视化模块
# 在测试集上随机选取图像，并排展示：输入图像、真实标签、预测结果
# 支持单模型可视化和模型对比可视化
# ============================================================

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

import config


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


def visualize_prediction(model, test_loader, device, save_dir, model_name="model", num_samples=3):
    """
    在测试集上随机选取图像进行推理预测并可视化

    参数:
        model: 训练好的模型
        test_loader: 测试集 DataLoader
        device: 计算设备
        save_dir: 图片保存目录
        model_name: 模型名称（用于文件名）
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
        axes[row, 2].set_title(f"Prediction ({model_name.upper()})", fontsize=14)
        axes[row, 2].axis("off")

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}_prediction_result.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n预测结果已保存至 {save_path}")
    plt.show()


def plot_training_history(history, save_dir, model_name="model"):
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
    axes[2].plot(epochs, history["val_pixel_acc"], label="Val Pixel Accuracy", color="orange", marker="o", markersize=3)
    axes[2].set_title("Pixel Accuracy Curve")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}_training_history.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"训练曲线已保存至 {save_path}")
    plt.show()


def predict_custom_images(model, image_paths, device, save_dir, model_name="model"):
    """
    对自定义图片进行道路分割预测并可视化
    如果图片所在的同级目录存在同名标签文件，则额外展示 Ground Truth（三列布局）

    参数:
        model: 训练好的模型
        image_paths: 图片路径列表（支持 .png/.jpg/.jpeg/.bmp）
        device: 计算设备
        save_dir: 图片保存目录
        model_name: 模型名称（用于文件名）
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
            annot_dir = os.path.join(parent_dir, os.path.basename(img_dir) + "annot")
            gt_path = os.path.join(annot_dir, os.path.basename(path))
            if os.path.isdir(annot_dir) and os.path.isfile(gt_path):
                found = gt_path
        has_gt.append(found is not None)
        if found:
            gt = np.array(Image.open(found))
            if len(gt.shape) == 2:
                gt_binary = (gt == 3).astype(np.float32)
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
        image = np.array(Image.open(path).convert("RGB"))
        augmented = transform(image=image)
        input_tensor = augmented["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            pred = torch.sigmoid(model(input_tensor))
            pred_mask = (pred > 0.5).float()

        img_np = denormalize(input_tensor[0])
        pred_np = pred_mask[0, 0].cpu().numpy()

        col = 0
        axes[row, col].imshow(img_np)
        axes[row, col].set_title(f"Input: {os.path.basename(path)}", fontsize=12)
        axes[row, col].axis("off")
        col += 1

        if has_gt[row]:
            gt_resized = np.array(Image.fromarray((gt_arrays[row] * 255).astype(np.uint8)).resize(
                (config.IMG_WIDTH, config.IMG_HEIGHT), Image.NEAREST)) / 255.0
            axes[row, col].imshow(gt_resized, cmap="gray", vmin=0, vmax=1)
            axes[row, col].set_title("Ground Truth Mask", fontsize=12)
            axes[row, col].axis("off")
            col += 1

        axes[row, col].imshow(pred_np, cmap="gray", vmin=0, vmax=1)
        axes[row, col].set_title(f"Predicted ({model_name.upper()})", fontsize=12)
        axes[row, col].axis("off")

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}_custom_prediction.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n自定义预测结果已保存至 {save_path}")
    plt.show()


# ============================================================
# 对比可视化
# ============================================================

def compare_training(histories, dataset_name, save_dir,
                     model_names=None, display_names=None):
    """
    绘制多个模型的训练曲线对比图

    参数:
        histories: dict，key 为模型名称，value 为训练历史字典
        dataset_name: 数据集名称
        save_dir: 保存目录
        model_names: 模型 key 列表（默认取 histories 的所有 key）
        display_names: 显示名称列表
    """
    if model_names is None:
        model_names = list(histories.keys())
    if display_names is None:
        display_names = model_names

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    markers = ["o", "s", "^", "D", "v"]

    metrics_to_plot = [
        ("train_loss", "Train Loss"),
        ("val_loss", "Val Loss"),
        ("val_iou", "Val IoU"),
        ("val_pixel_acc", "Val Pixel Accuracy"),
        ("val_precision", "Val Precision"),
        ("val_recall", "Val Recall"),
        ("val_f1", "Val F1-Score"),
        ("val_nll", "Val NLL"),
        ("val_ece", "Val ECE"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    axes = axes.flatten()

    for i, (key, title) in enumerate(metrics_to_plot):
        for j, name in enumerate(model_names):
            h = histories.get(name, {})
            if key in h and len(h[key]) > 0:
                epochs = range(1, len(h[key]) + 1)
                axes[i].plot(epochs, h[key], label=display_names[j],
                             color=colors[j % len(colors)],
                             marker=markers[j % len(markers)], markersize=2)
        axes[i].set_title(title, fontsize=12)
        axes[i].set_xlabel("Epoch")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    title_str = " vs ".join(display_names)
    plt.suptitle(f"{title_str} — {dataset_name.upper()}", fontsize=16, fontweight="bold")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{dataset_name}_compare_training.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"训练曲线对比已保存至 {save_path}")
    plt.show()


def compare_prediction(models, val_loader, device, dataset_name, save_dir,
                       model_names=None, display_names=None, num_samples=3):
    """
    同一张图上并排展示多个模型的预测结果

    参数:
        models: 模型实例列表
        val_loader: 验证集 DataLoader
        device: 计算设备
        dataset_name: 数据集名称
        save_dir: 保存目录
        model_names: 模型 key 列表
        display_names: 显示名称列表
        num_samples: 展示的样本数量
    """
    if display_names is None:
        display_names = model_names if model_names else [f"Model {i}" for i in range(len(models))]

    for m in models:
        m.eval()

    samples = list(val_loader)
    selected = random.sample(range(len(samples)), min(num_samples, len(samples)))

    # 列数：Input + GT + 每个模型一列
    n_cols = 2 + len(models)
    fig, axes = plt.subplots(num_samples, n_cols, figsize=(5 * n_cols, 5 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(selected):
        image, mask = samples[idx]
        image = image.to(device)
        mask = mask.to(device)

        img_np = denormalize(image[0])
        mask_np = mask[0, 0].cpu().numpy()

        axes[row, 0].imshow(img_np)
        axes[row, 0].set_title("Input Image", fontsize=12)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("Ground Truth", fontsize=12)
        axes[row, 1].axis("off")

        for j, model in enumerate(models):
            with torch.no_grad():
                pred = torch.sigmoid(model(image))
                pred_mask = (pred > 0.5).float()

            pred_np = pred_mask[0, 0].cpu().numpy()
            axes[row, 2 + j].imshow(pred_np, cmap="gray", vmin=0, vmax=1)
            axes[row, 2 + j].set_title(display_names[j], fontsize=12)
            axes[row, 2 + j].axis("off")

    title_str = " vs ".join(display_names)
    plt.suptitle(f"{title_str} — {dataset_name.upper()}", fontsize=16, fontweight="bold")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{dataset_name}_compare_prediction.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"预测对比已保存至 {save_path}")
    plt.show()


def print_comparison_table(histories, dataset_name,
                           model_names=None, display_names=None):
    """
    在终端打印多个模型的指标对比表

    参数:
        histories: dict，key 为模型名称，value 为训练历史字典
        dataset_name: 数据集名称
        model_names: 模型 key 列表
        display_names: 显示名称列表
    """
    if model_names is None:
        model_names = list(histories.keys())
    if display_names is None:
        display_names = model_names

    def last(h, key):
        vals = h.get(key, [])
        return vals[-1] if vals else 0.0

    metrics = [
        ("Loss",         "val_loss",      False),
        ("IoU",          "val_iou",       True),
        ("Pixel Acc",    "val_pixel_acc", True),
        ("Precision",    "val_precision", True),
        ("Recall",       "val_recall",    True),
        ("F1-Score",     "val_f1",        True),
        ("NLL",          "val_nll",       False),
        ("ECE",          "val_ece",       False),
    ]

    n = len(model_names)
    col_width = 12
    row_width = 14 + n * (col_width + 3)
    sep = "=" * row_width

    title_str = " vs ".join(display_names)
    print()
    print(sep)
    print(f"  {title_str} — {dataset_name.upper()} 对比结果")
    print(sep)

    # 表头
    header = f"  {'指标':<14}"
    for name in display_names:
        header += f"  {name:>{col_width}}"
    print(header)
    print("-" * row_width)

    for label, key, higher_better in metrics:
        values = [last(histories.get(name, {}), key) for name in model_names]
        best = max(values) if higher_better else min(values)

        row_str = f"  {label:<14}"
        for v in values:
            marker = " *" if v == best else "  "
            row_str += f"  {v:>{col_width}.4f}{marker}"
        print(row_str)

    # 额外指标
    print("-" * row_width)

    extra_metrics = [
        ("显存(MB)",     lambda h: h.get("gpu_memory_mb", 0),       ".1f"),
        ("推理FPS",      lambda h: h.get("inference_fps", 0),       ".1f"),
        ("总训练耗时(s)", lambda h: sum(h.get("epoch_time", [])),   ".1f"),
    ]

    for label, extract, fmt in extra_metrics:
        values = [extract(histories.get(name, {})) for name in model_names]
        best = max(values) if "FPS" in label else min(values)

        row_str = f"  {label:<14}"
        for v in values:
            marker = " *" if v == best else "  "
            row_str += f"  {v:>{col_width}{fmt}}{marker}"
        print(row_str)

    print(sep)
    print("  * 表示该指标更优的一方")
    print()
