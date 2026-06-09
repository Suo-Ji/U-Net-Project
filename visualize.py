# ============================================================
# 可视化模块
# 在测试集上随机选取图像，并排展示：输入图像、真实标签、预测结果
# 支持单模型可视化和模型对比可视化
# ============================================================

import os
import random
import numpy as np
import cv2
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
        image, mask, edge = samples[idx]
        image = image.to(device)
        mask = mask.to(device)
        edge = edge.to(device)

        # 推理预测
        with torch.no_grad():
            if getattr(model, 'use_edge', False):
                output = model(image, edge)
            else:
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
    from dataset import compute_edge_map

    model.eval()

    # 判断是否为边缘注意力模型，提取边缘类型
    use_edge = getattr(model, 'use_edge', False)
    if use_edge and "_" in model_name:
        edge_type = model_name.split("_", 1)[1]  # "unet_canny" → "canny"
    else:
        edge_type = None

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
            if use_edge and edge_type:
                raw_resized = cv2.resize(np.array(Image.open(path).convert("RGB")),
                                         (config.IMG_WIDTH, config.IMG_HEIGHT))
                edge_map = compute_edge_map(raw_resized, edge_type)
                edge_tensor = torch.from_numpy(edge_map).unsqueeze(0).unsqueeze(0).to(device)
                pred = torch.sigmoid(model(input_tensor, edge_tensor))
            else:
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

def compare_training(histories, dataset_name, save_dir):
    """
    绘制多个模型（N 个）的训练曲线对比图

    参数:
        histories: dict, {模型显示名: history字典}，如 {"U-Net": {...}, "U-Net+Canny": {...}}
        dataset_name: 数据集名称
        save_dir: 保存目录
    """
    # 为每个模型分配独特的颜色和标记样式
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    markers = ["o", "s", "^", "D", "v", "P"]

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

    fig, axes = plt.subplots(3, 3, figsize=(22, 16))
    axes = axes.flatten()

    for i, (key, title) in enumerate(metrics_to_plot):
        for j, (name, history) in enumerate(histories.items()):
            if key in history and len(history[key]) > 0:
                epochs = range(1, len(history[key]) + 1)
                axes[i].plot(epochs, history[key],
                             label=name,
                             color=colors[j % len(colors)],
                             marker=markers[j % len(markers)],
                             markersize=3,
                             linewidth=1.5)
        axes[i].set_title(title, fontsize=13, fontweight="bold")
        axes[i].set_xlabel("Epoch", fontsize=11)
        axes[i].legend(fontsize=9)
        axes[i].grid(True, alpha=0.3)

    model_names = " vs ".join(histories.keys())
    plt.suptitle(f"{model_names} — {dataset_name.upper()} Training Curves",
                 fontsize=17, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{dataset_name}_compare_training.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"训练曲线对比已保存至 {save_path}")
    plt.show()


def compare_prediction(models_dict, val_loader, device, dataset_name, save_dir, num_samples=3):
    """
    同一张图上并排展示多个模型的预测结果

    参数:
        models_dict: dict, {模型显示名: model对象}，如 {"U-Net": model1, "U-Net+Canny": model2, ...}
        val_loader: 验证集 DataLoader
        device: 计算设备
        dataset_name: 数据集名称
        save_dir: 保存目录
        num_samples: 展示的样本数量
    """
    # 设置所有模型为评估模式
    for model in models_dict.values():
        model.eval()

    samples = list(val_loader)
    selected = random.sample(range(len(samples)), min(num_samples, len(samples)))

    num_models = len(models_dict)
    # 列数: 2 (Input + GT) + N 个模型预测
    num_cols = 2 + num_models
    model_names = list(models_dict.keys())

    fig, axes = plt.subplots(num_samples, num_cols, figsize=(5 * num_cols, 5 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(selected):
        image, mask, edge = samples[idx]
        image = image.to(device)
        mask = mask.to(device)
        edge = edge.to(device)

        img_np = denormalize(image[0])
        mask_np = mask[0, 0].cpu().numpy()

        # Input Image
        axes[row, 0].imshow(img_np)
        axes[row, 0].set_title("Input Image", fontsize=12)
        axes[row, 0].axis("off")

        # Ground Truth
        axes[row, 1].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("Ground Truth", fontsize=12)
        axes[row, 1].axis("off")

        # 每个模型的预测结果
        for col, (name, model) in enumerate(models_dict.items(), start=2):
            with torch.no_grad():
                if getattr(model, 'use_edge', False):
                    output = model(image, edge)
                else:
                    output = model(image)
                pred = torch.sigmoid(output)
                pred_mask = (pred > 0.5).float()

            pred_np = pred_mask[0, 0].cpu().numpy()
            axes[row, col].imshow(pred_np, cmap="gray", vmin=0, vmax=1)
            axes[row, col].set_title(name, fontsize=12)
            axes[row, col].axis("off")

    title_text = " vs ".join(model_names)
    plt.suptitle(f"{title_text} — {dataset_name.upper()} Prediction Comparison",
                 fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{dataset_name}_compare_prediction.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"预测对比已保存至 {save_path}")
    plt.show()


def print_comparison_table(histories, dataset_name):
    """
    在终端打印多个模型的指标对比表

    参数:
        histories: dict, {模型显示名: history字典}
        dataset_name: 数据集名称
    """
    def last(h, key):
        vals = h.get(key, [])
        return vals[-1] if vals else 0.0

    model_names = list(histories.keys())
    n = len(model_names)

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

    # 动态计算列宽
    name_w = max(len(nm) for nm in model_names)
    col_w = max(name_w + 2, 12)
    metric_w = 14

    sep_total = metric_w + 1 + (col_w + 1) * n + 4  # 4 for " best"
    line = " " + "=" * sep_total
    dash = " " + "-" * sep_total

    print()
    print(line)
    title = " vs ".join(model_names)
    print(f"  {title} — {dataset_name.upper()} Training History 对比")
    print(line)

    # 表头
    header = f"  {'指标':<{metric_w}}"
    for name in model_names:
        header += f" {name:>{col_w}}"
    header += "  best"
    print(header)
    print(dash)

    for label, key, higher_better in metrics:
        vals = {name: last(histories[name], key) for name in model_names}
        # 找出最优值
        if higher_better:
            best_name = max(vals, key=vals.get)
        else:
            best_name = min(vals, key=vals.get)

        row = f"  {label:<{metric_w}}"
        for name in model_names:
            marker = " *" if name == best_name else "  "
            row += f" {vals[name]:>{col_w}.4f}{marker}"
        print(row)

    # 额外指标
    print(dash)

    extra_metrics = [
        ("显存(MB)",     lambda h: h.get("gpu_memory_mb", 0),     ".1f"),
        ("推理FPS",      lambda h: h.get("inference_fps", 0),     ".1f"),
        ("总训练耗时(s)", lambda h: sum(h.get("epoch_time", [])), ".1f"),
    ]

    for label, getter, fmt in extra_metrics:
        vals = {name: getter(histories[name]) for name in model_names}
        # 显存和耗时越低越好，FPS越高越好
        if "FPS" in label:
            best_name = max(vals, key=vals.get)
        else:
            best_name = min(vals, key=vals.get)

        row = f"  {label:<{metric_w}}"
        for name in model_names:
            marker = " *" if name == best_name else "  "
            row += f" {vals[name]:>{col_w}{fmt}}{marker}"
        print(row)

    print(line)
    print("  * 表示该指标最优的模型")
    print()


def visualize_edge_detection(image_dir, save_dir):
    """
    对指定目录下的图片生成 Canny / Sobel / Laplacian 三种边缘检测结果的可视化对比图

    参数:
        image_dir: 图片目录路径（扫描其中所有 png/jpg/jpeg/bmp 文件）
        save_dir:  图片保存目录
    """
    from dataset import compute_edge_map

    EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp')

    # 收集图片
    image_paths = sorted([
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if f.lower().endswith(EXTENSIONS)
    ])
    if not image_paths:
        print(f"目录 {image_dir} 中没有找到图片")
        return

    n = len(image_paths)
    # 4 列：原图 / Canny / Sobel / Laplacian
    fig, axes = plt.subplots(n, 4, figsize=(24, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    edge_types = [
        ("canny", "Canny"),
        ("sobel", "Sobel"),
        ("laplacian", "Laplacian"),
    ]

    for row, path in enumerate(image_paths):
        image_rgb = np.array(Image.open(path).convert("RGB"))
        image_rgb = cv2.resize(image_rgb, (config.IMG_WIDTH, config.IMG_HEIGHT))

        # 原图
        axes[row, 0].imshow(image_rgb)
        axes[row, 0].set_title(f"Input: {os.path.basename(path)}", fontsize=13)
        axes[row, 0].axis("off")

        # 三种边缘检测结果
        for col, (edge_type, title) in enumerate(edge_types, start=1):
            edge_map = compute_edge_map(image_rgb, edge_type)
            axes[row, col].imshow(edge_map, cmap="gray", vmin=0, vmax=1)
            axes[row, col].set_title(title, fontsize=13)
            axes[row, col].axis("off")

    plt.suptitle("Edge Detection Comparison — Canny vs Sobel vs Laplacian",
                 fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "edge_detection_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"边缘检测对比已保存至 {save_path}")
    plt.show()
