# ============================================================
# 训练与验证逻辑
# ============================================================

import json
import os
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import get_dataset_class
from losses import CombinedLoss, compute_all_metrics


def create_dataloaders(dataset_name, cfg):
    """
    创建训练、验证和测试的 DataLoader

    参数:
        dataset_name: 数据集名称 ("camvid" 或 "cityscapes")
        cfg: 数据集配置字典（由 config.get_config() 返回）
    """
    DatasetClass = get_dataset_class(dataset_name)

    train_dataset = DatasetClass(
        data_dir=cfg["data_dir"], split="train",
        img_size=(config.IMG_HEIGHT, config.IMG_WIDTH), augment=True
    )
    val_dataset = DatasetClass(
        data_dir=cfg["data_dir"], split="val",
        img_size=(config.IMG_HEIGHT, config.IMG_WIDTH), augment=False
    )
    test_dataset = DatasetClass(
        data_dir=cfg["data_dir"], split="test",
        img_size=(config.IMG_HEIGHT, config.IMG_WIDTH), augment=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE,
        shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1,
        shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个 epoch，返回平均损失和各项指标"""
    model.train()
    running_loss = 0.0
    running_metrics = {"iou": 0.0, "pixel_acc": 0.0, "precision": 0.0,
                       "recall": 0.0, "f1": 0.0, "nll": 0.0, "ece": 0.0}

    pbar = tqdm(train_loader, desc="  [Train]", leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, masks)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算指标
        with torch.no_grad():
            metrics = compute_all_metrics(outputs, masks)

        running_loss += loss.item() * images.size(0)
        for k in running_metrics:
            running_metrics[k] += metrics[k].item() * images.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{metrics['iou'].item():.4f}")

    n = len(train_loader.dataset)
    avg_metrics = {k: v / n for k, v in running_metrics.items()}
    return running_loss / n, avg_metrics


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """验证一个 epoch，返回平均损失和各项指标"""
    model.eval()
    running_loss = 0.0
    running_metrics = {"iou": 0.0, "pixel_acc": 0.0, "precision": 0.0,
                       "recall": 0.0, "f1": 0.0, "nll": 0.0, "ece": 0.0}

    pbar = tqdm(val_loader, desc="  [Val]", leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        metrics = compute_all_metrics(outputs, masks)

        running_loss += loss.item() * images.size(0)
        for k in running_metrics:
            running_metrics[k] += metrics[k].item() * images.size(0)

    n = len(val_loader.dataset)
    avg_metrics = {k: v / n for k, v in running_metrics.items()}
    return running_loss / n, avg_metrics


def measure_inference_speed(model, val_loader, device, num_batches=50):
    """
    测量推理速度 (FPS)
    返回: 每秒处理的图片数
    """
    model.eval()
    total_images = 0
    start_time = time.time()

    with torch.no_grad():
        for i, (images, _) in enumerate(val_loader):
            if i >= num_batches:
                break
            images = images.to(device)
            _ = model(images)
            total_images += images.size(0)

    elapsed = time.time() - start_time
    fps = total_images / elapsed if elapsed > 0 else 0
    return fps


def measure_gpu_memory(model, val_loader, device):
    """
    测量 GPU 峰值显存占用 (MB)
    """
    if not torch.cuda.is_available():
        return 0.0

    torch.cuda.reset_peak_memory_stats(device)
    model.eval()

    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device)
            _ = model(images)
            break  # 只需要一个 batch 即可测量显存

    peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return peak_mem_mb


@torch.no_grad()
def evaluate_on_test(model, test_loader, criterion, device):
    """
    在 test 集上进行全量评估，返回所有指标的均值

    参数:
        model: 模型实例
        test_loader: 测试集 DataLoader
        criterion: 损失函数
        device: 计算设备

    返回:
        dict: 包含 test_loss, test_iou 等指标均值
    """
    model.eval()
    running_loss = 0.0
    running_metrics = {"iou": 0.0, "pixel_acc": 0.0, "precision": 0.0,
                       "recall": 0.0, "f1": 0.0, "nll": 0.0, "ece": 0.0}

    pbar = tqdm(test_loader, desc="  [Test]", leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)
        metrics = compute_all_metrics(outputs, masks)

        running_loss += loss.item() * images.size(0)
        for k in running_metrics:
            running_metrics[k] += metrics[k].item() * images.size(0)

    n = len(test_loader.dataset)
    result = {"test_loss": running_loss / n}
    for k in running_metrics:
        result[f"test_{k}"] = running_metrics[k] / n
    return result


def train(model, device, dataset_name, model_name, cfg):
    """
    完整训练流程

    参数:
        model: 模型实例
        device: 计算设备
        dataset_name: 数据集名称
        model_name: 模型名称
        cfg: 数据集配置字典

    返回训练好的模型、验证 DataLoader 和训练历史记录
    """
    print("=" * 70)
    print(f"开始训练 {model_name.upper()} 道路分割模型 ({dataset_name})")
    print("=" * 70)

    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders(dataset_name, cfg)

    # 损失函数、优化器、学习率调度器
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.LR_SCHEDULER_FACTOR,
        patience=config.LR_SCHEDULER_PATIENCE, verbose=False
    )

    # 训练循环
    best_val_loss = float("inf")

    # history 包含每个 epoch 的所有指标
    history_keys = ["train_loss", "val_loss",
                    "train_iou", "val_iou", "train_pixel_acc", "val_pixel_acc",
                    "train_precision", "val_precision", "train_recall", "val_recall",
                    "train_f1", "val_f1", "train_nll", "val_nll", "train_ece", "val_ece",
                    "epoch_time", "lr"]
    history = {k: [] for k in history_keys}

    # 测量推理速度和显存（训练前）
    gpu_mem = measure_gpu_memory(model, val_loader, device)
    fps = measure_inference_speed(model, val_loader, device)
    history["gpu_memory_mb"] = gpu_mem
    history["inference_fps"] = fps
    print(f"  初始显存占用: {gpu_mem:.1f} MB | 初始推理速度: {fps:.1f} FPS")

    for epoch in range(1, config.NUM_EPOCHS + 1):
        epoch_start = time.time()

        print(f"\nEpoch {epoch}/{config.NUM_EPOCHS}  |  LR: {optimizer.param_groups[0]['lr']:.2e}")

        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)

        # 学习率衰减
        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start

        # 记录训练历史
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        for k in train_metrics:
            history[f"train_{k}"].append(train_metrics[k])
            history[f"val_{k}"].append(val_metrics[k])
        history["epoch_time"].append(epoch_time)
        history["lr"].append(optimizer.param_groups[0]['lr'])

        # 实时追加写入 JSON（训练中断也不丢失数据）
        history_path = os.path.join(cfg["figure_dir"], f"{model_name}_history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        # 打印本轮结果
        print(f"  Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}  |  "
              f"Val IoU: {val_metrics['iou']:.4f}  |  Val Acc: {val_metrics['pixel_acc']:.4f}  |  "
              f"Val F1: {val_metrics['f1']:.4f}  |  Time: {epoch_time:.1f}s")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), cfg["model_save_path"])
            print(f"  >>> 最佳模型已保存 (Val Loss: {val_loss:.4f})")

    # 训练结束后再次测量推理速度和显存
    history["inference_fps"] = measure_inference_speed(model, val_loader, device)
    history["gpu_memory_mb"] = measure_gpu_memory(model, val_loader, device)
    print(f"\n  最终显存占用: {history['gpu_memory_mb']:.1f} MB | 最终推理速度: {history['inference_fps']:.1f} FPS")

    print("\n" + "=" * 70)
    print("训练完成！")
    print("=" * 70)

    return model, val_loader, history
