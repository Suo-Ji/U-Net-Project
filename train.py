# ============================================================
# 训练与验证逻辑
# ============================================================

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import CamVidDataset
from losses import CombinedLoss, compute_iou, compute_pixel_accuracy


def create_dataloaders():
    """创建训练、验证和测试的 DataLoader"""
    train_dataset = CamVidDataset(
        data_dir=config.DATA_DIR, split="train",
        img_size=(config.IMG_HEIGHT, config.IMG_WIDTH), augment=True
    )
    val_dataset = CamVidDataset(
        data_dir=config.DATA_DIR, split="val",
        img_size=(config.IMG_HEIGHT, config.IMG_WIDTH), augment=False
    )
    test_dataset = CamVidDataset(
        data_dir=config.DATA_DIR, split="test",
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
    """训练一个 epoch，返回平均损失、IoU 和像素准确率"""
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    running_acc = 0.0

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
            iou = compute_iou(outputs, masks)
            acc = compute_pixel_accuracy(outputs, masks)

        running_loss += loss.item() * images.size(0)
        running_iou += iou.item() * images.size(0)
        running_acc += acc.item() * images.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{iou.item():.4f}")

    n = len(train_loader.dataset)
    return running_loss / n, running_iou / n, running_acc / n


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """验证一个 epoch，返回平均损失、IoU 和像素准确率"""
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    running_acc = 0.0

    pbar = tqdm(val_loader, desc="  [Val]", leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        iou = compute_iou(outputs, masks)
        acc = compute_pixel_accuracy(outputs, masks)

        running_loss += loss.item() * images.size(0)
        running_iou += iou.item() * images.size(0)
        running_acc += acc.item() * images.size(0)

    n = len(val_loader.dataset)
    return running_loss / n, running_iou / n, running_acc / n


def train(model, device):
    """
    完整训练流程
    返回训练好的模型、测试 DataLoader 和训练历史记录
    """
    print("=" * 70)
    print("开始训练 U-Net 道路分割模型")
    print("=" * 70)

    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders()

    # 损失函数、优化器、学习率调度器
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.LR_SCHEDULER_FACTOR,
        patience=config.LR_SCHEDULER_PATIENCE, verbose=False
    )

    # 训练循环
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_iou": [], "val_acc": []}

    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.NUM_EPOCHS}  |  LR: {optimizer.param_groups[0]['lr']:.2e}")

        train_loss, train_iou, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_iou, val_acc = validate(model, val_loader, criterion, device)

        # 学习率衰减
        scheduler.step(val_loss)

        # 记录训练历史
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)
        history["val_acc"].append(val_acc)

        # 打印本轮结果
        print(f"  Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}  |  "
              f"Val IoU: {val_iou:.4f}  |  Val Acc: {val_acc:.4f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"  >>> 最佳模型已保存 (Val Loss: {val_loss:.4f})")

    print("\n" + "=" * 70)
    print("训练完成！")
    print("=" * 70)

    return model, test_loader, history
