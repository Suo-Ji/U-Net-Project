# ============================================================
# 损失函数与评估指标
# ============================================================

import torch


class DiceLoss(torch.nn.Module):
    """
    Dice 损失函数
    用于衡量预测与真实标签之间的重叠程度，对类别不平衡问题更鲁棒
    Dice = 2 * |P ∩ G| / (|P| + |G|)
    """

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        # preds: (B, 1, H, W) logits, targets: (B, 1, H, W) 浮点掩膜
        preds = torch.sigmoid(preds)
        preds = preds.view(-1)
        targets = targets.view(-1)

        intersection = (preds * targets).sum()
        dice_score = (2.0 * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        return 1.0 - dice_score


class CombinedLoss(torch.nn.Module):
    """
    组合损失 = BCE Loss + Dice Loss
    BCE 提供像素级分类监督，Dice 缓解类别不平衡问题
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, preds, targets):
        bce_loss = self.bce(preds, targets)
        dice_loss = self.dice(preds, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def compute_iou(preds, targets, threshold=0.5):
    """
    计算交并比 (Intersection over Union)
    IoU = |P ∩ G| / |P ∪ G|
    """
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    targets = (targets > threshold).float()

    intersection = (preds * targets).sum(dim=(2, 3))
    union = ((preds + targets) >= 1).float().sum(dim=(2, 3))

    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean()


def compute_pixel_accuracy(preds, targets, threshold=0.5):
    """
    计算像素准确率 (Pixel Accuracy)
    PA = 正确预测的像素数 / 总像素数
    """
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    targets = (targets > threshold).float()

    correct = (preds == targets).float()
    return correct.mean()
