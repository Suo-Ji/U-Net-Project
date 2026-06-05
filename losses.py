# ============================================================
# 损失函数与评估指标
# 指标: Loss, IoU, Pixel Accuracy, Precision, Recall, F1, NLL, ECE
# ============================================================

import torch


# ============================================================
# 损失函数
# ============================================================

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


# ============================================================
# 评估指标（共享二值化辅助函数）
# ============================================================

def _binarize(preds, targets, threshold=0.5):
    """将 logits 预测和 targets 二值化，返回扁平化的预测和真实标签"""
    probs = torch.sigmoid(preds)
    pred_binary = (probs > threshold).float()
    target_binary = (targets > threshold).float()
    return pred_binary, target_binary


def compute_iou(preds, targets, threshold=0.5):
    """
    计算交并比 (Intersection over Union)
    IoU = |P ∩ G| / |P ∪ G|
    """
    pred_binary, target_binary = _binarize(preds, targets, threshold)

    intersection = (pred_binary * target_binary).sum(dim=(2, 3))
    union = ((pred_binary + target_binary) >= 1).float().sum(dim=(2, 3))

    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean()


def compute_pixel_accuracy(preds, targets, threshold=0.5):
    """
    计算像素准确率 (Pixel Accuracy)
    PA = 正确预测的像素数 / 总像素数
    """
    pred_binary, target_binary = _binarize(preds, targets, threshold)
    correct = (pred_binary == target_binary).float()
    return correct.mean()


def compute_precision(preds, targets, threshold=0.5):
    """
    计算精确率 (Precision)
    Precision = TP / (TP + FP)
    预测为道路的像素中，真正是道路的比例
    """
    pred_binary, target_binary = _binarize(preds, targets, threshold)

    tp = (pred_binary * target_binary).sum()
    fp = (pred_binary * (1 - target_binary)).sum()

    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    return precision


def compute_recall(preds, targets, threshold=0.5):
    """
    计算召回率 (Recall)
    Recall = TP / (TP + FN)
    真实道路像素中，被正确预测出来的比例
    """
    pred_binary, target_binary = _binarize(preds, targets, threshold)

    tp = (pred_binary * target_binary).sum()
    fn = ((1 - pred_binary) * target_binary).sum()

    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    return recall


def compute_f1(preds, targets, threshold=0.5):
    """
    计算 F1 分数
    F1 = 2 * Precision * Recall / (Precision + Recall)
    """
    precision = compute_precision(preds, targets, threshold)
    recall = compute_recall(preds, targets, threshold)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return f1


def compute_nll(preds, targets):
    """
    计算负对数似然 (Negative Log-Likelihood)
    NLL = -mean[y*log(p) + (1-y)*log(1-p)]
    衡量模型输出概率与真实标签的匹配程度
    """
    probs = torch.sigmoid(preds)
    probs = torch.clamp(probs, 1e-6, 1 - 1e-6)  # 避免 log(0)

    nll = -(targets * torch.log(probs) + (1 - targets) * torch.log(1 - probs))
    return nll.mean()


def compute_ece(preds, targets, n_bins=15):
    """
    计算期望校准误差 (Expected Calibration Error)
    ECE = Σ (n_b / N) * |acc_b - conf_b|
    衡量模型置信度与实际准确率的匹配程度，越低越好
    """
    probs = torch.sigmoid(preds).view(-1)
    targets_flat = (targets.view(-1) > 0.5).float()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    ece = torch.tensor(0.0, device=probs.device)

    for i in range(n_bins):
        low = bin_boundaries[i]
        high = bin_boundaries[i + 1]

        # 找出落在当前 bin 的样本
        if i == n_bins - 1:
            in_bin = (probs >= low) & (probs <= high)
        else:
            in_bin = (probs >= low) & (probs < high)

        n_b = in_bin.sum().float()
        if n_b > 0:
            acc_b = targets_flat[in_bin].mean()        # 实际准确率
            conf_b = probs[in_bin].mean()               # 平均置信度
            ece += (n_b / probs.numel()) * torch.abs(acc_b - conf_b)

    return ece


def compute_all_metrics(preds, targets, threshold=0.5):
    """
    一次性计算所有评估指标，避免重复二值化

    返回: dict 包含 iou, pixel_acc, precision, recall, f1, nll, ece
    """
    return {
        "iou": compute_iou(preds, targets, threshold),
        "pixel_acc": compute_pixel_accuracy(preds, targets, threshold),
        "precision": compute_precision(preds, targets, threshold),
        "recall": compute_recall(preds, targets, threshold),
        "f1": compute_f1(preds, targets, threshold),
        "nll": compute_nll(preds, targets),
        "ece": compute_ece(preds, targets),
    }
