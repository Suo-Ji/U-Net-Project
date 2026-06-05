# ============================================================
# U-Net 道路语义分割项目 - 配置文件
# ============================================================

import os

# -------------------- 项目根目录 --------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------- 图像尺寸 --------------------
IMG_HEIGHT = 256
IMG_WIDTH = 256

# -------------------- 训练超参数 --------------------
BATCH_SIZE = 8
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-8

# 学习率调度器参数
LR_SCHEDULER_FACTOR = 0.5       # 衰减因子
LR_SCHEDULER_PATIENCE = 5       # 容忍轮数

# -------------------- 模型参数 --------------------
NUM_FILTERS = 64                 # 第一层卷积滤波器数量，后续层依次翻倍

# -------------------- 数据增强参数 --------------------
# 训练时应用的增强策略
AUGMENT_PROB = 0.5               # 每种增强的触发概率

# -------------------- 其他 --------------------
NUM_WORKERS = 4                  # DataLoader 工作线程数
SEED = 42                        # 随机种子，确保可复现
DEVICE = "cuda"                  # 使用的设备 ("cuda" 或 "cpu")

# -------------------- 支持的数据集和模型 --------------------
VALID_DATASETS = ("camvid", "cityscapes")
VALID_MODELS = ("unet", "segnet", "deeplabv3plus")


def get_config(dataset_name, model_name="unet"):
    """
    根据数据集名称和模型名称返回对应的配置字典

    参数:
        dataset_name: 数据集名称，"camvid" 或 "cityscapes"
        model_name: 模型名称，"unet" 或 "segnet"

    返回:
        dict: 包含 data_dir, model_save_path, figure_dir 等配置
    """
    dataset_name = dataset_name.lower()
    model_name = model_name.lower()

    if dataset_name not in VALID_DATASETS:
        raise ValueError(f"不支持的数据集: {dataset_name}，可选: {VALID_DATASETS}")
    if model_name not in VALID_MODELS:
        raise ValueError(f"不支持的模型: {model_name}，可选: {VALID_MODELS}")

    if dataset_name == "camvid":
        data_dir = os.path.join(PROJECT_DIR, "CamVid")
    else:  # cityscapes
        data_dir = os.path.join(PROJECT_DIR, "Cityscapes")

    # 模型权重保存目录: checkpoints/{dataset}/
    checkpoint_dir = os.path.join(PROJECT_DIR, "checkpoints", dataset_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 可视化图片保存目录: figure/{dataset}/
    figure_dir = os.path.join(PROJECT_DIR, "figure", dataset_name)
    os.makedirs(figure_dir, exist_ok=True)

    # 对比可视化目录: figure/comparison/
    comparison_dir = os.path.join(PROJECT_DIR, "figure", "comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    return {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "data_dir": data_dir,
        "model_save_path": os.path.join(checkpoint_dir, f"{model_name}_best_model.pth"),
        "figure_dir": figure_dir,
        "comparison_dir": comparison_dir,
    }
