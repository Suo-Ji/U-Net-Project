# ============================================================
# U-Net 道路语义分割项目 - 配置文件
# ============================================================

import os

# -------------------- 数据集路径 --------------------
# CamVid 数据集根目录，需包含以下子目录结构：
#   CamVid/
#     train/          - 训练图像
#     train_labels/   - 训练标签（彩色掩膜）
#     val/            - 验证图像
#     val_labels/     - 验证标签
#     test/           - 测试图像
#     test_labels/    - 测试标签
#     class_dict.csv  - 类别定义文件（类别名,R,G,B）
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CamVid")

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

# 模型保存路径
MODEL_SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.pth")
