# ============================================================
# U-Net 道路语义分割项目 - 主入口
# ============================================================
#
# 所需第三方库安装指令：
#   pip install torch torchvision
#   pip install albumentations
#   pip install opencv-python

#   pip install matplotlib
#   pip install tqdm
#   pip install pillow
#
# 数据集说明：
#   请下载 CamVid 数据集并放置于项目根目录的 CamVid/ 文件夹中
#   目录结构应为：
#     CamVid/
#       train/          - 训练图像 (.png)
#       trainannot/     - 训练标签（灰度索引图，Road 类别索引 = 3）
#       val/            - 验证图像
#       valannot/       - 验证标签
#       test/           - 测试图像
#       testannot/      - 测试标签
#
#   CamVid 数据集下载地址：
#     https://www.kaggle.com/datasets/carlolepelaars/camvid
#     或
#     http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/
#
# ============================================================

import os
import random
import sys

import numpy as np
import torch

import config
from model import UNet
from train import train
from visualize import visualize_prediction, plot_training_history, predict_custom_images


def set_seed(seed):
    """设置随机种子以确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    # 设置随机种子
    set_seed(config.SEED)

    # 选择设备
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 检查数据集目录是否存在
    if not os.path.isdir(config.DATA_DIR):
        print(f"\n错误: 数据集目录不存在: {config.DATA_DIR}")
        print("请先下载 CamVid 数据集并解压到项目根目录的 CamVid/ 文件夹中。")
        print("下载地址: https://www.kaggle.com/datasets/carlolepelaars/camvid")
        return

    # 创建 U-Net 模型
    model = UNet(in_channels=3, out_channels=1, num_filters=config.NUM_FILTERS)
    model = model.to(device)

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params:,} (可训练: {trainable_params:,})")

    # ========== 训练 ==========
    model, test_loader, history = train(model, device)

    # 绘制训练历史曲线
    plot_training_history(history)

    # 加载最佳模型进行测试可视化
    print("\n加载最佳模型进行预测可视化...")
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device, weights_only=True))
    model = model.to(device)

    # ========== 可视化测试结果 ==========
    visualize_prediction(model, test_loader, device, num_samples=3)


def predict():
    """
    使用已训练好的模型对自定义图片进行道路分割预测

    用法：
        # 方式一：将图片放入项目目录下的 images/ 文件夹，然后运行：
        python main.py predict

        # 方式二：直接指定图片路径：
        python main.py predict photo1.jpg photo2.png
    """
    EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp')

    # 确定图片来源：命令行传入 或 自动扫描 images/ 目录
    if len(sys.argv) >= 3:
        image_paths = sys.argv[2:]
        for path in image_paths:
            if not os.path.isfile(path):
                print(f"错误: 图片不存在 - {path}")
                return
    else:
        img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
        if not os.path.isdir(img_dir):
            os.makedirs(img_dir, exist_ok=True)
            print(f"已创建图片目录: {img_dir}")
            print("请将待预测的图片放入该目录后重新运行 python main.py predict")
            return
        image_paths = sorted([
            os.path.join(img_dir, f) for f in os.listdir(img_dir)
            if f.lower().endswith(EXTENSIONS)
        ])
        if not image_paths:
            print(f"images/ 目录中没有图片，请放入图片后重新运行")
            return

    print(f"找到 {len(image_paths)} 张图片:")
    for p in image_paths:
        print(f"  - {os.path.basename(p)}")

    # 检查模型权重是否存在
    if not os.path.isfile(config.MODEL_SAVE_PATH):
        print(f"\n错误: 模型权重不存在 - {config.MODEL_SAVE_PATH}")
        print("请先运行 python main.py 完成训练")
        return

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=1, num_filters=config.NUM_FILTERS)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device, weights_only=True))
    model = model.to(device)
    print(f"模型已加载 ({device})，开始预测...")

    predict_custom_images(model, image_paths, device)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "predict":
        predict()
    else:
        main()
