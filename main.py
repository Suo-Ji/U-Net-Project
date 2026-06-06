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
# 运行方式：
#   python main.py <数据集> <模型>                        训练
#   python main.py test <数据集> <模型>                  测试可视化
#   python main.py predict <数据集> <模型> [图片...]      预测
#   python main.py compare <数据集>                       对比两个模型
#
#   <数据集>: camvid 或 cityscapes
#   <模型>:   unet 或 segnet
#
# ============================================================

import os
import random
import sys
import json
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore", message=".*check_version.*")

import config
from model import UNet, SegNet, get_model_class
from train import train, create_dataloaders, measure_inference_speed, measure_gpu_memory, evaluate_on_test
from visualize import (visualize_prediction, plot_training_history,
                       predict_custom_images, compare_training,
                       compare_prediction, print_comparison_table)


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


def _create_model(model_name, device):
    """创建模型并加载到设备"""
    ModelClass = get_model_class(model_name)
    model = ModelClass(in_channels=3, out_channels=1, num_filters=config.NUM_FILTERS)
    model = model.to(device)
    return model


def _load_model(model_name, cfg, device):
    """加载已有模型权重"""
    model = _create_model(model_name, device)
    model.load_state_dict(torch.load(cfg["model_save_path"], map_location=device, weights_only=True))
    model = model.to(device)
    return model


def _print_model_info(model, model_name):
    """打印模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型: {model_name.upper()}  |  参数量: {total_params:,} (可训练: {trainable_params:,})")


def print_usage():
    """打印使用帮助"""
    print("""
使用方式:
  python main.py <数据集> <模型>                         训练模型
  python main.py test <数据集> <模型>                   测试可视化
  python main.py predict <数据集> <模型>                批量预测 images/ 下的图片
  python main.py predict <数据集> <模型> <图片1> <图片2> 预测指定图片
  python main.py compare <数据集>                       对比两个模型

  <数据集>: camvid 或 cityscapes
  <模型>:   unet 或 segnet
""")


def do_train(dataset_name, model_name):
    """训练指定数据集和模型"""
    cfg = config.get_config(dataset_name, model_name)

    set_seed(config.SEED)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 检查数据集目录
    if not os.path.isdir(cfg["data_dir"]):
        print(f"\n错误: 数据集目录不存在: {cfg['data_dir']}")
        return

    # 创建模型
    model = _create_model(model_name, device)
    _print_model_info(model, model_name)

    # 训练（JSON 已在训练过程中实时写入）
    model, val_loader, history = train(model, device, dataset_name, model_name, cfg)

    # 绘制训练历史曲线
    plot_training_history(history, cfg["figure_dir"], model_name)

    # 加载最佳模型，使用验证集进行预测可视化
    print("\n加载最佳模型，使用验证集进行预测可视化...")
    model = _load_model(model_name, cfg, device)
    visualize_prediction(model, val_loader, device, cfg["figure_dir"], model_name, num_samples=3)


def do_test(dataset_name, model_name):
    """加载已有权重，运行测试可视化"""
    cfg = config.get_config(dataset_name, model_name)

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 检查模型权重
    if not os.path.isfile(cfg["model_save_path"]):
        print(f"\n错误: 模型权重不存在: {cfg['model_save_path']}")
        print(f"请先运行 python main.py {dataset_name} {model_name} 完成训练")
        return

    # 检查数据集目录
    if not os.path.isdir(cfg["data_dir"]):
        print(f"\n错误: 数据集目录不存在: {cfg['data_dir']}")
        return

    # 加载模型
    model = _load_model(model_name, cfg, device)
    _print_model_info(model, model_name)

    # 使用验证集进行可视化
    _, val_loader, _ = create_dataloaders(dataset_name, cfg)
    visualize_prediction(model, val_loader, device, cfg["figure_dir"], model_name, num_samples=3)

    # 打印推理速度和显存
    fps = measure_inference_speed(model, val_loader, device)
    gpu_mem = measure_gpu_memory(model, val_loader, device)
    print(f"推理速度: {fps:.1f} FPS  |  显存占用: {gpu_mem:.1f} MB")


def do_predict(dataset_name, model_name):
    """使用已训练好的模型对自定义图片进行道路分割预测"""
    EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp')

    cfg = config.get_config(dataset_name, model_name)

    # 确定图片来源
    # sys.argv: [main.py, predict, <dataset>, <model>, <img1>, <img2>, ...]
    if len(sys.argv) >= 6:
        image_paths = sys.argv[5:]
        for path in image_paths:
            if not os.path.isfile(path):
                print(f"错误: 图片不存在 - {path}")
                return
    else:
        img_dir = os.path.join(config.PROJECT_DIR, "images")
        if not os.path.isdir(img_dir):
            os.makedirs(img_dir, exist_ok=True)
            print(f"已创建图片目录: {img_dir}")
            print("请将待预测的图片放入该目录后重新运行")
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

    # 检查模型权重
    if not os.path.isfile(cfg["model_save_path"]):
        print(f"\n错误: 模型权重不存在 - {cfg['model_save_path']}")
        print(f"请先运行 python main.py {dataset_name} {model_name} 完成训练")
        return

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    model = _load_model(model_name, cfg, device)
    _print_model_info(model, model_name)
    print(f"开始预测...")

    predict_custom_images(model, image_paths, device, cfg["figure_dir"], model_name)


def do_compare(dataset_name):
    """对比两个模型在指定数据集上的表现"""
    from losses import CombinedLoss

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    models = ["unet", "segnet"]

    # 检查模型权重
    for model_name in models:
        cfg = config.get_config(dataset_name, model_name)
        if not os.path.isfile(cfg["model_save_path"]):
            print(f"\n错误: {model_name.upper()} 的模型权重不存在: {cfg['model_save_path']}")
            print(f"请先运行 python main.py {dataset_name} {model_name} 完成训练")
            return

    # 加载模型
    loaded_models = {}
    for model_name in models:
        cfg = config.get_config(dataset_name, model_name)
        loaded_models[model_name] = _load_model(model_name, cfg, device)

    # 创建 test 集 DataLoader
    cfg_ref = config.get_config(dataset_name, "unet")
    _, _, test_loader = create_dataloaders(dataset_name, cfg_ref)
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)

    # 在 test 集上评估两个模型
    print(f"\n在 test 集上评估两个模型...")
    test_results = {}
    for model_name in models:
        print(f"\n  评估 {model_name.upper()}...")
        result = evaluate_on_test(loaded_models[model_name], test_loader, criterion, device)
        fps = measure_inference_speed(loaded_models[model_name], test_loader, device)
        gpu_mem = measure_gpu_memory(loaded_models[model_name], test_loader, device)
        result["test_fps"] = fps
        result["test_gpu_mem"] = gpu_mem
        result["params"] = sum(p.numel() for p in loaded_models[model_name].parameters())
        test_results[model_name] = result

    # 1. 终端打印指标对比表（test 集）
    _print_compare_table(test_results["unet"], test_results["segnet"], dataset_name)

    # 2. 训练曲线对比图（依赖 JSON，有则绘制，无则跳过）
    histories = {}
    for model_name in models:
        cfg = config.get_config(dataset_name, model_name)
        history_path = os.path.join(cfg["figure_dir"], f"{model_name}_history.json")
        if os.path.isfile(history_path):
            with open(history_path, "r") as f:
                histories[model_name] = json.load(f)

    if len(histories) == 2:
        compare_training(histories["unet"], histories["segnet"], dataset_name,
                         cfg_ref["comparison_dir"], name_a="U-Net", name_b="SegNet")
    else:
        missing = [m.upper() for m in models if m not in histories]
        print(f"\n  跳过训练曲线对比（缺少 {', '.join(missing)} 的 history JSON）")

    # 3. 预测结果对比图（test 集）
    compare_prediction(loaded_models["unet"], loaded_models["segnet"], test_loader, device,
                       dataset_name, cfg_ref["comparison_dir"],
                       name_a="U-Net", name_b="SegNet", num_samples=3)


def _print_compare_table(result_a, result_b, dataset_name, name_a="U-Net", name_b="SegNet"):
    """在终端打印两个模型在 test 集上的指标对比表"""
    metrics = [
        ("Loss",       "test_loss",      False),
        ("IoU",        "test_iou",       True),
        ("Pixel Acc",  "test_pixel_acc", True),
        ("Precision",  "test_precision", True),
        ("Recall",     "test_recall",    True),
        ("F1-Score",   "test_f1",        True),
        ("NLL",        "test_nll",       False),
        ("ECE",        "test_ece",       False),
    ]

    print()
    print("=" * 65)
    print(f"  {name_a} vs {name_b} — {dataset_name.upper()} Test 对比结果")
    print("=" * 65)
    print(f"  {'指标':<14} {name_a:>12}   {name_b:>12}   {'差值':>10}")
    print("-" * 65)

    for label, key, higher_better in metrics:
        va = result_a.get(key, 0)
        vb = result_b.get(key, 0)
        diff = vb - va
        sign = "+" if diff >= 0 else ""
        marker = " *" if (higher_better and va > vb) or (not higher_better and va < vb) else "  "
        print(f"  {label:<14} {va:>12.4f}   {vb:>12.4f}   {sign}{diff:>9.4f}{marker}")

    print("-" * 65)
    fps_a = result_a.get("test_fps", 0)
    fps_b = result_b.get("test_fps", 0)
    gpu_a = result_a.get("test_gpu_mem", 0)
    gpu_b = result_b.get("test_gpu_mem", 0)
    p_a = result_a.get("params", 0)
    p_b = result_b.get("params", 0)

    print(f"  {'参数量':<14} {p_a:>12,}   {p_b:>12,}   {sign}{p_b - p_a:>9,}")
    print(f"  {'显存(MB)':<14} {gpu_a:>12.1f}   {gpu_b:>12.1f}   {sign}{gpu_b - gpu_a:>9.1f}")
    print(f"  {'推理FPS':<14} {fps_a:>12.1f}   {fps_b:>12.1f}   {sign}{fps_b - fps_a:>9.1f}")
    print("=" * 65)
    print("  * 表示该指标更优的一方")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(0)

    cmd = sys.argv[1].lower()

    # python main.py <dataset> <model> — 训练
    if cmd in config.VALID_DATASETS:
        if len(sys.argv) < 3 or sys.argv[2].lower() not in config.VALID_MODELS:
            print(f"错误: 请指定模型名称，如: python main.py {cmd} unet")
            print_usage()
            sys.exit(1)
        do_train(cmd, sys.argv[2].lower())

    # python main.py test <dataset> <model>
    elif cmd == "test":
        if len(sys.argv) < 4 or sys.argv[2].lower() not in config.VALID_DATASETS \
                or sys.argv[3].lower() not in config.VALID_MODELS:
            print("错误: 用法: python main.py test <数据集> <模型>")
            print("示例: python main.py test camvid unet")
            print_usage()
            sys.exit(1)
        do_test(sys.argv[2].lower(), sys.argv[3].lower())

    # python main.py predict <dataset> <model> [图片...]
    elif cmd == "predict":
        if len(sys.argv) < 4 or sys.argv[2].lower() not in config.VALID_DATASETS \
                or sys.argv[3].lower() not in config.VALID_MODELS:
            print("错误: 用法: python main.py predict <数据集> <模型> [图片...]")
            print("示例: python main.py predict camvid unet")
            print_usage()
            sys.exit(1)
        do_predict(sys.argv[2].lower(), sys.argv[3].lower())

    # python main.py compare <dataset>
    elif cmd == "compare":
        if len(sys.argv) < 3 or sys.argv[2].lower() not in config.VALID_DATASETS:
            print("错误: 用法: python main.py compare <数据集>")
            print("示例: python main.py compare camvid")
            print("注意: 需要先完成两个模型的训练")
            print_usage()
            sys.exit(1)
        do_compare(sys.argv[2].lower())

    else:
        print(f"错误: 未知命令 '{cmd}'")
        print_usage()
        sys.exit(1)
