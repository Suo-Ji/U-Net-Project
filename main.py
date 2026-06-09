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
#   python main.py compare <数据集>                       对比所有模型
#
#   <数据集>: camvid 或 cityscapes
#   <模型>:   unet, unet_canny, unet_sobel 或 unet_laplacian
#
# ============================================================


import os
import random
import sys
import json
import warnings
warnings.filterwarnings("ignore", message=".*check_version.*")
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import numpy as np
import torch

import config
from model import UNet, UNetEdgeAttention, get_model_class
from train import train, create_dataloaders, measure_inference_speed, measure_gpu_memory, evaluate_on_test
from visualize import (visualize_prediction, plot_training_history,
                       predict_custom_images, compare_training,
                       compare_prediction, print_comparison_table,
                       visualize_edge_detection)



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
  python main.py compare <数据集>                       对比所有模型
  python main.py edge                                   显示边缘检测对比图

  <数据集>: camvid 或 cityscapes
  <模型>:   unet
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
    """对比所有模型在指定数据集上的表现"""
    from losses import CombinedLoss

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    # 打印环境配置信息
    print("=" * 60)
    print("  环境配置信息")
    print("=" * 60)
    print(f"  Python:       {sys.version.split()[0]}")
    print(f"  PyTorch:      {torch.__version__}")
    print(f"  CUDA:         {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    if torch.cuda.is_available():
        print(f"  GPU:          {torch.cuda.get_device_name(0)}")
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU 显存:     {gpu_total:.1f} GB")
    print(f"  计算设备:     {device}")
    print(f"  图像尺寸:     {config.IMG_HEIGHT}×{config.IMG_WIDTH}")
    print(f"  Batch Size:   {config.BATCH_SIZE}")
    print(f"  Epochs:       {config.NUM_EPOCHS}")
    print(f"  学习率:       {config.LEARNING_RATE}")
    print(f"  随机种子:     {config.SEED}")
    print("=" * 60)

    models = list(config.VALID_MODELS)
    display_names = {
        "unet": "U-Net",
        "unet_canny": "U-Net+Canny",
        "unet_sobel": "U-Net+Sobel",
        "unet_laplacian": "U-Net+Laplacian",
    }

    # 检查模型权重
    for model_name in models:
        cfg = config.get_config(dataset_name, model_name)
        if not os.path.isfile(cfg["model_save_path"]):
            print(f"\n错误: {display_names[model_name]} 的模型权重不存在: {cfg['model_save_path']}")
            print(f"请先运行 python main.py {dataset_name} {model_name} 完成训练")
            return

    # 加载模型
    loaded_models = {}
    for model_name in models:
        cfg = config.get_config(dataset_name, model_name)
        loaded_models[model_name] = _load_model(model_name, cfg, device)

    # 创建 DataLoader：CamVid 使用 test 集，Cityscapes 使用 val 集（test 集 GT 不完整）
    cfg_ref = config.get_config(dataset_name, models[0])
    _, val_loader, test_loader = create_dataloaders(dataset_name, cfg_ref)
    eval_loader = val_loader if dataset_name == "cityscapes" else test_loader
    eval_set_name = "val" if dataset_name == "cityscapes" else "test"
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)

    # 在评估集上评估所有模型
    print(f"\n在 {eval_set_name} 集上评估所有模型...")
    test_results = {}
    for model_name in models:
        print(f"\n  评估 {display_names[model_name]}...")
        result = evaluate_on_test(loaded_models[model_name], eval_loader, criterion, device)
        fps = measure_inference_speed(loaded_models[model_name], eval_loader, device)
        gpu_mem = measure_gpu_memory(loaded_models[model_name], eval_loader, device)
        result["test_fps"] = fps
        result["test_gpu_mem"] = gpu_mem
        result["params"] = sum(p.numel() for p in loaded_models[model_name].parameters())
        test_results[model_name] = result

    # 1. 终端打印指标对比表（test 集，所有模型统一对比）
    _print_compare_table(test_results, dataset_name, display_names)

    # 2. 训练曲线对比图（依赖 JSON，有则绘制，无则跳过）
    histories = {}
    for model_name in models:
        cfg = config.get_config(dataset_name, model_name)
        history_path = os.path.join(cfg["figure_dir"], f"{model_name}_history.json")
        if os.path.isfile(history_path):
            with open(history_path, "r") as f:
                histories[display_names[model_name]] = json.load(f)

    if len(histories) >= 2:
        # 训练曲线对比仅绘制 U-Net vs U-Net+Laplacian
        curve_compare_keys = [display_names["unet"], display_names["unet_laplacian"]]
        curve_histories = {k: histories[k] for k in curve_compare_keys if k in histories}
        if len(curve_histories) >= 2:
            compare_training(curve_histories, dataset_name, cfg_ref["comparison_dir"])
        else:
            compare_training(histories, dataset_name, cfg_ref["comparison_dir"])
    else:
        print(f"\n  跳过训练曲线对比（需要至少 2 个模型的 history JSON）")

    # 3. 预测结果对比图（评估集，所有模型统一对比）
    if len(loaded_models) >= 2:
        models_for_compare = {display_names[k]: v for k, v in loaded_models.items()}
        compare_prediction(models_for_compare, eval_loader, device,
                           dataset_name, cfg_ref["comparison_dir"], num_samples=3)
    else:
        print(f"\n  跳过预测对比（需要至少 2 个训练好的模型）")


def _print_compare_table(test_results, dataset_name, display_names):
    """在终端打印所有模型在 test 集上的统一对比表"""
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

    model_keys = list(test_results.keys())
    names = [display_names[k] for k in model_keys]
    n = len(names)

    # 动态计算列宽
    name_w = max(len(nm) for nm in names)
    col_w = max(name_w + 2, 12)
    metric_w = 14

    sep_total = metric_w + 1 + (col_w + 2) * n + 3  # +2 for " *", +3 for " best"
    line = " " + "=" * sep_total
    dash = " " + "-" * sep_total

    print()
    print(line)
    title = " vs ".join(names)
    print(f"  {title} — {dataset_name.upper()} Test 对比结果")
    print(line)

    # 表头
    header = f"  {'指标':<{metric_w}}"
    for name in names:
        header += f" {name:>{col_w}}"
    header += " best"
    print(header)
    print(dash)

    for label, key, higher_better in metrics:
        vals = {k: test_results[k].get(key, 0) for k in model_keys}
        # 找出最优
        if higher_better:
            best_key = max(vals, key=vals.get)
        else:
            best_key = min(vals, key=vals.get)

        row = f"  {label:<{metric_w}}"
        for k in model_keys:
            marker = " *" if k == best_key else "  "
            row += f" {vals[k]:>{col_w}.4f}{marker}"
        print(row)

    # 额外指标
    print(dash)

    extra = [
        ("参数量",     "params",       ",",   False),  # 参数量越少越好
        ("显存(MB)",   "test_gpu_mem", ".1f",  False),
        ("推理FPS",    "test_fps",     ".1f",  True),
    ]

    for label, key, fmt, higher_better in extra:
        vals = {k: test_results[k].get(key, 0) for k in model_keys}
        if higher_better:
            best_key = max(vals, key=vals.get)
        else:
            best_key = min(vals, key=vals.get)

        row = f"  {label:<{metric_w}}"
        for k in model_keys:
            marker = " *" if k == best_key else "  "
            if fmt == ",":
                row += f" {int(vals[k]):>{col_w},}{marker}"
            else:
                row += f" {vals[k]:>{col_w}{fmt}}{marker}"
        print(row)

    print(line)
    print("  * 表示该指标最优的模型")
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
            print("注意: 需要先完成所有模型的训练")
            print_usage()
            sys.exit(1)
        do_compare(sys.argv[2].lower())

    # python main.py edge
    elif cmd == "edge":
        image_dir = os.path.join(config.PROJECT_DIR, "images")
        if not os.path.isdir(image_dir) or not os.listdir(image_dir):
            print(f"错误: images/ 目录中没有图片")
            sys.exit(1)
        comparison_dir = os.path.join(config.PROJECT_DIR, "figure", "comparison")
        visualize_edge_detection(image_dir, comparison_dir)

    else:
        print(f"错误: 未知命令 '{cmd}'")
        print_usage()
        sys.exit(1)
