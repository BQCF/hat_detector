# train.py
# 训练脚本：在 hat_detector 数据集上微调 YOLOv8
# 用法：激活虚拟环境后，在 hat_detector 目录下执行：
#   python train.py                          # 用本项目 data.yaml 训练
#   python train.py --data coco8.yaml        # 先用官方小数据集跑通流程（无需真实数据）
#   python train.py --epochs 100             # 自定义训练轮数
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

# 项目根目录（本文件所在目录），保证从任意目录运行都指向正确的文件
BASE_DIR = Path(__file__).resolve().parent
DATA_YAML = BASE_DIR / "data.yaml"            # 数据集配置文件
MODEL_PATH = BASE_DIR / "yolov8n.pt"          # 预训练权重（首次运行自动下载到项目目录）
PROJECT_DIR = BASE_DIR / "runs" / "detect"    # 训练结果保存根目录
EXPERIMENT_NAME = "hat_detector"              # 本次实验名 -> runs/detect/hat_detector/


def check_dataset() -> None:
    """训练前校验数据集目录结构，缺失时给出可操作的提示，避免训练到一半才报错。"""
    ds_root = BASE_DIR / "data" / "dataset"
    needed = {
        "images/train": "训练集图片",
        "images/val": "验证集图片",
        "labels/train": "训练集标注",
        "labels/val": "验证集标注",
    }
    missing = [f"  - {rel}（{desc}）" for rel, desc in needed.items() if not (ds_root / rel).is_dir()]
    if missing:
        print(f"[错误] 数据集目录 {ds_root} 缺少以下子目录：")
        print("\n".join(missing))
        print("\n请把数据集整理成如下结构后重试：")
        print("  data/dataset/images/train/  data/dataset/images/val/   图片")
        print("  data/dataset/labels/train/  data/dataset/labels/val/   与图片同名的 .txt 标注")
        print("（Roboflow 导出的是 valid 目录时，改名为 val，或同步修改 data.yaml 的 val 路径）")
        raise SystemExit(1)


def main() -> None:
    # 0. 命令行参数：--data 指定数据集配置，--epochs 指定训练轮数
    parser = argparse.ArgumentParser(description="在 hat_detector 数据集上微调 YOLOv8")
    parser.add_argument("--data", type=str, default=None,
                        help="数据集配置文件（默认 data.yaml；跑通流程可用官方自带 coco8.yaml）")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数，默认 50")
    args = parser.parse_args()
    data_yaml = Path(args.data) if args.data else DATA_YAML

    # 1. 选择设备：有 NVIDIA GPU 用 "0"（第一块显卡），否则退回 CPU
    if torch.cuda.is_available():
        device = "0"
        print(f"检测到 GPU: {torch.cuda.get_device_name(0)}，使用 GPU 训练")
    else:
        device = "cpu"
        print("未检测到 GPU，使用 CPU 训练（会很慢，建议把 epochs 改小）")

    # 2. 训练前校验数据集结构（只有用本项目 data.yaml 时才校验本地目录）
    if args.data is None:
        check_dataset()

    # 3. 加载预训练模型；yolov8n.pt 不存在时 ultralytics 会自动下载到项目目录
    model = YOLO(MODEL_PATH)  # nano 版：体积小、速度快，约 6MB

    # 4. 开始训练
    model.train(
        data=data_yaml,                          # 数据集配置文件
        epochs=args.epochs,                      # 训练轮数（CPU 建议先改成 20）
        imgsz=640,                               # 输入图片大小
        batch=8 if device == "cpu" else 16,      # CPU 用小 batch，GPU 可适当加大
        device=device,                           # "0" = 第一块 GPU，"cpu" = CPU
        patience=20,                             # 早停：连续 20 轮验证集无提升则停止
        project=str(PROJECT_DIR),                # 结果保存根目录
        name=EXPERIMENT_NAME,                    # 实验名 -> runs/detect/hat_detector/
        exist_ok=True,                           # 允许覆盖同名实验目录
        verbose=True,                            # 打印详细训练日志
    )

    best = PROJECT_DIR / EXPERIMENT_NAME / "weights" / "best.pt"
    print(f"\n训练完成！最佳模型保存在：{best}")


if __name__ == "__main__":
    main()
