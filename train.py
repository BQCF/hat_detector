#!/usr/bin/env python
# train.py
# 训练脚本：在指定数据集上微调 YOLOv8。
# 实验目录自动带时间戳（runs/detect/<名称>_<YYYYmmdd_HHMMSS>/），不会覆盖历史结果。
#
# 用法：
#   python train.py                                  # 用 data.yaml 训练，实验名 hat_<时间戳>
#   python train.py --data data2cls.yaml --name hat_2cls        # 二分类数据集
#   python train.py --epochs 100 --cache ram          # 多轮 + 图片缓存提速
#   python train.py --data coco8.yaml                 # 官方小数据集先跑通流程
#   python train.py --model yolov11n.pt --cos-lr      # 换模型 / 换学习率策略
import argparse
from datetime import datetime
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR / "runs" / "detect"      # 训练结果保存根目录
DEFAULT_DATA = BASE_DIR / "data.yaml"
DEFAULT_MODEL = BASE_DIR / "yolov8n.pt"          # 预训练权重（缺失时 ultralytics 自动下载）


def labels_dir_for(images_dir: Path) -> Path:
    """按 YOLO 约定从 .../images/xxx 推出 .../labels/xxx。"""
    parts = list(images_dir.parts)
    if "images" in parts:
        parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
        return Path(*parts)
    return images_dir.parent / "labels" / images_dir.name


def check_dataset(data_yaml: Path) -> None:
    """训练前校验数据集目录结构，缺失时给出可操作的提示，避免训练到一半才报错。"""
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    root = Path(cfg.get("path") or ".")
    if not root.is_absolute():
        root = (data_yaml.resolve().parent / root).resolve()
    missing = []
    for split in ("train", "val"):
        rel = cfg.get(split)
        if not rel:
            continue
        img_dir = root / rel
        lab_dir = labels_dir_for(img_dir)
        if not img_dir.is_dir():
            missing.append(f"  - {img_dir}（{split} 图片）")
        if not lab_dir.is_dir():
            missing.append(f"  - {lab_dir}（{split} 标注）")
    if missing:
        print(f"[错误] 数据集配置 {data_yaml} 指向的目录 {root} 缺少以下子目录：")
        print("\n".join(missing))
        print("\n请把数据集整理成 YOLO 标准结构后重试：")
        print("  <root>/images/train/  <root>/images/val/   图片")
        print("  <root>/labels/train/  <root>/labels/val/   与图片同名的 .txt 标注")
        print("（Roboflow 导出的是 valid 目录时，改名为 val，或同步修改 data.yaml 的 val 路径）")
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="在 YOLO 数据集上微调 YOLOv8（实验目录带时间戳，不覆盖历史结果）")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="数据集配置文件（默认 data.yaml）")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="预训练权重或模型 yaml（默认 yolov8n.pt）")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数，默认 50")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图片大小，默认 640")
    parser.add_argument("--batch", type=int, default=0, help="batch 大小，0=自动（GPU 16 / CPU 8）")
    parser.add_argument("--device", default="", help="留空=自动；0=GPU，cpu=CPU")
    parser.add_argument("--cache", default="none", choices=["none", "ram", "disk"],
                        help="图片缓存，ram/disk 能明显加快训练（默认 none）")
    parser.add_argument("--cos-lr", action="store_true", help="用余弦退火学习率")
    parser.add_argument("--close-mosaic", type=int, default=10,
                        help="最后 N 轮关掉 mosaic 增强（0=不关），默认 10")
    parser.add_argument("--patience", type=int, default=20, help="早停：连续 N 轮验证集无提升就停，默认 20")
    parser.add_argument("--seed", type=int, default=0, help="随机种子，便于复现")
    parser.add_argument("--workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--name", default="", help="实验名前缀（默认 hat），最终目录为 <前缀>_<时间戳>")
    parser.add_argument("--no-check", action="store_true", help="跳过训练前的数据集目录检查")
    args = parser.parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise SystemExit(f"[错误] 找不到数据集配置：{data_yaml}")

    # 1. 选择设备
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "0"
        print(f"检测到 GPU: {torch.cuda.get_device_name(0)}，使用 GPU 训练")
    else:
        device = "cpu"
        print("未检测到 GPU，使用 CPU 训练（会很慢，建议把 epochs 改小）")

    batch = args.batch if args.batch > 0 else (8 if device == "cpu" else 16)

    # 2. 训练前校验数据集结构
    if not args.no_check:
        check_dataset(data_yaml)

    # 3. 实验名：前缀 + 时间戳，保证每次训练都在新目录里，不会覆盖旧结果
    prefix = args.name or "hat"
    run_name = f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}"

    # 4. 加载预训练模型并开始训练
    model = YOLO(args.model)  # nano 版：体积小、速度快，约 6MB
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        device=device,
        cache=None if args.cache == "none" else args.cache,
        patience=args.patience,
        seed=args.seed,
        workers=args.workers,
        cos_lr=args.cos_lr,
        close_mosaic=args.close_mosaic if args.close_mosaic > 0 else 0,
        project=str(PROJECT_DIR),
        name=run_name,
        exist_ok=False,        # ← 不再覆盖：目录带时间戳，历史结果全部保留
        verbose=True,
    )

    best = PROJECT_DIR / run_name / "weights" / "best.pt"
    print(f"\n训练完成！实验目录：{PROJECT_DIR / run_name}")
    print(f"最佳模型：{best}")
    print(f"指标评估：python evaluate.py --data {data_yaml.name} --split val")
    print(f"测试集  ：python evaluate.py --data {data_yaml.name} --split test")


if __name__ == "__main__":
    main()
