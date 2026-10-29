#!/usr/bin/env python
# evaluate.py
# 评估脚本：在 val / test 集上评测训练好的最佳模型，打印分类指标并可落盘 JSON。
#
# 用法：
#   python evaluate.py                                # 用 data.yaml 评 val，默认取最新 best.pt
#   python evaluate.py --split test                   # 评测试集（883 张）
#   python evaluate.py --data data2cls.yaml --split val
#   python evaluate.py --model runs/detect/xxx/weights/best.pt   # 指定模型
#   python evaluate.py --split test --json runs/eval_test.json   # 落盘 JSON
import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = BASE_DIR / "data.yaml"
PROJECT_DIR = BASE_DIR / "runs" / "detect"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def count_images(data_yaml: Path, split: str) -> int:
    """数一下某个 split 目录里的图片张数（ultralytics 的 metrics 没有暴露图片总数）。"""
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    root = Path(cfg.get("path") or ".")
    if not root.is_absolute():
        root = (data_yaml.resolve().parent / root).resolve()
    d = root / (cfg.get(split) or "")
    return sum(1 for p in d.rglob("*") if p.suffix.lower() in IMG_EXTS) if d.is_dir() else 0


def find_best_model(model_arg: str) -> Path:
    """返回 best.pt 路径：显式指定优先，否则取 runs/detect 下最新的那份。"""
    if model_arg:
        p = Path(model_arg)
        if not p.exists():
            raise SystemExit(f"[错误] 指定的模型不存在：{p}")
        return p
    cands = sorted(PROJECT_DIR.glob("*/weights/best.pt"),
                   key=lambda x: x.stat().st_mtime, reverse=True)
    if not cands:
        print(f"[错误] 在 {PROJECT_DIR} 下找不到任何 weights/best.pt")
        print("请先运行 python train.py 完成训练。")
        lasts = sorted(PROJECT_DIR.glob("*/weights/last.pt"),
                       key=lambda x: x.stat().st_mtime, reverse=True)
        if lasts:
            print(f"提示：检测到训练中断留下的 last.pt（{lasts[0]}），")
            print("      如需继续，可在 train.py 中给 model.train(...) 加 resume=True 参数。")
        raise SystemExit(1)
    return cands[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="在 val/test 集上评估训练好的模型")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="数据集配置文件（默认 data.yaml）")
    parser.add_argument("--model", default="", help="模型权重路径；留空=自动取 runs/detect 下最新的 best.pt")
    parser.add_argument("--split", default="val", choices=["val", "test"], help="评估集：val 或 test")
    parser.add_argument("--imgsz", type=int, default=640, help="推理图片大小")
    parser.add_argument("--conf", type=float, default=0.001,
                        help="评估用的置信度下限（默认 0.001，算 mAP 时要足够低）")
    parser.add_argument("--json", default="", help="把结果写成 JSON（留空=不写）")
    args = parser.parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise SystemExit(f"[错误] 找不到数据集配置：{data_yaml}")
    model_path = find_best_model(args.model)
    device = "0" if torch.cuda.is_available() else "cpu"

    model = YOLO(model_path)
    metrics = model.val(data=str(data_yaml), split=args.split,
                        imgsz=args.imgsz, conf=args.conf,
                        device=device, verbose=True)

    box = metrics.box
    names = metrics.names                     # {class_id: name}
    rows = []
    for i, cls_id in enumerate(box.ap_class_index):
        rows.append({
            "class_id": int(cls_id),
            "class": names.get(int(cls_id), str(int(cls_id))),
            "precision": float(box.p[i]),
            "recall": float(box.r[i]),
            "mAP50": float(box.ap50[i]),
            "mAP50-95": float(box.ap[i]),
        })

    summary = {
        "model": str(model_path),
        "split": args.split,
        "data": str(data_yaml.resolve()),
        "images": count_images(data_yaml, args.split),
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "per_class": rows,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    print("\n" + "=" * 62)
    print(f"评估结果 · {args.split} 集   模型: {model_path}")
    print(f"数据集: {data_yaml.resolve()}    图片 {summary['images']} 张")
    print("=" * 62)
    print(f"{'类别':<12}{'Precision':>11}{'Recall':>9}{'mAP50':>9}{'mAP50-95':>10}")
    print("-" * 62)
    for r in rows:
        print(f"{r['class']:<12}{r['precision']:>11.4f}{r['recall']:>9.4f}"
              f"{r['mAP50']:>9.4f}{r['mAP50-95']:>10.4f}")
    print("-" * 62)
    print(f"{'总体':<12}{summary['precision']:>11.4f}{summary['recall']:>9.4f}"
          f"{summary['mAP50']:>9.4f}{summary['mAP50-95']:>10.4f}")
    print("=" * 62)

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结果已写入：{out}")


if __name__ == "__main__":
    main()
