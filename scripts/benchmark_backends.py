#!/usr/bin/env python
# scripts/benchmark_backends.py
# 对比 torch(ultralytics) 与 onnx(onnxruntime) 两个后端：速度、检测一致性。
# 换模型 / 换机器后跑一遍，确认换后端不影响结果。
#
# 用法：
#   python scripts/benchmark_backends.py                     # 自动找最新 best.pt / best.onnx
#   python scripts/benchmark_backends.py --images 50 --imgsz 640
#   python scripts/benchmark_backends.py --imgsz 320
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))      # 让 scripts/ 下的脚本能 import 到 core/


def iou(a, b) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    bb = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    return inter / (aa + bb - inter + 1e-9) if (aa + bb - inter) > 0 else 0.0


def match_rate(ref, other, thr: float = 0.5) -> tuple[int, int, int]:
    """把 other 与 ref 对齐：返回 (配对成功数, ref 数, other 数)。"""
    used = set()
    hit = 0
    for r in ref:
        for j, o in enumerate(other):
            if j in used or o.cls_name != r.cls_name:
                continue
            if iou(r, o) >= thr:
                used.add(j)
                hit += 1
                break
    return hit, len(ref), len(other)


def bench(det, images, imgsz, label):
    for p in images[:5]:        # 预热
        det.infer(cv2.imread(str(p)), imgsz=imgsz)
    times = []
    all_dets = []
    for p in images:
        img = cv2.imread(str(p))
        t0 = time.perf_counter()
        dets = det.infer(img, imgsz=imgsz)
        times.append((time.perf_counter() - t0) * 1000)
        all_dets.append(dets)
    t = np.array(times)
    print(f"{label:<28} {t.mean():6.1f} ms 均值 | {np.percentile(t,50):6.1f} 中位 | "
          f"{np.percentile(t,95):6.1f} P95 | {1000 / t.mean():6.1f} FPS")
    return all_dets


def main() -> None:
    from core.detector import Detector, find_latest_model

    p = argparse.ArgumentParser(description="对比 torch / onnx 后端")
    p.add_argument("--images", type=int, default=40, help="参与对比的图片数")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--pt", default="", help="torch 权重路径")
    p.add_argument("--onnx", default="", help="onnx 模型路径")
    p.add_argument("--only", default="both", choices=["both", "torch", "onnx"],
                   help="只跑一个后端（推荐）：torch 与 onnxruntime 的线程池同进程会互相抢 CPU，"
                        "要拿干净的速度请分别开进程跑 --only torch / --only onnx")
    args = p.parse_args()

    root = BASE_DIR / "data" / "dataset_2cls" / "images" / "val"
    if not root.is_dir():
        root = BASE_DIR / "data" / "dataset" / "images" / "val"
    images = sorted(root.glob("*.jpg"))[: args.images]
    if not images:
        raise SystemExit(f"[错误] {root} 下没有图片")
    print(f"样本: {len(images)} 张来自 {root}")

    pt_path = Path(args.pt) if args.pt else None
    onnx_path = Path(args.onnx) if args.onnx else None
    if pt_path is None or onnx_path is None:
        latest = find_latest_model()
        if pt_path is None:
            pt_path = latest if latest and latest.suffix == ".pt" else None
        if onnx_path is None:
            onnx_path = latest if latest and latest.suffix == ".onnx" else None
        # 顺带找同目录的另一个
        if pt_path and not onnx_path and pt_path.with_suffix(".onnx").exists():
            onnx_path = pt_path.with_suffix(".onnx")
        if onnx_path and not pt_path and onnx_path.with_suffix(".pt").exists():
            pt_path = onnx_path.with_suffix(".pt")

    print(f"\n=== 速度对比（imgsz={args.imgsz}）===")
    results = {}
    if pt_path and args.only in ("both", "torch"):
        dt = Detector(model_path=str(pt_path), backend="torch", conf=args.conf)
        print(f"torch 权重: {pt_path.name}  设备={dt.device}")
        results["torch"] = bench(dt, images, args.imgsz, "torch (ultralytics)")
    if onnx_path and args.only in ("both", "onnx"):
        do = Detector(model_path=str(onnx_path), backend="onnx", conf=args.conf)
        print(f"onnx  模型: {onnx_path.name}  设备={do.device}")
        results["onnx"] = bench(do, images, args.imgsz, "onnx (onnxruntime)")
    if not results:
        raise SystemExit("[错误] 一个后端都没跑起来，请检查模型路径")

    if "torch" in results and "onnx" in results:
        print("\n=== 检测一致性（IoU>=0.5 且同类别算匹配）===")
        hit = ref_n = oth_n = 0
        per_cls = {}
        for a, b in zip(results["torch"], results["onnx"]):
            h, rn, on = match_rate(a, b)
            hit += h
            ref_n += rn
            oth_n += on
            for d in a:
                per_cls.setdefault(d.cls_name, [0, 0])
                per_cls[d.cls_name][0] += 1
            for d in b:
                per_cls.setdefault(d.cls_name, [0, 0])
                per_cls[d.cls_name][1] += 1
        prec = hit / oth_n * 100 if oth_n else 0
        rec = hit / ref_n * 100 if ref_n else 0
        print(f"  torch 检出 {ref_n} 个，onnx 检出 {oth_n} 个，配对成功 {hit} 个")
        print(f"  一致率：onnx→torch {prec:.2f}%   torch→onnx {rec:.2f}%")
        print(f"  分类计数对比: " + "，".join(
            f"{k} torch={v[0]}/onnx={v[1]}" for k, v in sorted(per_cls.items())))
        if prec >= 95 and rec >= 95:
            print("  ✅ 两个后端结果基本一致，可以放心用 onnx 打包")
        else:
            print("  ⚠️ 差异偏大，建议检查阈值/NMS 设置后再打包")
    else:
        print("\n（只跑了一个后端，跳过一致性对比）")


if __name__ == "__main__":
    main()
