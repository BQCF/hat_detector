"""把 COCO 预训练的 yolov8n 导出成 onnx，供 core/gate.py 的「人形门控」使用。

用法：
  python scripts/export_person.py                    # 默认导出到 models/person_yolov8n.onnx
  python scripts/export_person.py --imgsz 416        # 门控用小分辨率时一起改

说明：门控只需要「人在哪」，不需要 80 类里别的类别；onnx 里照样带全部 80 类，
cpu 上 416 分辨率约 8~15ms。
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--static", action="store_true",
                    help="导出固定尺寸（默认导出动态尺寸，门控才能自由改 imgsz）")
    ap.add_argument("--out", default="models/person_yolov8n.onnx")
    args = ap.parse_args()

    from ultralytics import YOLO

    src = ROOT / args.weights
    if not src.exists():
        raise SystemExit(f"找不到 {src}，先放一个 COCO 预训练 yolov8n.pt 到项目根目录")
    print(f"导出 {src.name} → onnx (imgsz={args.imgsz}) …")
    model = YOLO(str(src))
    path = model.export(format="onnx", imgsz=args.imgsz, simplify=False,
                        dynamic=not args.static)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), out)
    mb = out.stat().st_size / 1024 / 1024
    print(f"完成：{out}（{mb:.1f} MB）")

    # 顺手验证一下它能认出人
    import cv2
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta().custom_metadata_map or {}
    print("类别数 meta:", (meta.get("names") or "")[:60], "…")
    print("输入:", sess.get_inputs()[0].shape)
    _ = np.zeros((1, 3, args.imgsz, args.imgsz), np.float32)
    sess.run(None, {sess.get_inputs()[0].name: _})
    print("onnx 推理自检通过")


if __name__ == "__main__":
    main()
