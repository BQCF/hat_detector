#!/usr/bin/env python
# scripts/export.py
# 把训练好的 best.pt 导出成 ONNX，供 onnxruntime 推理（打包成 exe 后不用带 torch）。
#
# 用法：
#   python scripts/export.py                          # 自动找最新的 best.pt
#   python scripts/export.py --model runs/detect/xxx/weights/best.pt
#   python scripts/export.py --imgsz 320              # 导出给小分辨率用
#   python scripts/export.py --half                   # FP16（体积减半，仅部分运行时支持）
import argparse
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def find_best(model_arg: str) -> Path:
    if model_arg:
        p = Path(model_arg)
        if not p.exists():
            raise SystemExit(f"[错误] 模型不存在：{p}")
        return p
    cands = sorted((BASE_DIR / "runs" / "detect").glob("*/weights/best.pt"),
                   key=lambda x: x.stat().st_mtime, reverse=True)
    if not cands:
        raise SystemExit(f"[错误] 在 {BASE_DIR / 'runs' / 'detect'} 下找不到 best.pt，请先训练。")
    return cands[0]


def smoke_test(onnx_path: Path, imgsz: int) -> None:
    """用 onnxruntime 加载一次，确认模型能跑、输出形状符合预期。"""
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    dummy = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
    out = sess.run(None, {name: dummy})[0]
    print(f"  onnxruntime 冒烟测试通过：输入 {name} {dummy.shape} -> 输出 {out.shape}")
    print(f"  运行时可用 provider：{ort.get_available_providers()}")


def main() -> None:
    p = argparse.ArgumentParser(description="导出 ONNX 模型")
    p.add_argument("--model", default="", help="best.pt 路径（默认自动找最新的）")
    p.add_argument("--imgsz", type=int, default=640, help="导出用的输入尺寸（导出后可动态改变）")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset 版本")
    p.add_argument("--no-simplify", action="store_true", help="不简化计算图")
    p.add_argument("--half", action="store_true", help="导出 FP16")
    p.add_argument("--out", default="", help="输出路径（默认与权重同目录 best.onnx）")
    args = p.parse_args()

    from ultralytics import YOLO

    pt = find_best(args.model)
    print("=== 导出 ONNX ===")
    print(f"权重  : {pt}")

    model = YOLO(str(pt))
    print(f"类别  : {model.names}")

    t0 = time.time()
    out_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=not args.no_simplify,
        half=args.half,
        dynamic=True,          # 允许换 batch / 换分辨率
        device="cpu",          # 导出本身用 CPU 就够，避免占用显卡
    )
    out_path = Path(out_path)
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        out_path.replace(target)
        out_path = target

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"耗时  : {time.time() - t0:.1f}s")
    print(f"输出  : {out_path}  ({size_mb:.2f} MB)")
    smoke_test(out_path, args.imgsz)
    print(f"\n下一步：")
    print(f"  python cli.py --model {out_path.relative_to(BASE_DIR)} --type image --source xxx.jpg")
    print(f"  python desktop.py --model {out_path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
