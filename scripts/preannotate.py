"""把采到的现场帧用当前模型预标注成 YOLO 格式，省掉 80% 标注工作。

用法：
  python scripts/preannotate.py                          # 读 data/scene/raw
  python scripts/preannotate.py --conf 0.15 --test 100 --val 60

产出：
  data/scene/images/{train,val,test}/...     图片（复制）
  data/scene/labels/{train,val,test}/...txt  预标注（需要你人工改错）
  data/scene/data.yaml                       训练/评估配置（可直接给 train.py / evaluate.py）
  data/scene/classes.txt                     给标注工具用
  docs/analysis/08_scene_preannotated.png    抽查拼图

注意：预标注的置信度故意放低（默认 0.15），所以会**多框**；标注时你的活主要是删掉错框、
补上没框到的，而不是从零画。
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
BG, CARD, FG = "#0F1115", "#171B22", "#E6E8EB"
NAMES = ["head", "helmet"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/scene/raw")
    ap.add_argument("--out", default="data/scene")
    ap.add_argument("--model", default="runs/detect/hat_2cls_20261029_085425/weights/best.pt")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--test", type=int, default=100, help="留多少张做现场测试集（不参与训练）")
    ap.add_argument("--val", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="目标目录非空也继续（会覆盖同名文件）")
    args = ap.parse_args()

    src = ROOT / args.src
    out = ROOT / args.out
    files = sorted(p for p in src.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not files:
        raise SystemExit(f"{src} 里没有图片，先跑 scripts/collect_scene.py")
    if out.exists() and any(out.glob("images/*/*")) and not args.force:
        raise SystemExit(f"{out} 里已经有数据了，确认要重做请加 --force")

    rng = random.Random(args.seed)
    order = files[:]
    rng.shuffle(order)
    n_test = min(args.test, max(1, len(order) // 5))
    n_val = min(args.val, max(1, len(order) // 10))
    splits = {}
    for i, f in enumerate(order):
        splits[f] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
    print(f"共 {len(order)} 张：test {n_test} / val {n_val} / train {len(order) - n_test - n_val}")

    from ultralytics import YOLO
    model = YOLO(str(ROOT / args.model))

    for sub in ("images", "labels"):
        for sp in ("train", "val", "test"):
            (out / sub / sp).mkdir(parents=True, exist_ok=True)

    boxes_per_class = Counter()
    empty = 0
    sheet = []
    for i, f in enumerate(files):
        img = cv2.imread(str(f))
        h, w = img.shape[:2]
        r = model.predict(source=img, conf=args.conf, iou=args.iou, imgsz=640, verbose=False)[0]
        lines, drawn = [], []
        for xyxy, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            x1, y1, x2, y2 = xyxy
            cls = int(c)
            lines.append(f"{cls} {((x1 + x2) / 2) / w:.6f} {((y1 + y2) / 2) / h:.6f} "
                         f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            boxes_per_class[NAMES[cls]] += 1
            drawn.append((cls, x1, y1, x2, y2, s))
        if not lines:
            empty += 1
        sp = splits[f]
        shutil.copy2(f, out / "images" / sp / f.name)
        (out / "labels" / sp / (f.stem + ".txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")
        if len(sheet) < 12 and i % max(1, len(files) // 12) == 0:
            sheet.append((img, drawn, f.name))
        if i % 50 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    (out / "classes.txt").write_text("\n".join(NAMES) + "\n", encoding="utf-8")
    (out / "data.yaml").write_text(
        f"# 现场数据集，由 scripts/preannotate.py 生成\n"
        f"path: {out.as_posix()}\n"
        f"train: images/train\nval: images/val\ntest: images/test\n\n"
        f"nc: 2\nnames: {NAMES}\n", encoding="utf-8")

    # ---- 抽查拼图 ----
    if sheet:
        fig, axs = plt.subplots(3, 4, figsize=(12.6, 7.0), facecolor=BG)
        for ax in axs.ravel():
            ax.axis("off")
        for ax, (img, drawn, name) in zip(axs.ravel(), sheet):
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(name[:24], fontsize=7.5, color=FG, pad=2)
            for cls, x1, y1, x2, y2, s in drawn:
                col = "#FF5A3C" if cls == 0 else "#00E676"
                ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=col, lw=1.5))
                ax.text(x1, y1 - 3, f"{NAMES[cls]} {s:.2f}", fontsize=6, color=col)
        fig.suptitle("预标注抽查：绿=helmet 红=head。多出来的框删掉、漏掉的补上", color=FG, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        (ROOT / "docs/analysis").mkdir(parents=True, exist_ok=True)
        fig.savefig(ROOT / "docs/analysis/08_scene_preannotated.png", dpi=140, facecolor=BG)
        plt.close(fig)

    stats = {"images": len(files), "splits": {"test": n_test, "val": n_val,
                                             "train": len(files) - n_test - n_val},
             "boxes": dict(boxes_per_class), "images_without_box": empty,
             "conf": args.conf}
    (out / "preannotate_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    print("\n完成：")
    print(f"  图片 {len(files)} 张，框 {dict(boxes_per_class)}")
    print(f"  一个框都没有的图 {empty} 张（空镜正常；如果里面明明有人，说明现场确实漏检）")
    print(f"  配置 {out / 'data.yaml'}（已可直接用于 train.py / evaluate.py）")
    if empty > len(files) * 0.6:
        print("  警告：超过 60% 的图一个框都没有 —— 先确认镜头有没有对准、是不是太暗")
    print("\n下一步：改标注（删除误框、补漏框），工具见 docs/labeling_guide.md")


if __name__ == "__main__":
    main()
