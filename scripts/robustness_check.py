"""鲁棒性体检：把测试集人为做坏（暗光/逆光/模糊/低分辨率/压缩/噪声），看精度掉多少。

用来回答一个问题：「模型在自己数据集上只有 5% 的框错，为什么现场用起来错很多？」
输出 docs/analysis/06_robustness.png、07_imgsz.png 与 runs/robustness/report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
BG, CARD, FG, DIM = "#0F1115", "#171B22", "#E6E8EB", "#8A93A0"


def load_gt(p: Path, w: int, h: int):
    out = []
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.split()
        if len(s) < 5:
            continue
        c = int(s[0])
        cx, cy, bw, bh = map(float, s[1:5])
        out.append((c, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def iou(a, b):
    ix1, iy1 = max(a[1], b[1]), max(a[2], b[2])
    ix2, iy2 = min(a[3], b[3]), min(a[4], b[4])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[3] - a[1]) * (a[4] - a[2]) + (b[3] - b[1]) * (b[4] - b[2]) - inter
    return inter / ua if ua > 0 else 0.0


def ap50(records):
    """records: [(preds:[(c,x1,y1,x2,y2,s)], gts:[(c,x1,y1,x2,y2)])]，按类别算 101 点 AP50。"""
    out = {}
    for c in (0, 1):
        npos = sum(1 for _, g in records for gg in g if gg[0] == c)
        entries = []
        for preds, gts in records:
            gc = [g for g in gts if g[0] == c]
            used = [False] * len(gc)
            for p in sorted([p for p in preds if p[0] == c], key=lambda x: -x[5]):
                best, bj = 0.0, -1
                for j, g in enumerate(gc):
                    if used[j]:
                        continue
                    v = iou(p, g)
                    if v > best:
                        best, bj = v, j
                if best >= 0.5 and bj >= 0:
                    used[bj] = True
                    entries.append((p[5], 1))
                else:
                    entries.append((p[5], 0))
        if npos == 0:
            out[c] = float("nan")
            continue
        entries.sort(key=lambda x: -x[0])
        tp = fp = 0
        rec, prec = [], []
        for _, is_tp in entries:
            tp += is_tp
            fp += 1 - is_tp
            rec.append(tp / npos)
            prec.append(tp / max(1, tp + fp))
        mrec = np.concatenate(([0.0], np.array(rec), [1.0]))
        mpre = np.concatenate(([1.0], np.array(prec), [0.0]))
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        grid = np.linspace(0, 1, 101)
        ap = float(np.mean(np.interp(grid, mrec, mpre)))
        out[c] = ap
    return out


def degrade(img, kind):
    if kind == "baseline":
        return img
    if kind == "dark":
        return np.clip(img.astype(np.float32) * 0.32, 0, 255).astype(np.uint8)
    if kind == "backlit":
        return np.clip(img.astype(np.float32) * 1.9 + 18, 0, 255).astype(np.uint8)
    if kind == "motion":
        k = np.zeros((9, 9), np.float32)
        k[4, :] = 1 / 9
        return cv2.filter2D(img, -1, k)
    if kind == "blur":
        return cv2.GaussianBlur(img, (9, 9), 0)
    if kind == "lowres":
        h, w = img.shape[:2]
        s = cv2.resize(img, (max(16, w // 3), max(16, h // 3)), interpolation=cv2.INTER_AREA)
        return cv2.resize(s, (w, h), interpolation=cv2.INTER_LINEAR)
    if kind == "jpeg":
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 22])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if kind == "noise":
        n = np.random.default_rng(0).normal(0, 16, img.shape)
        return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)
    raise ValueError(kind)


CN = {"baseline": "原图（基线）", "dark": "暗光（亮度×0.32）", "backlit": "逆光/过曝",
      "motion": "运动模糊", "blur": "失焦模糊", "lowres": "低分辨率（÷3）",
      "jpeg": "高压缩（JPEG q22）", "noise": "强噪点"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/detect/hat_2cls_20261029_085425/weights/best.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--out", default="docs/analysis")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(str(ROOT / args.model))
    img_dir = ROOT / "data/dataset_2cls/images" / args.split
    lbl_dir = ROOT / "data/dataset_2cls/labels" / args.split
    files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"})[: args.limit]
    print(f"{len(files)} 张图", flush=True)

    kinds = ["baseline", "dark", "backlit", "motion", "blur", "lowres", "jpeg", "noise"]
    res = {}
    for kind in kinds:
        records = []
        for i, f in enumerate(files):
            img = cv2.imread(str(f))
            h, w = img.shape[:2]
            gt = load_gt(lbl_dir / (f.stem + ".txt"), w, h)
            aug = degrade(img, kind)
            r = model.predict(source=aug, conf=0.01, iou=0.45, imgsz=640, verbose=False, device=None)[0]
            preds = [(int(c), *xyxy, s) for xyxy, c, s in zip(
                r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
            records.append((preds, gt))
            if i % 100 == 0:
                print(f"  {kind} {i}/{len(files)}", flush=True)
        aps = ap50(records)
        # 置信度 0.25 时的召回（更接近现场阈值）
        rec25, fp25 = {}, {}
        for c in (0, 1):
            npos = sum(1 for _, g in records for gg in g if gg[0] == c)
            hit = 0
            for preds, gts in records:
                gc = [g for g in gts if g[0] == c]
                used = [False] * len(gc)
                for p in sorted([p for p in preds if p[0] == c and p[5] >= 0.25], key=lambda x: -x[5]):
                    best, bj = 0.0, -1
                    for j, g in enumerate(gc):
                        if used[j]:
                            continue
                        v = iou(p, g)
                        if v > best:
                            best, bj = v, j
                    if best >= 0.5 and bj >= 0:
                        used[bj] = True
                        hit += 1
            rec25[c] = hit / max(1, npos)
        res[kind] = {"ap50_head": aps[0], "ap50_helmet": aps[1],
                     "recall25_head": rec25[0], "recall25_helmet": rec25[1]}
        print(f"{kind}: AP50 head={aps[0]:.3f} helmet={aps[1]:.3f} R@.25={rec25[0]:.3f}/{rec25[1]:.3f}", flush=True)

    # ---- 图 06：鲁棒性 ----
    fig, ax = plt.subplots(figsize=(9.0, 3.9), facecolor=BG)
    ax.set_facecolor(CARD)
    x = np.arange(len(kinds))
    ax.plot(x, [res[k]["ap50_head"] for k in kinds], "-o", color="#FF5A3C", lw=2, ms=5, label="head AP50")
    ax.plot(x, [res[k]["ap50_helmet"] for k in kinds], "-o", color="#00E676", lw=2, ms=5, label="helmet AP50")
    ax.plot(x, [res[k]["recall25_head"] for k in kinds], "--o", color="#FF8A65", lw=1.4, ms=4,
            label="head 召回@conf0.25")
    ax.plot(x, [res[k]["recall25_helmet"] for k in kinds], "--o", color="#69F0AE", lw=1.4, ms=4,
            label="helmet 召回@conf0.25")
    for i, k in enumerate(kinds):
        ax.annotate(f"{res[k]['ap50_helmet']:.2f}", (i, res[k]["ap50_helmet"]), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7, color="#00E676")
    base = res["baseline"]["ap50_helmet"]
    ax.axhline(base, color="#252B36", ls=":", lw=1)
    ax.set_xticks(x, [CN[k] for k in kinds], rotation=18, ha="right", fontsize=8, color=FG)
    ax.tick_params(colors=FG)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("精度 / 召回", color=FG, fontsize=9)
    ax.set_title(f"⑥ 现场干扰下精度怎么塌（{len(files)} 张测试图，逐项施加）", color=FG, fontsize=11)
    ax.grid(color="#252B36")
    ax.set_axisbelow(True)
    ax.legend(facecolor=CARD, edgecolor="#252B36", labelcolor=FG, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(ROOT / args.out / "06_robustness.png", dpi=150, facecolor=BG)
    plt.close(fig)

    # ---- 图 07：输入分辨率 ----
    sizes = [320, 480, 640, 960]
    sz_res = {}
    for sz in sizes:
        records = []
        for f in files:
            img = cv2.imread(str(f))
            h, w = img.shape[:2]
            gt = load_gt(lbl_dir / (f.stem + ".txt"), w, h)
            r = model.predict(source=img, conf=0.01, iou=0.45, imgsz=sz, verbose=False)[0]
            preds = [(int(c), *xyxy, s) for xyxy, c, s in zip(
                r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
            records.append((preds, gt))
        aps = ap50(records)
        sz_res[sz] = aps
        print(f"imgsz {sz}: {aps}", flush=True)

    fig, ax = plt.subplots(figsize=(5.6, 3.6), facecolor=BG)
    ax.set_facecolor(CARD)
    ax.plot(sizes, [sz_res[s][0] for s in sizes], "-o", color="#FF5A3C", lw=2, label="head AP50")
    ax.plot(sizes, [sz_res[s][1] for s in sizes], "-o", color="#00E676", lw=2, label="helmet AP50")
    ax.set_xticks(sizes, [f"{s}px" for s in sizes], color=FG)
    ax.tick_params(colors=FG)
    ax.set_xlabel("推理输入尺寸（原图先缩到这个尺寸）", color=FG, fontsize=9)
    ax.set_ylabel("AP50", color=FG, fontsize=9)
    ax.set_title("⑦ 输入分辨率对精度的影响", color=FG, fontsize=11)
    ax.grid(color="#252B36")
    ax.set_axisbelow(True)
    ax.legend(facecolor=CARD, edgecolor="#252B36", labelcolor=FG, fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / args.out / "07_imgsz.png", dpi=150, facecolor=BG)
    plt.close(fig)

    out = ROOT / "runs/robustness/report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n_images": len(files), "conditions": res, "imgsz": sz_res},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print("done →", out)


if __name__ == "__main__":
    main()
