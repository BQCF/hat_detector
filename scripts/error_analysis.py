"""测试集错误分析：把「错在哪」拆成可量化的几类，并出图。

输出：
  runs/error_analysis_<split>/error_stats.json   原始统计
  docs/analysis/*.png                            图表
用法：
  python scripts/error_analysis.py --split test
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

NAME = {0: "head（没戴）", 1: "helmet（戴着）"}
SHORT = {0: "head", 1: "helmet"}
BG, CARD, FG, DIM = "#0F1115", "#171B22", "#E6E8EB", "#8A93A0"


def load_gt(label_path: Path, w: int, h: int):
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        c = int(parts[0])
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        boxes.append((c, (cx - bw / 2) * w, (cy - bh / 2) * h,
                      (cx + bw / 2) * w, (cy + bh / 2) * h))
    return boxes


def iou(a, b) -> float:
    ix1, iy1 = max(a[1], b[1]), max(a[2], b[2])
    ix2, iy2 = min(a[3], b[3]), min(a[4], b[4])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[3] - a[1]) * (a[4] - a[2]) + (b[3] - b[1]) * (b[4] - b[2]) - inter
    return inter / ua if ua > 0 else 0.0


def greedy_match(gt, preds, same_class: bool, thr: float, used_g: set, used_p: set):
    cands = []
    for gi, g in enumerate(gt):
        if gi in used_g:
            continue
        for pi, p in enumerate(preds):
            if pi in used_p:
                continue
            if ((g[0] == p[0]) if same_class else (g[0] != p[0])) is False:
                continue
            v = iou(g, p)
            if v >= thr:
                cands.append((v, gi, pi))
    cands.sort(reverse=True)
    out = []
    for v, gi, pi in cands:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        out.append((gi, pi, v))
    return out


def analyse(gt, preds, thr=0.5):
    ug, up = set(), set()
    tp = greedy_match(gt, preds, True, thr, ug, up)
    conf = greedy_match(gt, preds, False, thr, ug, up)
    fn = [g for g in range(len(gt)) if g not in ug]
    fp = [p for p in range(len(preds)) if p not in up]
    return tp, conf, fn, fp


def max_iou(box, others):
    return max([iou(box, o) for o in others], default=0.0)


def resolve_dirs(data_yaml: Path, split: str):
    """按 ultralytics 的 data.yaml 规则解析出图片目录与标签目录。"""
    import yaml

    data_yaml = data_yaml if data_yaml.is_absolute() else (ROOT / data_yaml)
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    base = Path(cfg.get("path", data_yaml.parent))
    if not base.is_absolute():
        base = (ROOT / base).resolve()
    rel = Path(cfg[split])
    img_dir = base / rel
    lbl_dir = base / Path(*("labels" if p == "images" else p for p in rel.parts))
    return img_dir, lbl_dir


def crop_around(img: Image.Image, box, pad=0.7, target=340):
    x1, y1, x2, y2 = box[1:5]
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    cx1 = int(max(0, x1 - pad * bw))
    cy1 = int(max(0, y1 - pad * bh))
    cx2 = int(min(img.width, x2 + pad * bw))
    cy2 = int(min(img.height, y2 + pad * bh))
    if cx2 - cx1 < 24 or cy2 - cy1 < 24:
        return None
    c = img.crop((cx1, cy1, cx2, cy2))
    s = target / max(c.width, c.height)
    return c.resize((max(1, int(c.width * s)), max(1, int(c.height * s)))), cx1, cy1


def draw(ax, crop, ox, oy, gt, preds, title):
    ax.imshow(crop)
    ax.set_title(title, fontsize=8.5, color=FG, pad=3)
    ax.axis("off")
    for g in gt:
        ax.add_patch(plt.Rectangle((g[1] - ox, g[2] - oy), g[3] - g[1], g[4] - g[2],
                                   fill=False, ec="#00E676", lw=1.8))
    for p in preds:
        ax.add_patch(plt.Rectangle((p[1] - ox, p[2] - oy), p[3] - p[1], p[4] - p[2],
                                   fill=False, ec="#FF5A3C", lw=1.5, ls="--"))
        ax.text(p[1] - ox, p[2] - oy - 4, f"{SHORT[p[0]]} {p[5]:.2f}", fontsize=6.5,
                color="#FF8A65")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/detect/hat_2cls_20261029_085425/weights/best.pt")
    ap.add_argument("--data", default="data2cls.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="")
    ap.add_argument("--out", default="docs/analysis")
    args = ap.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_dir = ROOT / "runs" / f"error_analysis_{Path(args.data).stem}_{args.split}"
    stats_dir.mkdir(parents=True, exist_ok=True)

    img_dir, lbl_dir = resolve_dirs(ROOT / args.data, args.split)
    print(f"数据集：{img_dir}")
    files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.limit:
        files = files[: args.limit]
    print(f"[1/3] {len(files)} 张图，开始推理 …", flush=True)

    from ultralytics import YOLO
    model = YOLO(str(ROOT / args.model))

    per_image, agg = [], Counter()
    buckets = [(0.0, 0.001), (0.001, 0.005), (0.005, 0.02), (0.02, 0.1), (0.1, 1.01)]
    bn = ["<0.1%", "0.1-0.5%", "0.5-2%", "2-10%", ">10%"]
    fn_by_size = {c: np.zeros(len(buckets)) for c in (0, 1)}
    gt_by_size = {c: np.zeros(len(buckets)) for c in (0, 1)}
    fn_by_crowd = {c: Counter() for c in (0, 1)}
    gt_by_crowd = {c: Counter() for c in (0, 1)}
    tp_conf, fp_conf = {0: [], 1: []}, {0: [], 1: []}
    fp_iou_bucket = Counter()   # 误报框与同类真值框的 IoU 分布（区分「偏框」和「凭空」）
    fn_iou_bucket = Counter()   # 漏检框与最近预测框的 IoU 分布（区分「没看见」和「被压掉」）

    def bucket_of(area, table):
        for bi, (lo, hi) in enumerate(buckets):
            if lo <= area < hi:
                table[bi] += 1
                return

    for idx, f in enumerate(files):
        img = Image.open(f).convert("RGB")
        w, h = img.size
        gt = load_gt(lbl_dir / (f.stem + ".txt"), w, h)
        r = model.predict(source=str(f), conf=args.conf, iou=args.iou,
                          imgsz=args.imgsz, verbose=False, device=args.device or None)[0]
        preds = [(int(c), *xyxy, s) for xyxy, c, s in zip(
            r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
        tp, conf, fn, fp = analyse(gt, preds)
        if idx % 100 == 0:
            print(f"    {idx}/{len(files)}", flush=True)

        crowd = "1-2" if len(gt) <= 2 else ("3-5" if len(gt) <= 5 else "6+")
        for gi, pi, _ in tp:
            c = gt[gi][0]
            agg["tp"] += 1
            agg[f"tp_{c}"] += 1
            tp_conf[c].append(preds[pi][5])
        for gi, pi, _ in conf:
            agg["confusion"] += 1
            agg[f"conf_{gt[gi][0]}_{preds[pi][0]}"] += 1
            fp_conf[preds[pi][0]].append(preds[pi][5])
        for gi in fn:
            c = gt[gi][0]
            agg["fn"] += 1
            agg[f"fn_{c}"] += 1
            bucket_of((gt[gi][3] - gt[gi][1]) * (gt[gi][4] - gt[gi][2]) / (w * h), fn_by_size[c])
            fn_by_crowd[c][crowd] += 1
            same = [p for p in preds if p[0] == c]
            v = max_iou(gt[gi], same)
            fn_iou_bucket["≥0.5（被别的框顶掉）" if v >= 0.5 else
                          ("0.3-0.5（框偏了）" if v >= 0.3 else
                           ("0.1-0.3（位置大致对）" if v >= 0.1 else "<0.1（完全没看见）"))] += 1
        for g in gt:
            c = g[0]
            bucket_of((g[3] - g[1]) * (g[4] - g[2]) / (w * h), gt_by_size[c])
            gt_by_crowd[c][crowd] += 1
        for pi in fp:
            agg["fp"] += 1
            agg[f"fp_{preds[pi][0]}"] += 1
            fp_conf[preds[pi][0]].append(preds[pi][5])
            same = [g for g in gt if g[0] == preds[pi][0]]
            v = max_iou(preds[pi], same)
            fp_iou_bucket["≥0.5" if v >= 0.5 else ("0.3-0.5（框偏了）" if v >= 0.3 else
                                                   ("0.1-0.3（框大了）" if v >= 0.1 else
                                                    "<0.1（凭空多出来）"))] += 1

        per_image.append({
            "file": f.name, "w": w, "h": h, "gt": len(gt), "pred": len(preds),
            "tp": len(tp), "fn": len(fn), "fp": len(fp), "confusion": len(conf),
            "fn_boxes": [[gt[gi][0], *[round(v, 1) for v in gt[gi][1:5]]] for gi in fn],
            "fp_boxes": [[preds[pi][0], *[round(v, 1) for v in preds[pi][1:5]], round(preds[pi][5], 3)]
                         for pi in fp],
            "conf_boxes": [[gt[gi][0], preds[pi][0], round(v, 2),
                            *[round(v2, 1) for v2 in preds[pi][1:5]]] for gi, pi, v in conf],
        })

    print(f"[2/3] TP={agg['tp']} FN={agg['fn']} FP={agg['fp']} 跨类错判={agg['confusion']}", flush=True)

    # ---------- ① 混淆矩阵 ----------
    cm = np.zeros((3, 3))
    for c in (0, 1):
        cm[c, c] = agg[f"tp_{c}"]
        cm[c, 1 - c] = agg[f"conf_{c}_{1 - c}"]
        cm[c, 2] = agg[f"fn_{c}"]
        cm[2, c] = agg[f"fp_{c}"]
    fig, ax = plt.subplots(figsize=(5.6, 4.6), facecolor=BG)
    ax.set_facecolor(CARD)
    row_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(row_norm, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(3), ["预测 head", "预测 helmet", "漏检"], color=FG)
    ax.set_yticks(range(3), ["真值 head", "真值 helmet", "无目标(全算误报)"], color=FG)
    for i in range(3):
        for j in range(3):
            if i == 2 and j == 2:
                continue
            ax.text(j, i, f"{int(cm[i, j])}\n{row_norm[i, j]:.0%}", ha="center", va="center",
                    color="#0F1115" if row_norm[i, j] > 0.55 else FG, fontsize=9)
    ax.set_title(f"① 混淆矩阵（{len(files)} 张测试图，conf≥{args.conf}）", color=FG, fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "01_confusion.png", dpi=150, facecolor=BG)
    plt.close(fig)

    # ---------- ② 漏检率 vs 目标大小 ----------
    fig, ax = plt.subplots(figsize=(7.0, 3.9), facecolor=BG)
    ax.set_facecolor(CARD)
    x = np.arange(len(bn))
    for k, c in enumerate((0, 1)):
        rate = np.where(gt_by_size[c] > 0, fn_by_size[c] / np.maximum(gt_by_size[c], 1), np.nan)
        bars = ax.bar(x + (k - 0.5) * 0.36, rate * 100, 0.36, label=NAME[c],
                      color="#FF5A3C" if c == 0 else "#00E676", alpha=0.85)
        for b, n in zip(bars, gt_by_size[c]):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2, f"{int(n)}",
                    ha="center", fontsize=7, color=DIM)
    ax.set_xticks(x, bn, color=FG)
    ax.tick_params(colors=FG)
    ax.set_ylabel("漏检率 %（柱顶数字=该类该尺寸真值个数）", color=FG, fontsize=9)
    ax.set_title("② 漏检率随目标大小变化（框面积占整图比例）", color=FG, fontsize=11)
    ax.grid(axis="y", color="#252B36")
    ax.set_axisbelow(True)
    ax.legend(facecolor=CARD, edgecolor="#252B36", labelcolor=FG, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "02_fn_by_size.png", dpi=150, facecolor=BG)
    plt.close(fig)

    # ---------- ③ 漏检率 vs 画面人数 ----------
    fig, ax = plt.subplots(figsize=(6.2, 3.6), facecolor=BG)
    ax.set_facecolor(CARD)
    crowds = ["1-2", "3-5", "6+"]
    x = np.arange(len(crowds))
    for k, c in enumerate((0, 1)):
        rate = [fn_by_crowd[c][q] / gt_by_crowd[c][q] * 100 if gt_by_crowd[c][q] else np.nan
                for q in crowds]
        bars = ax.bar(x + (k - 0.5) * 0.34, rate, 0.34, label=NAME[c],
                      color="#FF5A3C" if c == 0 else "#00E676", alpha=0.85)
        for b, q in zip(bars, crowds):
            if gt_by_crowd[c][q]:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8,
                        f"{gt_by_crowd[c][q]}", ha="center", fontsize=7, color=DIM)
    ax.set_xticks(x, ["画面 1-2 人", "3-5 人", "6 人以上"], color=FG)
    ax.tick_params(colors=FG)
    ax.set_ylabel("漏检率 %", color=FG, fontsize=9)
    ax.set_title("③ 漏检率随画面人数变化（柱顶数字=真值个数）", color=FG, fontsize=11)
    ax.grid(axis="y", color="#252B36")
    ax.set_axisbelow(True)
    ax.legend(facecolor=CARD, edgecolor="#252B36", labelcolor=FG, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "03_fn_by_crowd.png", dpi=150, facecolor=BG)
    plt.close(fig)

    # ---------- ④ 置信度分布 ----------
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.2), facecolor=BG)
    for ax, c in zip(axs, (0, 1)):
        ax.set_facecolor(CARD)
        bins = np.linspace(args.conf, 1.0, 24)
        if tp_conf[c]:
            ax.hist(tp_conf[c], bins=bins, color="#00E676", alpha=0.75, label="正确的")
        if fp_conf[c]:
            ax.hist(fp_conf[c], bins=bins, color="#FF5A3C", alpha=0.75, label="误报的")
        ax.set_yscale("log")
        ax.tick_params(colors=FG, labelsize=8)
        ax.set_title(NAME[c], color=FG, fontsize=9)
        ax.set_xlabel("置信度", color=FG, fontsize=8)
        ax.grid(color="#252B36", axis="y")
        ax.set_axisbelow(True)
        ax.legend(facecolor=CARD, edgecolor="#252B36", labelcolor=FG, fontsize=7)
    fig.suptitle("④ 置信度分布：误报集中在低分段吗", color=FG, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "04_conf_hist.png", dpi=150, facecolor=BG)
    plt.close(fig)

    # ---------- ⑤ 误差归因 ----------
    order_fp = ["<0.1（凭空多出来）", "0.1-0.3（框大了）", "0.3-0.5（框偏了）", "≥0.5"]
    order_fn = ["<0.1（完全没看见）", "0.1-0.3（位置大致对）", "0.3-0.5（框偏了）", "≥0.5（被别的框顶掉）"]
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 3.4), facecolor=BG)
    for ax, keys, title, data in (
        (axs[0], order_fp, f"误报 {agg['fp'] + agg['confusion']} 个：凭空 vs 框偏", fp_iou_bucket),
        (axs[1], order_fn, f"漏检 {agg['fn'] + agg['confusion']} 个：没看见 vs 框偏", fn_iou_bucket),
    ):
        ax.set_facecolor(CARD)
        vals = [data.get(k, 0) for k in keys]
        bars = ax.barh(range(len(keys)), vals, color=["#EF5350", "#FF8A65", "#FFB74D", "#66BB6A"])
        ax.set_yticks(range(len(keys)), [k.replace("（", " ").replace("）", "") for k in keys],
                      fontsize=7.5, color=FG)
        ax.invert_yaxis()
        total = max(1, sum(vals))
        for b, v in zip(bars, vals):
            ax.text(v + total * 0.01, b.get_y() + b.get_height() / 2,
                    f"{v}（{v / total:.0%}）", va="center", fontsize=8, color=FG)
        ax.set_xlim(0, max(vals) * 1.25 if max(vals) else 1)
        ax.set_title(title, color=FG, fontsize=10)
        ax.set_xlabel("个数", color=FG, fontsize=8)
        ax.tick_params(colors=FG, labelsize=7)
        ax.grid(axis="x", color="#252B36")
        ax.set_axisbelow(True)
    fig.suptitle("⑤ 误差归因：与「最近的同类真值/预测」框的重叠程度", color=FG, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "05_error_cause.png", dpi=150, facecolor=BG)
    plt.close(fig)

    # ---------- ⑥ 典型错例（放大到出错的那个框） ----------
    def area_ratio(p, b):
        return (b[3] - b[1]) * (b[4] - b[2]) / (p["w"] * p["h"])

    def candidates(key, min_area=0.003, max_gt=6):
        out = []
        for p in per_image:
            if p["gt"] == 0:
                continue
            if key == "confusion":
                for b in p["conf_boxes"]:
                    out.append((p, (0, *b[3:7])))
            else:
                if p["gt"] > max_gt:
                    continue
                for b in p[key]:
                    if area_ratio(p, b) >= min_area:
                        out.append((p, (0, *b[1:5])))
        out.sort(key=lambda t: -area_ratio(t[0], (0, *t[1])))
        return out

    rows = [("跨类错判：把头盔认成头 / 把头认成头盔", candidates("confusion"), "conf"),
            ("误报：画面上没有的东西被框出来", candidates("fp_boxes"), "fp"),
            ("漏检：有真值但没有框", candidates("fn_boxes"), "fn")]
    ncol = 4
    fig, axs = plt.subplots(len(rows), ncol, figsize=(12.4, 8.0), facecolor=BG)
    for ax in axs.ravel():
        ax.axis("off")
    for ri, (title, cands, kind) in enumerate(rows):
        axs[ri, 0].set_ylabel(title, color=FG, fontsize=9)
        used = set()
        k = 0
        for p, box in cands:
            if k >= ncol:
                break
            if p["file"] in used:
                continue
            f = img_dir / p["file"]
            img = Image.open(f).convert("RGB")
            got = crop_around(img, box)
            if not got:
                continue
            crop, ox, oy = got
            r = model.predict(source=str(f), conf=args.conf, iou=args.iou,
                              imgsz=args.imgsz, verbose=False)[0]
            gt = load_gt(lbl_dir / (f.stem + ".txt"), img.width, img.height)
            preds = [(int(c), *xyxy, s) for xyxy, c, s in zip(
                r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
            gt_near = [g for g in gt if
                       (g[1] > ox - 40 and g[3] < ox + crop.width + 40 and
                        g[2] > oy - 40 and g[4] < oy + crop.height + 40)]
            pr_near = [q for q in preds if q[1] > ox - 40 and q[3] < ox + crop.width + 40 and
                       q[2] > oy - 40 and q[4] < oy + crop.height + 40]
            draw(axs[ri, k], crop, ox, oy, gt_near, pr_near, p["file"][:9] + "…")
            used.add(p["file"])
            k += 1
    handles = [plt.Line2D([], [], color="#00E676", lw=1.8, label="真值框"),
               plt.Line2D([], [], color="#FF5A3C", lw=1.5, ls="--", label="模型预测（后面是置信度）")]
    fig.legend(handles=handles, loc="lower center", ncol=2, facecolor=CARD,
               edgecolor="#252B36", labelcolor=FG, fontsize=9)
    fig.suptitle("⑥ 典型错例（已放大到出错区域）", color=FG, fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(out_dir / "06_examples.png", dpi=140, facecolor=BG)
    plt.close(fig)

    # ---------- 落盘 ----------
    summary = {
        "split": args.split, "images": len(files), "conf": args.conf, "imgsz": args.imgsz,
        "gt": {c: int(gt_by_size[c].sum()) for c in (0, 1)},
        "tp": agg["tp"], "fn": agg["fn"], "fp": agg["fp"], "confusion": agg["confusion"],
        "conf_matrix": {"head->head": int(agg["tp_0"]), "helmet->helmet": int(agg["tp_1"]),
                        "head->helmet": int(agg["conf_0_1"]), "helmet->head": int(agg["conf_1_0"])},
        "per_class": {},
        "fp_cause": dict(fp_iou_bucket), "fn_cause": dict(fn_iou_bucket),
        "fn_by_size": {SHORT[c]: {"buckets": bn, "gt": gt_by_size[c].astype(int).tolist(),
                                  "fn": fn_by_size[c].astype(int).tolist()} for c in (0, 1)},
        "fn_by_crowd": {SHORT[c]: {q: {"gt": gt_by_crowd[c][q], "fn": fn_by_crowd[c][q]}
                                   for q in crowds} for c in (0, 1)},
        "images_with_error": sum(1 for p in per_image if p["fn"] + p["fp"] + p["confusion"] > 0),
        "per_image": per_image,
    }
    for c in (0, 1):
        tp_c = agg[f"tp_{c}"]
        fp_c = agg[f"fp_{c}"] + agg[f"conf_{1 - c}_{c}"]
        fn_c = agg[f"fn_{c}"] + agg[f"conf_{c}_{1 - c}"]
        summary["per_class"][SHORT[c]] = {
            "tp": tp_c, "fp": fp_c, "fn": fn_c,
            "precision": round(tp_c / max(1, tp_c + fp_c), 4),
            "recall": round(tp_c / max(1, tp_c + fn_c), 4),
        }
    (stats_dir / "error_stats.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[3/3] 完成 → {stats_dir / 'error_stats.json'}，图在 {out_dir}")
    for c in (0, 1):
        s = summary["per_class"][SHORT[c]]
        print(f"  {SHORT[c]}: P={s['precision']:.3f} R={s['recall']:.3f} "
              f"TP={s['tp']} FP={s['fp']} FN={s['fn']}")


if __name__ == "__main__":
    main()
