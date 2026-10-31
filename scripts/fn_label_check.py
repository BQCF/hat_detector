"""把一部分「漏检」的真值框单独裁出来放大，用来肉眼判断是模型没看见，还是标注本身画歪了。

用法：python scripts/fn_label_check.py
输出：docs/analysis/07_fn_label_check.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
BG, CARD, FG = "#0F1115", "#171B22", "#E6E8EB"

data = json.loads((ROOT / "runs/error_analysis_data2cls_test/error_stats.json").read_text(encoding="utf-8"))
img_dir = ROOT / "data/dataset_2cls/images/test"
items = []
for p in data["per_image"]:
    for b in p["fn_boxes"]:
        x1, y1, x2, y2 = b[1:5]
        if (x2 - x1) * (y2 - y1) >= 900:  # 只挑够大的，看得清
            items.append((p["file"], b[0], x1, y1, x2, y2))
items.sort(key=lambda t: -(t[4] - t[2]) * (t[5] - t[3]))
picks = items[:24]
print(f"漏检里够大的 {len(items)} 个，取前 {len(picks)} 个")

cols, rows = 8, 3
fig, axs = plt.subplots(rows, cols, figsize=(13.0, 4.6), facecolor=BG)
for ax in axs.ravel():
    ax.axis("off")
for ax, (fname, cls, x1, y1, x2, y2) in zip(axs.ravel(), picks):
    img = Image.open(img_dir / fname).convert("RGB")
    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = bw * 0.35, bh * 0.35
    cx1 = int(max(0, x1 - pad_x))
    cy1 = int(max(0, y1 - pad_y))
    cx2 = int(min(img.width, x2 + pad_x))
    cy2 = int(min(img.height, y2 + pad_y))
    crop = img.crop((cx1, cy1, cx2, cy2))
    s = 150 / max(crop.width, crop.height)
    crop = crop.resize((max(1, int(crop.width * s)), max(1, int(crop.height * s))))
    ax.imshow(crop)
    ax.set_facecolor(CARD)
    ax.add_patch(plt.Rectangle(((x1 - cx1) * s, (y1 - cy1) * s), bw * s, bh * s,
                               fill=False, ec="#FFD54F", lw=1.4))
    ax.set_title("没戴" if cls == 0 else "戴着", fontsize=7,
                 color="#FF8A65" if cls == 0 else "#69F0AE", pad=2)
fig.suptitle("⑦ 漏检样本单独放大：黄框=数据集真值。哪些其实是标注画歪了？", color=FG, fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = ROOT / "docs/analysis/07_fn_label_check.png"
fig.savefig(out, dpi=150, facecolor=BG)
print("→", out)
