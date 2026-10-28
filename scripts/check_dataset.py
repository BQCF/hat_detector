#!/usr/bin/env python
# scripts/check_dataset.py
# 数据集体检：标注配对、类别分布、坐标合法性、图片完整性、重复图片。
# 用法：
#   python scripts/check_dataset.py                 # 体检 data.yaml 里的 train/val/test
#   python scripts/check_dataset.py --data data.yaml
#   python scripts/check_dataset.py --json runs/dataset_check.json
#   python scripts/check_dataset.py --full          # 全量解码图片（慢，默认只读文件头）
# 退出码：0 = 没发现错误；1 = 存在错误级问题（缺标注、坐标越界、类别越界等）
"""在训练前把数据集的问题一次性查出来，避免训练到一半才报错。"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

OK, WARN, ERR = "OK", "WARN", "ERR"

problems: list[tuple[str, str, str]] = []  # (level, split, message)


def add(level: str, split: str, msg: str) -> None:
    problems.append((level, split, msg))


def resolve_config(data_yaml: Path) -> tuple[dict, Path, dict[str, Path], list[str]]:
    """读取 data.yaml，返回 (原始配置, 数据集根目录, 各 split 的图片目录, 类别名)。"""
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    root = Path(cfg.get("path") or ".")
    if not root.is_absolute():
        root = (data_yaml.resolve().parent / root).resolve()

    splits: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        rel = cfg.get(split)
        if rel:
            splits[split] = (root / rel).resolve()

    names = cfg.get("names")
    if isinstance(names, dict):  # {0: 'head', ...}
        names = [names[k] for k in sorted(names)]
    names = list(names or [])
    if cfg.get("nc") not in (None, len(names)):
        add(ERR, "-", f"data.yaml 的 nc={cfg.get('nc')} 与 names 数量 {len(names)} 不一致")
    return cfg, root, splits, names


def labels_dir_for(images_dir: Path) -> Path:
    """按 YOLO 约定从 .../images/xxx 推出 .../labels/xxx。"""
    parts = list(images_dir.parts)
    if "images" in parts:
        parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
        return Path(*parts)
    return images_dir.parent / "labels" / images_dir.name


def scan_split(split: str, images_dir: Path, names: list[str]) -> dict:
    """检查一个 split 的配对、标注格式与坐标，返回统计信息。"""
    result = {
        "split": split,
        "images_dir": str(images_dir),
        "images": 0,
        "labels": 0,
        "missing_label": [],
        "orphan_label": [],
        "empty_label": 0,
        "instances": Counter(),
        "bad_lines": [],
        "oob_boxes": 0,
        "degenerate_boxes": 0,
        "bad_class": 0,
        "out_of_frame": 0,
    }
    if not images_dir.is_dir():
        add(ERR, split, f"图片目录不存在：{images_dir}")
        return result

    labels_dir = labels_dir_for(images_dir)
    if not labels_dir.is_dir():
        add(ERR, split, f"标注目录不存在：{labels_dir}")
        return result

    images = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)
    labels = sorted(labels_dir.rglob("*.txt"))
    result["images"] = len(images)
    result["labels"] = len(labels)
    if not images:
        add(ERR, split, "图片目录里没有找到任何图片")
        return result

    label_set = {p.relative_to(labels_dir).with_suffix("").as_posix() for p in labels}
    image_set: set[str] = set()

    nc = len(names)
    for img in images:
        key = img.relative_to(images_dir).with_suffix("").as_posix()
        image_set.add(key)
        lab = labels_dir / f"{key}.txt"
        if not lab.exists():
            result["missing_label"].append(key)
            continue
        lines = [ln.strip() for ln in lab.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
        if not lines:
            result["empty_label"] += 1
            continue
        for i, line in enumerate(lines, 1):
            parts = line.split()
            if len(parts) < 5:
                result["bad_lines"].append(f"{key}:{i} 字段数 {len(parts)} < 5")
                continue
            try:
                cls = int(float(parts[0]))
                cx, cy, w, h = (float(v) for v in parts[1:5])
            except ValueError:
                result["bad_lines"].append(f"{key}:{i} 无法解析：{line[:40]}")
                continue
            if not 0 <= cls < nc:
                result["bad_class"] += 1
                continue
            result["instances"][names[cls]] += 1
            if w <= 0 or h <= 0:
                result["degenerate_boxes"] += 1
            if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)):
                result["oob_boxes"] += 1
            # 框略微出界属常见标注误差，只统计不报错
            if (cx - w / 2 < -0.01 or cy - h / 2 < -0.01
                    or cx + w / 2 > 1.01 or cy + h / 2 > 1.01):
                result["out_of_frame"] += 1

    for key in sorted(label_set - image_set):
        result["orphan_label"].append(key)

    if result["missing_label"]:
        add(ERR, split, f"{len(result['missing_label'])} 张图片缺少同名 .txt 标注（前 5 个：{', '.join(result['missing_label'][:5])}）")
    if result["orphan_label"]:
        add(ERR, split, f"{len(result['orphan_label'])} 个标注找不到对应图片（前 5 个：{', '.join(result['orphan_label'][:5])}）")
    if result["bad_lines"]:
        add(ERR, split, f"{len(result['bad_lines'])} 行标注格式非法（前 3 条：{'; '.join(result['bad_lines'][:3])}）")
    if result["bad_class"]:
        add(ERR, split, f"{result['bad_class']} 个框的类别 id 超出 0~{nc - 1}")
    if result["oob_boxes"]:
        add(ERR, split, f"{result['oob_boxes']} 个框坐标不在 [0,1] 内")
    if result["degenerate_boxes"]:
        add(ERR, split, f"{result['degenerate_boxes']} 个框宽或高 <= 0")
    if result["out_of_frame"]:
        add(WARN, split, f"{result['out_of_frame']} 个框超出画面边缘（超过 1%，通常是标注误差）")
    if result["empty_label"]:
        add(WARN, split, f"{result['empty_label']} 个空标注文件（纯背景图，可保留作为负样本）")
    return result


def check_images(paths: list[Path], full: bool) -> None:
    """检查图片能否打开、尺寸是否过小；full=True 时完整解码验证是否损坏。"""
    try:
        from PIL import Image
    except ImportError:
        add(WARN, "-", "未安装 Pillow，跳过图片完整性检查")
        return

    broken, tiny, sizes = [], [], Counter()
    for p in paths:  # 默认只读文件头，几千张图也就几秒
        try:
            with Image.open(p) as im:
                w, h = im.size
                if full:
                    im.load()
        except Exception as exc:  # noqa: BLE001 - 任何解码异常都算坏图
            broken.append(f"{p.name}({exc.__class__.__name__})")
            continue
        if min(w, h) < 32:
            tiny.append(f"{p.name}({w}x{h})")
        sizes[f"{w}x{h}"] += 1
    if broken:
        add(ERR, "-", f"{len(broken)} 张图片无法打开/已损坏（前 5 个：{', '.join(broken[:5])}）")
    if tiny:
        add(WARN, "-", f"{len(tiny)} 张图片最短边 < 32 像素，检测基本无效（前 5 个：{', '.join(tiny[:5])}）")
    top = sizes.most_common(5)
    print(f"  图片尺寸 Top5: " + "，".join(f"{k}×{v}张" for k, v in top))


def check_duplicates(paths: list[Path]) -> None:
    """按文件内容 md5 找重复图片（重复会放大过拟合，也让 val 指标虚高）。"""
    by_hash: dict[str, list[str]] = defaultdict(list)
    for p in paths:
        h = hashlib.md5(p.read_bytes()).hexdigest()
        by_hash[h].append(p.name)
    dup_groups = {h: v for h, v in by_hash.items() if len(v) > 1}
    if dup_groups:
        n_extra = sum(len(v) - 1 for v in dup_groups.values())
        sample = list(dup_groups.values())[0]
        add(WARN, "-", f"{len(dup_groups)} 组重复图片，多出 {n_extra} 张（例：{', '.join(sample[:3])}）")
    else:
        print("  未发现重复图片")


def print_report(results: list[dict], names: list[str]) -> None:
    print("\n=== 类别分布（实例数） ===")
    header = f"{'类别':<10}" + "".join(f"{r['split']:>10}" for r in results)
    print(header)
    print("-" * len(header))
    for name in names:
        row = f"{name:<10}" + "".join(f"{r['instances'].get(name, 0):>10}" for r in results)
        print(row)
    print("-" * len(header))
    print(f"{'合计':<10}" + "".join(f"{sum(r['instances'].values()):>10}" for r in results))
    print()
    print("=== 各 split 概况 ===")
    for r in results:
        print(f"  {r['split']:<6} 图片 {r['images']:>5} 张 | 标注 {r['labels']:>5} 个 | "
              f"空标注 {r['empty_label']:>4} | 实例 {sum(r['instances'].values()):>6}")
    # 类别极度不平衡提示：某个类占比 < 5% 时模型基本学不会
    total = Counter()
    for r in results:
        total.update(r["instances"])
    grand = sum(total.values()) or 1
    for name in names:
        ratio = total.get(name, 0) / grand
        if ratio < 0.05:
            add(WARN, "-", f"类别 {name} 只占 {ratio:.1%} 的实例（{total.get(name, 0)} 个），大概率学不出来，考虑补数据或删除该类")


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLO 数据集体检")
    parser.add_argument("--data", default="data.yaml", help="数据集配置（默认 data.yaml）")
    parser.add_argument("--json", default=None, help="把体检结果写成 JSON 文件")
    parser.add_argument("--full", action="store_true", help="完整解码每张图片（默认只读文件头，速度快）")
    args = parser.parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        print(f"[错误] 找不到数据集配置：{data_yaml}")
        return 1
    cfg, root, splits, names = resolve_config(data_yaml)

    print("=== 数据集体检 ===")
    print(f"配置：{data_yaml.resolve()}")
    print(f"根目录：{root}")
    print(f"类别：{names}（nc={len(names)}）")

    results = []
    all_images: list[Path] = []
    for split, img_dir in splits.items():
        print(f"\n--- {split}: {img_dir}")
        r = scan_split(split, img_dir, names)
        results.append(r)
        all_images.extend(p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)

    print("\n=== 图片完整性 ===")
    check_images(all_images, args.full)

    print("\n=== 重复图片 ===")
    check_duplicates(all_images)

    print_report(results, names)

    # 训练/验证是否同一批图片（会让 val 指标虚高）
    def _names(d: Path) -> set[str]:
        return {p.name for p in d.rglob("*") if p.suffix.lower() in IMG_EXTS} if d and d.is_dir() else set()

    train_keys = _names(splits.get("train"))
    val_keys = _names(splits.get("val"))
    if train_keys & val_keys:
        add(ERR, "-", f"训练集与验证集有 {len(train_keys & val_keys)} 张同名图片，val 指标会虚高")

    errors = [p for p in problems if p[0] == ERR]
    warns = [p for p in problems if p[0] == WARN]
    print("=== 问题清单 ===")
    if not problems:
        print("  [OK] 未发现问题，可以开始训练")
    for level, split, msg in problems:
        icon = "[X]" if level == ERR else "[!]"
        print(f"  {icon} [{split}] {msg}")
    print(f"\n结论：错误 {len(errors)} 项，警告 {len(warns)} 项"
          + ("（有错误，建议先修复再训练）" if errors else "（可以训练）"))

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "data_yaml": str(data_yaml.resolve()),
            "root": str(root),
            "names": names,
            "splits": [{k: (dict(v) if isinstance(v, Counter) else v) for k, v in r.items()} for r in results],
            "problems": [{"level": l, "split": s, "message": m} for l, s, m in problems],
            "errors": len(errors),
            "warnings": len(warns),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"体检结果已写入：{out}")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
