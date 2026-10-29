#!/usr/bin/env python
# scripts/make_2class.py
# 从多类别 YOLO 数据集派生"只保留指定类别"的子数据集（默认去掉 person，只留 head/helmet）。
#
# 产出（默认）：
#   data/dataset_2cls/images/{train,val,test}/   图片（同盘硬链接，不额外占空间；--copy 改为复制）
#   data/dataset_2cls/labels/{train,val,test}/   只保留目标类别的标注
#   data2cls.yaml                                新数据集配置（nc=2, names=['head','helmet']）
#
# 用法：
#   python scripts/make_2class.py                        # 去掉 person，保留 head/helmet
#   python scripts/make_2class.py --copy                 # 复制图片而不是硬链接
#   python scripts/make_2class.py --keep head            # 只保留 head 一类
#   python scripts/make_2class.py --src data.yaml --out-dir data/dataset_x --out-yaml data_x.yaml
from __future__ import annotations

import argparse
import os
import shutil
from collections import Counter
from pathlib import Path

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
BASE_DIR = Path(__file__).resolve().parent.parent


def resolve(data_yaml: Path) -> tuple[dict, Path, dict[str, Path], list[str]]:
    """读取数据集配置，返回 (原始配置, 数据集根目录, 各 split 图片目录, 类别名)。"""
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
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return cfg, root, splits, list(names or [])


def labels_dir_for(images_dir: Path) -> Path:
    """按 YOLO 约定从 .../images/xxx 推出 .../labels/xxx。"""
    parts = list(images_dir.parts)
    if "images" in parts:
        parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
        return Path(*parts)
    return images_dir.parent / "labels" / images_dir.name


def place_image(src: Path, dst: Path, copy: bool) -> bool:
    """把图片放到目标位置：默认硬链接（同盘不占空间），失败/指定时复制。返回是否新建。"""
    if dst.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not copy:
        try:
            os.link(src, dst)
            return True
        except OSError:
            pass  # 跨盘或文件系统不支持硬链接，退回复制
    shutil.copy2(src, dst)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="派生只保留指定类别的 YOLO 子数据集")
    parser.add_argument("--src", default=str(BASE_DIR / "data.yaml"), help="源数据集配置（默认 data.yaml）")
    parser.add_argument("--keep", default="head,helmet", help="保留的类别名，逗号分隔（默认 head,helmet）")
    parser.add_argument("--out-dir", default=str(BASE_DIR / "data" / "dataset_2cls"), help="输出数据集目录")
    parser.add_argument("--out-yaml", default=str(BASE_DIR / "data2cls.yaml"), help="输出数据集配置文件")
    parser.add_argument("--copy", action="store_true", help="复制图片而不是硬链接（占用额外磁盘空间）")
    args = parser.parse_args()

    src_yaml = Path(args.src)
    if not src_yaml.exists():
        raise SystemExit(f"[错误] 找不到源数据集配置：{src_yaml}")

    cfg, root, splits, names = resolve(src_yaml)
    keep_names = [n.strip() for n in args.keep.split(",") if n.strip()]
    unknown = [n for n in keep_names if n not in names]
    if unknown:
        raise SystemExit(f"[错误] 类别 {unknown} 不在 {names} 里，请检查 --keep")
    keep_ids = [names.index(n) for n in keep_names]
    id_map = {old: new for new, old in enumerate(keep_ids)}  # 旧类别 id -> 新类别 id

    out_dir = Path(args.out_dir)
    out_yaml = Path(args.out_yaml)

    print("=== 派生子数据集 ===")
    print(f"源配置  : {src_yaml.resolve()}")
    print(f"源类别  : {names}")
    print(f"保留类别: {keep_names}  (旧 id {keep_ids} -> 新 id 0..{len(keep_ids)-1})")
    print(f"输出目录: {out_dir}")
    print(f"图片方式: {'复制' if args.copy else '硬链接（同盘不占空间）'}")

    grand_before: Counter = Counter()
    grand_after: Counter = Counter()
    total_images = 0
    total_new_img = 0
    total_empty = 0

    for split, img_dir in splits.items():
        lab_dir = labels_dir_for(img_dir)
        if not img_dir.is_dir() or not lab_dir.is_dir():
            print(f"[跳过] {split}: {img_dir} / {lab_dir} 不存在")
            continue

        out_img_dir = out_dir / "images" / split
        out_lab_dir = out_dir / "labels" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lab_dir.mkdir(parents=True, exist_ok=True)

        before: Counter = Counter()
        after: Counter = Counter()
        n_img = n_new = n_empty = 0
        images = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)
        for img in images:
            key = img.relative_to(img_dir).with_suffix("")
            src_lab = lab_dir / f"{key.as_posix()}.txt"
            dst_lab = out_lab_dir / f"{key.as_posix()}.txt"
            dst_lab.parent.mkdir(parents=True, exist_ok=True)

            kept: list[str] = []
            if src_lab.exists():
                for line in src_lab.read_text(encoding="utf-8", errors="ignore").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    try:
                        cls = int(float(parts[0]))
                    except ValueError:
                        continue
                    before[names[cls] if 0 <= cls < len(names) else f"?{cls}"] += 1
                    if cls in id_map:
                        parts[0] = str(id_map[cls])
                        kept.append(" ".join(parts))
                        after[keep_names[id_map[cls]]] += 1

            dst_lab.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
            if not kept:
                n_empty += 1

            if place_image(img, out_img_dir / f"{key.as_posix()}{img.suffix}", args.copy):
                n_new += 1
            n_img += 1

        grand_before.update(before)
        grand_after.update(after)
        total_images += n_img
        total_new_img += n_new
        total_empty += n_empty

        print(f"\n--- {split}")
        print(f"  图片 {n_img} 张（新建 {n_new}），标注实例 {sum(before.values())} -> {sum(after.values())}"
              f"（删掉 {sum(before.values()) - sum(after.values())} 行）")
        print(f"  各类前 -> 后: " + "，".join(
            f"{n} {before.get(n, 0)}->{after.get(n, 0)}" for n in names))
        print(f"  变成空标注（原本只有 person）: {n_empty} 张")

    # 写出新的 data yaml（路径相对 yaml 所在目录）
    rel_root = os.path.relpath(out_dir, start=out_yaml.resolve().parent).replace(os.sep, "/")
    out_yaml.write_text(
        f"# 由 scripts/make_2class.py 自动生成，勿手改\n"
        f"path: ./{rel_root}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"\n"
        f"nc: {len(keep_names)}\n"
        f"names: {keep_names}\n",
        encoding="utf-8",
    )

    print("\n=== 汇总 ===")
    print(f"图片 {total_images} 张（新建 {total_new_img}），其中 {total_empty} 张变成空标注（纯背景图）")
    print(f"实例 {sum(grand_before.values())} -> {sum(grand_after.values())}，"
          f"删掉 {sum(grand_before.values()) - sum(grand_after.values())} 行")
    print(f"配置已写入：{out_yaml}")
    print(f"\n下一步：")
    print(f"  python train.py --data {out_yaml.name} --name hat_2cls")
    print(f"  python evaluate.py --data {out_yaml.name} --split val")
    print(f"  python evaluate.py --data {out_yaml.name} --split test")


if __name__ == "__main__":
    main()
