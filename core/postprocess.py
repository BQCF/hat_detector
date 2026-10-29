# core/postprocess.py
# 后处理：统计各类数量、可选的跨类别去重。
from __future__ import annotations

from .detector import Detection


def count_classes(dets: list[Detection], names: dict[int, str] | None = None) -> dict[str, int]:
    """统计各类别的检测数量，输出 {类别名: 数量}。"""
    counts: dict[str, int] = {}
    for d in dets:
        counts[d.cls_name] = counts.get(d.cls_name, 0) + 1
    if names:
        for n in names.values():
            counts.setdefault(n, 0)
    return counts


def _iou(a: Detection, b: Detection) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def cross_class_dedup(dets: list[Detection], iou_thr: float = 0.5) -> list[Detection]:
    """跨类别去重：同一个位置既判成 head 又判成 helmet 时，只保留置信度更高的那个。

    本项目里 head(未戴帽) 与 helmet(安全帽) 不应同时出现在同一个人头上，
    高度重叠的两个框通常是误检，去掉可避免重复计数与误报警。
    """
    kept = sorted(dets, key=lambda d: d.conf, reverse=True)
    result: list[Detection] = []
    for d in kept:
        if any(_iou(d, k) >= iou_thr for k in result):
            continue
        result.append(d)
    return result
