# core/tracker.py
# 轻量多目标跟踪器（ByteTrack 的简化版：IoU 贪心匹配 + 轨迹生命周期）。
#
# 为什么不直接用 ultralytics 的 ByteTrack：它依赖 torch，而打包后的 exe 走
# onnxruntime、不带 torch。这里只用 numpy，两种后端都能跑。
#
# 作用：给每个目标一个稳定的 track_id，让"同一个人"只报一次警，
# 而不是每帧/每秒重复报同一个人。
from __future__ import annotations

from dataclasses import dataclass

from .detector import Detection


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    """一条轨迹。"""
    track_id: int
    cls_id: int
    cls_name: str
    box: tuple
    conf: float
    hits: int = 1          # 累计匹配上的帧数
    age: int = 0           # 连续多少帧没匹配上
    missed: bool = False

    @property
    def center(self):
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2


class IouTracker:
    """贪心 IoU 匹配跟踪器。

    - 每帧把检测框和已有轨迹按 IoU 做一对一匹配（同类别才算）；
    - 匹配上的轨迹刷新框、清零 age；没匹配上的 age +1，超过 max_age 回收；
    - 未匹配的检测框新建轨迹，分配递增 id。

    max_age 越大越"粘"（短暂遮挡后还能续上同一个人），但误跟也越多。
    """

    def __init__(self, iou_thr: float = 0.3, max_age: int = 30):
        self.iou_thr = float(iou_thr)
        self.max_age = int(max_age)
        self.tracks: list[Track] = []
        self._next_id = 1

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1

    def update(self, dets: list[Detection]) -> list[Detection]:
        """给每个检测框打上 track_id（写回 Detection.track_id 并返回同一列表）。"""
        for t in self.tracks:
            t.missed = True

        # 置信度高的检测先匹配，避免高置信目标抢不到轨迹
        order = sorted(range(len(dets)), key=lambda i: dets[i].conf, reverse=True)
        used: set[int] = set()

        for i in order:
            d = dets[i]
            best = None
            best_iou = self.iou_thr
            for t in self.tracks:
                if not t.missed or t.cls_id != d.cls_id:
                    continue
                v = _iou(t.box, d.box)
                if v >= best_iou:
                    best, best_iou = t, v
            if best is not None:
                best.box = d.box
                best.conf = d.conf
                best.hits += 1
                best.age = 0
                best.missed = False
                d.track_id = best.track_id
                used.add(i)
            else:
                d.track_id = -1

        # 没匹配上的检测框 → 新建轨迹
        for i in order:
            if i in used:
                continue
            d = dets[i]
            t = Track(track_id=self._next_id, cls_id=d.cls_id,
                      cls_name=d.cls_name, box=d.box, conf=d.conf, missed=False)
            self._next_id += 1
            self.tracks.append(t)
            d.track_id = t.track_id

        # 回收长时间没匹配上的轨迹
        for t in self.tracks:
            if t.missed:
                t.age += 1
        self.tracks = [t for t in self.tracks if t.age <= self.max_age]
        return dets

    def stats(self) -> dict[str, int]:
        """当前这一帧看得见的轨迹数，按类别分开（界面显示"在场人数"）。"""
        out: dict[str, int] = {}
        for t in self.tracks:
            if t.age == 0:
                out[t.cls_name] = out.get(t.cls_name, 0) + 1
        return out
