# core/roi.py
# 危险区域（ROI）：在画面上画一个多边形，只有落在区域内的目标才计入统计/触发报警。
#
# 关键点：多边形用**归一化坐标**（0~1）保存，这样换分辨率、换摄像头、
# 换了窗口大小都不用重新画。
from __future__ import annotations

import cv2
import numpy as np

from .detector import Detection

DANGER_BGR = (60, 90, 255)          # 橙红（与界面危险色一致）
OUTSIDE_BGR = (120, 120, 120)       # 区域外的框画成灰色


class RoiRegion:
    """危险区域多边形。points 是归一化坐标列表 [(x, y), ...]。"""

    def __init__(self, points=None, enabled: bool = False, only_inside: bool = True):
        self.points: list[tuple[float, float]] = [
            (float(p[0]), float(p[1])) for p in (points or [])
            if isinstance(p, (list, tuple)) and len(p) >= 2
        ]
        self.enabled = bool(enabled)
        self.only_inside = bool(only_inside)

    # ---------------- 状态 ----------------
    @property
    def active(self) -> bool:
        """有至少 3 个点且启用，区域才生效。"""
        return self.enabled and len(self.points) >= 3

    def clear(self) -> None:
        self.points.clear()

    def to_config(self) -> list[list[float]]:
        return [[round(x, 5), round(y, 5)] for x, y in self.points]

    @classmethod
    def from_config(cls, cfg) -> "RoiRegion":
        """从 AppConfig.roi 构造。"""
        return cls(getattr(cfg, "points", None),
                   enabled=getattr(cfg, "enabled", False),
                   only_inside=getattr(cfg, "only_inside", True))

    # ---------------- 几何 ----------------
    def contains_norm(self, nx: float, ny: float) -> bool:
        """归一化坐标是否落在多边形内（射线法）。"""
        if not self.active:
            return True
        poly = self.points
        inside = False
        n = len(poly)
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            # 水平射线与边 (j, i) 相交，且交点在点左侧
            if (yi > ny) != (yj > ny):
                x_cross = (xj - xi) * (ny - yi) / (yj - yi + 1e-12) + xi
                if nx < x_cross:
                    inside = not inside
            j = i
        return inside

    def contains_det(self, det: Detection, w: int, h: int) -> bool:
        """检测框中心是否在区域内。头/安全帽框用中心点比"脚点"更稳。"""
        if not self.active:
            return True
        cx = (det.x1 + det.x2) / 2 / max(1, w)
        cy = (det.y1 + det.y2) / 2 / max(1, h)
        return self.contains_norm(cx, cy)

    def partition(self, dets: list[Detection], w: int, h: int):
        """拆成 (区域内, 区域外)。区域没启用时全算区域内。"""
        if not self.active:
            return list(dets), []
        inside, outside = [], []
        for d in dets:
            (inside if self.contains_det(d, w, h) else outside).append(d)
        return inside, outside

    # ---------------- 绘制 ----------------
    def draw(self, frame: np.ndarray, label: str = "DANGER ZONE") -> np.ndarray:
        """在帧上画出半透明多边形（原地修改并返回）。"""
        if not self.active:
            return frame
        h, w = frame.shape[:2]
        pts = np.array([[int(x * w), int(y * h)] for x, y in self.points], dtype=np.int32)

        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], DANGER_BGR)
        cv2.addWeighted(overlay, 0.16, frame, 0.84, 0, frame)   # 半透明填充
        cv2.polylines(frame, [pts], True, DANGER_BGR, 2, cv2.LINE_AA)

        # 顶点小圆点，方便看清区域边界
        for p in pts:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, DANGER_BGR, -1, cv2.LINE_AA)

        if label:
            x, y = int(pts[:, 0].min()), int(pts[:, 1].min())
            cv2.putText(frame, label, (x + 4, max(16, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, DANGER_BGR, 2, cv2.LINE_AA)
        return frame
