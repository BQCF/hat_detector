# core/gate.py
# 人形门控：先用一个通用的 COCO person 检测器找出"画面里有哪些人"，
# 再把 head/helmet 框里"不在任何人体框内"的那些丢掉。
#
# 为什么需要：二分类模型只学过"像不像头/帽"，遇到床头、衣架、墙面反光也会硬框一个。
# 加一道"必须落在人身上"的闸门，可以砍掉这类凭空误报。
# 用的是独立的 person 模型（默认 yolov8n COCO 导出的 onnx），与主模型互不影响。
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .detector import Detection


@dataclass
class GateConfig:
    """人形门控设置。"""
    enabled: bool = True
    model_path: str = "models/person_yolov8n.onnx"   # COCO 预训练 yolov8n 导出的 onnx
    imgsz: int = 416                   # 人物检测分辨率（人比头好认，可以小一点换速度）
    conf: float = 0.35                 # 人物置信度阈值
    min_cover: float = 0.3             # 检测框有多大比例落在人体框内才算"在人身上"
    expand: float = 0.25               # 人体框向外扩的比例（照顾头部/边缘；实测 0.25 比 0.06 少丢一半真目标）
    empty_policy: str = "keep"         # 一个人都没检出来时：keep=全留（更保险）/ drop=全丢（更干净）
    threads: int = 2                   # 门控会话的线程数（主 detector 之外再开一个会话，限线程免得抢 CPU）


def cover_ratio(inner, outer) -> float:
    """inner 框有多少面积落在 outer 框里（0~1）。"""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area = max(1e-9, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return float(inter / area)


def expand_box(box, ratio: float):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    dx, dy = w * ratio, h * ratio
    return (x1 - dx, y1 - dy, x2 + dx, y2 + dy)


def keep_covered(dets: list[Detection], persons: list[tuple], min_cover: float = 0.5,
                 empty_policy: str = "drop") -> list[Detection]:
    """纯逻辑：保留"落在人体框内"的检测。persons 是 [(x1,y1,x2,y2), ...]。"""
    if not persons:
        return [] if empty_policy == "drop" else list(dets)
    kept = []
    for d in dets:
        box = d.box
        if any(cover_ratio(box, p) >= min_cover for p in persons):
            kept.append(d)
    return kept


class PersonGate:
    """人形门控：懒加载 person 模型，每帧先找人、再筛 head/helmet 框。"""

    def __init__(self, cfg: GateConfig | None = None, base_dir: Path | None = None,
                 device: str = "auto"):
        self.cfg = cfg or GateConfig()
        self.device = device
        self.base_dir = base_dir
        self._impl = None
        self.reason = ""                     # 不可用时的原因（界面/日志用）
        self._dropped = 0
        self._persons = 0
        self._ready = False
        if self.cfg.enabled:
            self._load()

    # ---------------- 初始化 ----------------
    def _resolve(self) -> Path | None:
        p = Path(self.cfg.model_path)
        if p.is_absolute():
            return p if p.exists() else None
        cands = []
        if self.base_dir is not None:
            cands.append(Path(self.base_dir) / p)
        from .paths import resource_dir, user_dir
        cands += [resource_dir() / p, user_dir() / p, Path.cwd() / p]
        for c in cands:
            if c.exists():
                return c
        return None

    def _load(self) -> None:
        path = self._resolve()
        if path is None:
            self.reason = f"找不到人物模型 {self.cfg.model_path}（可运行 scripts/export_person.py 生成）"
            self._ready = False
            return
        try:
            from .detector import _OnnxBackend
            self._impl = _OnnxBackend(path, "cpu", iou=0.5, max_det=20, threads=self.cfg.threads)
            self._person_ids = [i for i, n in self._impl.names.items() if n == "person"]
            # 模型若是固定尺寸导出，就把 imgsz 对齐成它的输入尺寸，免得运行时报维度错
            shape = self._impl.session.get_inputs()[0].shape
            if isinstance(shape[-1], int) and shape[-1] > 0:
                self.cfg.imgsz = int(shape[-1])
            self._ready = True
            self.reason = ""
        except Exception as exc:            # noqa: BLE001
            self.reason = f"人物模型加载失败：{type(exc).__name__}: {exc}"
            self._ready = False

    @property
    def available(self) -> bool:
        return bool(self.cfg.enabled and self._ready and self._impl is not None)

    # ---------------- 主流程 ----------------
    def find_persons(self, frame: np.ndarray) -> list[tuple]:
        if not self.available:
            return []
        dets = self._impl.infer(frame, conf=self.cfg.conf, imgsz=self.cfg.imgsz)
        out = []
        for d in dets:
            if d.cls_id in getattr(self, "_person_ids", [0]):
                out.append(expand_box(d.box, self.cfg.expand))
        self._persons = len(out)
        return out

    def apply(self, frame: np.ndarray, dets: list[Detection]) -> list[Detection]:
        """过滤掉"不在人身上"的检测；门控不可用时原样返回。"""
        if not self.available or not dets:
            return dets
        persons = self.find_persons(frame)
        kept = keep_covered(dets, persons, self.cfg.min_cover, self.cfg.empty_policy)
        self._dropped = len(dets) - len(kept)
        return kept

    def stats(self) -> dict:
        return {
            "available": self.available,
            "persons": self._persons,
            "dropped": self._dropped,
            "reason": self.reason,
        }
