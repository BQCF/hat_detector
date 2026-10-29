# core/detector.py
# 检测器：加载模型并对一帧 BGR 图做推理，返回结构化的检测框。
# 只依赖 ultralytics / opencv / numpy，不依赖 Qt。
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# 检测框配色（BGR）：head=红(未戴帽/危险) helmet=蓝 person=绿
CLASS_COLORS = {
    "head": (79, 77, 255),
    "helmet": (255, 144, 24),
    "person": (60, 200, 60),
}


@dataclass
class Detection:
    """一个检测框。坐标是像素值。"""
    x1: float
    y1: float
    x2: float
    y2: float
    cls_id: int
    cls_name: str
    conf: float

    @property
    def box(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2


def find_latest_model(root: Path | None = None) -> Path | None:
    """在 runs/detect 下找最新的 weights/best.pt。"""
    root = root or Path(__file__).resolve().parent.parent
    cands = sorted((root / "runs" / "detect").glob("*/weights/best.pt"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


class Detector:
    """封装 YOLO 模型，统一返回 Detection 列表。"""

    def __init__(self, model_path: str = "", device: str = "auto",
                 conf: float = 0.25, iou: float = 0.45, imgsz: int = 640,
                 max_det: int = 50):
        from ultralytics import YOLO
        import torch

        if device == "auto":
            device = "0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.max_det = max_det

        path = Path(model_path) if model_path else find_latest_model()
        if not path or not Path(path).exists():
            raise FileNotFoundError(
                f"找不到模型权重：{model_path or 'runs/detect/*/weights/best.pt'}。"
                "请先运行 python train.py 训练，或在设置里指定模型路径。")
        self.model_path = Path(path)
        self.model = YOLO(str(self.model_path))
        self.names: dict[int, str] = dict(self.model.names)
        self.gpu = self.device != "cpu"

    # ---------------- 推理 ----------------
    def infer(self, frame_bgr: np.ndarray,
              conf: float | None = None, imgsz: int | None = None) -> list[Detection]:
        """对一帧 BGR 图做检测，返回 Detection 列表。"""
        results = self.model.predict(
            source=frame_bgr,
            conf=conf if conf is not None else self.conf,
            iou=self.iou,
            imgsz=imgsz or self.imgsz,
            device=self.device,
            max_det=self.max_det,
            verbose=False,
        )
        res = results[0]
        out: list[Detection] = []
        boxes = res.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().tolist()
            cls = boxes.cls.cpu().tolist()
            score = boxes.conf.cpu().tolist()
            for (x1, y1, x2, y2), c, s in zip(xyxy, cls, score):
                cid = int(c)
                out.append(Detection(x1, y1, x2, y2, cid,
                                     self.names.get(cid, str(cid)), float(s)))
        return out

    # ---------------- 画框 ----------------
    def draw(self, frame_bgr: np.ndarray, dets: list[Detection],
             thickness: int = 2) -> np.ndarray:
        """在帧上画检测框（原地修改并返回同一张图）。"""
        for d in dets:
            color = CLASS_COLORS.get(d.cls_name, (255, 255, 255))
            p1 = (int(d.x1), int(d.y1))
            p2 = (int(d.x2), int(d.y2))
            cv2.rectangle(frame_bgr, p1, p2, color, thickness)
            label = f"{d.cls_name} {d.conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            ty = max(p1[1] - 6, th + 6)
            cv2.rectangle(frame_bgr, (p1[0], ty - th - 8), (p1[0] + tw + 6, ty + 2),
                          color, -1)
            cv2.putText(frame_bgr, label, (p1[0] + 3, ty - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        return frame_bgr
