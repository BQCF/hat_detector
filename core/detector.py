# core/detector.py
# 检测器：加载模型并对一帧 BGR 图做推理，返回结构化的检测框。
# 支持两种后端，接口完全一致，选哪个由 backend 参数 / 环境自动决定：
#   torch  —— ultralytics + PyTorch（开发机有 GPU 时最快，约 7ms）
#   onnx   —— onnxruntime（体积小、免 CUDA，打包成 exe 用它，CPU 约 20-40ms）
# 只依赖 numpy / cv2；ultralytics 与 onnxruntime 都是**用到才 import**。
from __future__ import annotations

import ast
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
    track_id: int = -1

    @property
    def box(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2


def find_latest_model(root: Path | None = None) -> Path | None:
    """在 runs/detect 与 models 下找最新的 best.pt / best.onnx（开发与打包后都对）。"""
    from .paths import resource_dir, user_dir
    roots = [root] if root else [resource_dir(), user_dir()]
    cands: list[Path] = []
    for r in roots:
        if r is None:
            continue
        cands += list((r / "runs" / "detect").glob("*/weights/best.*"))
        cands += list((r / "models").glob("*.onnx"))
        cands += list((r / "models").glob("*.pt"))
    cands = [p for p in cands if p.suffix.lower() in (".pt", ".onnx")]
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


# ---------------------------------------------------------------- torch 后端
class _TorchBackend:
    name = "torch"

    def __init__(self, model_path: Path, device: str, iou: float, max_det: int):
        import torch
        from ultralytics import YOLO

        if device == "auto":
            device = "0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.gpu = self.device != "cpu"
        self.iou = iou
        self.max_det = max_det
        self.model = YOLO(str(model_path))
        self.names: dict[int, str] = dict(self.model.names)

    def infer(self, frame_bgr: np.ndarray, conf: float, imgsz: int) -> list[Detection]:
        res = self.model.predict(
            source=frame_bgr, conf=conf, iou=self.iou, imgsz=imgsz,
            device=self.device, max_det=self.max_det, verbose=False)[0]
        out: list[Detection] = []
        boxes = res.boxes
        if boxes is not None and len(boxes) > 0:
            for (x1, y1, x2, y2), c, s in zip(boxes.xyxy.cpu().tolist(),
                                              boxes.cls.cpu().tolist(),
                                              boxes.conf.cpu().tolist()):
                cid = int(c)
                out.append(Detection(x1, y1, x2, y2, cid,
                                     self.names.get(cid, str(cid)), float(s)))
        return out


# ---------------------------------------------------------------- onnx 后端
class _OnnxBackend:
    name = "onnx"

    def __init__(self, model_path: Path, device: str, iou: float, max_det: int):
        import onnxruntime as ort

        self.iou = iou
        self.max_det = max_det
        avail = ort.get_available_providers()
        if device not in ("", "cpu", "auto") and "CUDAExecutionProvider" in avail:
            self.device = "cuda"
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            self.device = "cpu"
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.names = self._read_names()

    def _read_names(self) -> dict[int, str]:
        """ultralytics 导出的 ONNX 会把类别名写进 metadata。"""
        meta = self.session.get_modelmeta().custom_metadata_map or {}
        raw = meta.get("names")
        if not raw:
            return {}
        try:
            d = ast.literal_eval(raw) if isinstance(raw, str) else dict(raw)
            return {int(k): str(v) for k, v in d.items()}
        except Exception:
            return {}

    @staticmethod
    def _letterbox(img: np.ndarray, size: int):
        """等比缩放 + 灰边填充到 size×size，返回 (画布, 缩放比, 左边距, 上边距)。"""
        h, w = img.shape[:2]
        scale = min(size / w, size / h)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR) \
            if (nw, nh) != (w, h) else img
        left, top = (size - nw) // 2, (size - nh) // 2
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        canvas[top:top + nh, left:left + nw] = resized
        return canvas, scale, left, top

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
        """纯 numpy 的非极大值抑制，避免依赖 cv2 版本差异。"""
        if len(boxes) == 0:
            return []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = scores.argsort()[::-1]
        keep: list[int] = []
        while order.size:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            iw = np.maximum(0.0, np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]))
            ih = np.maximum(0.0, np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]))
            inter = iw * ih
            iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
            order = rest[iou <= iou_thr]
        return keep

    def infer(self, frame_bgr: np.ndarray, conf: float, imgsz: int) -> list[Detection]:
        h, w = frame_bgr.shape[:2]
        canvas, scale, dw, dh = self._letterbox(frame_bgr, imgsz)
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[None])   # (1,3,S,S)

        out = self.session.run(None, {self.input_name: blob})[0]
        pred = out[0].T                                             # (N, 4+nc)
        boxes_xywh = pred[:, :4]
        scores_all = pred[:, 4:]
        cls_ids = scores_all.argmax(axis=1)
        confs = scores_all.max(axis=1)

        mask = confs >= conf
        if not mask.any():
            return []
        boxes_xywh, confs, cls_ids = boxes_xywh[mask], confs[mask], cls_ids[mask]

        # cx,cy,w,h -> x1,y1,x2,y2（画布坐标）
        bx1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        by1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        bx2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        by2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

        # 还原到原图坐标
        bx1 = np.clip((bx1 - dw) / scale, 0, w)
        bx2 = np.clip((bx2 - dw) / scale, 0, w)
        by1 = np.clip((by1 - dh) / scale, 0, h)
        by2 = np.clip((by2 - dh) / scale, 0, h)
        boxes = np.stack([bx1, by1, bx2, by2], axis=1)

        # 按类别分别做 NMS（与 ultralytics 默认行为一致）
        kept: list[int] = []
        for cid in np.unique(cls_ids):
            idx = np.where(cls_ids == cid)[0]
            kept.extend(int(idx[k]) for k in self._nms(boxes[idx], confs[idx], self.iou))
        kept = sorted(kept, key=lambda i: float(confs[i]), reverse=True)[: self.max_det]

        return [Detection(float(bx1[i]), float(by1[i]), float(bx2[i]), float(by2[i]),
                          int(cls_ids[i]), self.names.get(int(cls_ids[i]), str(int(cls_ids[i]))),
                          float(confs[i]))
                for i in kept]


# ---------------------------------------------------------------- 统一入口
class Detector:
    """统一的检测器外观：内部按配置选择 torch 或 onnx 后端。"""

    def __init__(self, model_path: str = "", device: str = "auto",
                 conf: float = 0.25, iou: float = 0.45, imgsz: int = 640,
                 max_det: int = 50, backend: str = "auto"):
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.max_det = max_det

        path = Path(model_path) if model_path else find_latest_model()
        if not path or not Path(path).exists():
            raise FileNotFoundError(
                f"找不到模型权重：{model_path or 'runs/detect/*/weights/best.*'}。"
                "请先运行 python train.py 训练，或在设置里指定模型路径。")
        self.model_path = Path(path)
        self.backend_name = self._pick_backend(backend, self.model_path)

        if self.backend_name == "onnx":
            self._impl = _OnnxBackend(self.model_path, device, iou, max_det)
        else:
            self._impl = _TorchBackend(self.model_path, device, iou, max_det)
        self.names: dict[int, str] = self._impl.names
        self.device = getattr(self._impl, "device", "cpu")
        self.gpu = getattr(self._impl, "gpu", False)

    @staticmethod
    def _pick_backend(backend: str, path: Path) -> str:
        b = (backend or "auto").lower()
        if b in ("torch", "onnx"):
            return b
        if path.suffix.lower() == ".onnx":
            return "onnx"
        try:
            import ultralytics  # noqa: F401
            return "torch"
        except ImportError:
            pass
        fallback = path.with_suffix(".onnx")
        if fallback.exists():
            return "onnx"
        raise RuntimeError(
            "没有可用后端：既没装 ultralytics(torch)，也没找到同目录的 ONNX 模型。"
            f"\n请先 pip install ultralytics，或运行 python scripts/export.py 导出 {fallback.name}。")

    # ---------------- 推理 ----------------
    def infer(self, frame_bgr: np.ndarray,
              conf: float | None = None, imgsz: int | None = None) -> list[Detection]:
        return self._impl.infer(frame_bgr,
                                conf if conf is not None else self.conf,
                                imgsz or self.imgsz)

    # ---------------- 画框 ----------------
    def draw(self, frame_bgr: np.ndarray, dets: list[Detection],
             thickness: int = 2, secondary: list[Detection] | None = None) -> np.ndarray:
        """在帧上画检测框；secondary 里的框画成灰色（危险区域外）。

        标签会带上跟踪 id（如 head#12 0.87），便于看出"同一个人"被去重。
        """
        for d in dets:
            self._draw_one(frame_bgr, d, CLASS_COLORS.get(d.cls_name, (255, 255, 255)),
                           thickness, dim=False)
        for d in (secondary or []):
            self._draw_one(frame_bgr, d, (120, 120, 120), thickness, dim=True)
        return frame_bgr

    @staticmethod
    def _draw_one(frame_bgr, d: Detection, color, thickness: int, dim: bool) -> None:
        p1 = (int(d.x1), int(d.y1))
        p2 = (int(d.x2), int(d.y2))
        cv2.rectangle(frame_bgr, p1, p2, color, thickness)
        label = d.cls_name
        if d.track_id >= 0:
            label += f"#{d.track_id}"
        label += f" {d.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ty = max(p1[1] - 6, th + 6)
        cv2.rectangle(frame_bgr, (p1[0], ty - th - 8), (p1[0] + tw + 6, ty + 2),
                      color, -1)
        cv2.putText(frame_bgr, label, (p1[0] + 3, ty - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
