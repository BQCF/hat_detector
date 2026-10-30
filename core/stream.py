# core/stream.py
# Stream：一路视频源的完整封装 = 模型 + 采集源 + 报警 + 推理管道。
# 多路模式下，每路一个 Stream（各自跑一个模型），互不干扰。
from __future__ import annotations

import re

from .alarm import AlarmManager
from .config import AppConfig
from .detector import Detector
from .paths import user_dir
from .pipeline import InferPipeline, PipelineResult
from .roi import RoiRegion
from .sources import create_source
from .tracker import IouTracker


def safe_dir(name: str) -> str:
    """把显示名转成可当目录名的安全串。"""
    s = re.sub(r"[^\w\-]", "_", str(name)).strip("_")
    return s or "stream"


class Stream:
    """一路视频源。spec 是 {"name","source_type","source_value"}。"""

    def __init__(self, spec: dict, cfg: AppConfig, on_alarm):
        self.spec = spec
        self.cfg = cfg
        self._on_alarm = on_alarm
        self.name = str(spec.get("name") or f"{spec.get('source_type')} {spec.get('source_value')}")
        self.detector: Detector | None = None
        self.alarm: AlarmManager | None = None
        self.pipeline: InferPipeline | None = None
        self.error: str = ""
        self._finished = False

    @property
    def running(self) -> bool:
        return self.pipeline is not None and self.pipeline.running

    # ---------------- 生命周期 ----------------
    def start(self) -> None:
        try:
            cfg = self.cfg
            self.detector = Detector(
                model_path=cfg.model_path, device=cfg.device, conf=cfg.conf,
                iou=cfg.iou, imgsz=cfg.imgsz, max_det=cfg.max_det, backend=cfg.backend)

            src = create_source(self.spec.get("source_type", "camera"),
                                str(self.spec.get("source_value", "0")),
                                cfg.camera_width, cfg.camera_height, cfg.video_loop)
            src.open()

            # 每路独立的报警目录，避免多路写同一个 CSV 打架
            import copy
            acfg = copy.deepcopy(cfg.alarm)
            acfg.snapshot_dir = f"captures/{safe_dir(self.name)}"
            acfg.events_csv = f"captures/{safe_dir(self.name)}/events.csv"
            self.alarm = AlarmManager(acfg, base_dir=user_dir(),
                                      on_alarm=self._wrap_alarm)

            tracker = IouTracker(cfg.track.iou_thr, cfg.track.max_age) if cfg.track.enabled else None
            roi = RoiRegion.from_config(cfg.roi)
            self.pipeline = InferPipeline(self.detector, src, alarm=self.alarm,
                                          dedup=True, tracker=tracker, roi=roi)
            self.pipeline.start()
            self.error = ""
        except Exception as exc:                            # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            self.stop()

    def stop(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None
        self.detector = None

    def _wrap_alarm(self, ev) -> None:
        try:
            self._on_alarm(self.name, ev)
        except Exception:
            pass

    def latest(self) -> PipelineResult | None:
        if self.pipeline is None:
            return None
        return self.pipeline.latest()

    def finished(self) -> bool:
        return self.pipeline is not None and self.pipeline.finished
