# core/pipeline.py
# 推理管道：采集线程 -> 有界队列（丢旧帧）-> 推理线程 -> 结果队列。
# 主线程只负责取"最新一帧结果"来显示，永远不会被推理卡住。
from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .alarm import AlarmManager
from .detector import Detection, Detector
from .postprocess import count_classes, cross_class_dedup
from .sources import FrameSource


@dataclass
class PipelineResult:
    """一帧的完整结果。"""
    frame: np.ndarray
    detections: list[Detection] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    track_stats: dict = field(default_factory=dict)
    roi_active: bool = False
    infer_ms: float = 0.0
    fps: float = 0.0
    timestamp: float = 0.0
    alarming: bool = False


class InferPipeline:
    """采集 + 推理双线程管道。

    - 采集线程只负责读帧，队列满了就丢掉最旧的一帧（保证实时性）；
    - 推理线程取最新帧做检测，结果也只保留最新的几帧；
    - 主线程用 latest() 取当前结果来刷新界面。
    """

    def __init__(self, detector: Detector, source: FrameSource,
                 alarm: AlarmManager | None = None, dedup: bool = True,
                 tracker=None, roi=None):
        self.detector = detector
        self.source = source
        self.alarm = alarm
        self.dedup = dedup
        self.tracker = tracker
        self.roi = roi

        self._frame_q: queue.Queue = queue.Queue(maxsize=2)
        self._result_q: queue.Queue = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._latest: PipelineResult | None = None
        self._stamps: deque[float] = deque(maxlen=60)
        self._error: str = ""
        self._finished = False
        self._infer_done = False

    # ---------------- 生命周期 ----------------
    @property
    def running(self) -> bool:
        return bool(self._threads) and not self._stop.is_set()

    @property
    def finished(self) -> bool:
        """采集源读完**且**推理线程把剩下的帧处理完了。

        只看采集是否结束是不够的：图片源读完一帧就结束，但那一帧还要推理。
        """
        return self._finished and self._infer_done

    @property
    def error(self) -> str:
        return self._error

    def start(self) -> None:
        self._stop_threads()          # 只停线程，**不释放采集源**
        self._stop.clear()
        self._finished = False
        self._infer_done = False
        self._error = ""
        self._threads = [
            threading.Thread(target=self._capture_loop, daemon=True, name="capture"),
            threading.Thread(target=self._infer_loop, daemon=True, name="infer"),
        ]
        for t in self._threads:
            t.start()

    def _stop_threads(self) -> None:
        """停掉工作线程，但保留采集源（供重新 start 使用）。"""
        self._stop.set()
        for _ in range(len(self._threads)):
            try:
                self._frame_q.put_nowait(None)      # 唤醒可能阻塞的推理线程
            except queue.Full:
                break
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads = []

    def stop(self) -> None:
        """彻底停止：停线程并释放采集源（摄像头、视频句柄）。"""
        self._stop_threads()
        try:
            self.source.release()
        except Exception:
            pass

    def latest(self) -> PipelineResult | None:
        """取最新的结果（不阻塞）。"""
        try:
            while True:
                self._latest = self._result_q.get_nowait()
        except queue.Empty:
            pass
        return self._latest

    def _put_latest(self, q: queue.Queue, item) -> None:
        """往有界队列塞一帧；满了就丢掉最旧的，绝不阻塞采集/推理线程。"""
        while True:
            try:
                q.put_nowait(item)
                return
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    return

    # ---------------- 线程体 ----------------
    def _capture_loop(self) -> None:
        while not self._stop.is_set():
            ok, frame = self.source.read()
            if not ok or frame is None:
                self._finished = True
                break
            self._put_latest(self._frame_q, frame)

    def _infer_loop(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    frame = self._frame_q.get(timeout=0.5)
                except queue.Empty:
                    if self._finished:
                        break
                    continue
                if frame is None:
                    break
                try:
                    self._process(frame)
                except Exception as exc:            # noqa: BLE001
                    self._error = f"{type(exc).__name__}: {exc}"
                    break
        finally:
            self._infer_done = True

    def _process(self, frame: np.ndarray) -> None:
        t0 = time.time()
        dets = self.detector.infer(frame)
        infer_ms = (time.time() - t0) * 1000.0

        if self.dedup:
            dets = cross_class_dedup(dets, iou_thr=0.5)
        if self.tracker is not None:
            dets = self.tracker.update(dets)

        # 危险区域：拆成区域内 / 区域外，只对区域内统计与报警
        h, w = frame.shape[:2]
        if self.roi is not None and self.roi.active:
            inside, outside = self.roi.partition(dets, w, h)
            roi_active = True
        else:
            inside, outside = dets, []
            roi_active = False
        counts = count_classes(inside, self.detector.names)

        # 画框（复制一份，避免污染采集缓冲）：区域内正常色、区域外灰色，再叠区域多边形
        annotated = self.detector.draw(frame.copy(), inside, secondary=outside)
        if self.roi is not None:
            self.roi.draw(annotated)

        alarming = False
        if self.alarm is not None:
            alarming = self.alarm.update(counts, annotated.copy(), inside)

        track_stats = self.tracker.stats() if self.tracker is not None else {}

        now = time.time()
        self._stamps.append(now)
        fps = 0.0
        if len(self._stamps) >= 2:
            span = self._stamps[-1] - self._stamps[0]
            if span > 0:
                fps = (len(self._stamps) - 1) / span

        result = PipelineResult(frame=annotated, detections=inside, counts=counts,
                                track_stats=track_stats, roi_active=roi_active,
                                infer_ms=infer_ms, fps=fps, timestamp=now,
                                alarming=alarming)
        self._put_latest(self._result_q, result)
