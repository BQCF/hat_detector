# core/alarm.py
# 报警闭环：连续帧去抖确认 -> 抓拍 + 事件 CSV + 声音提示。
from __future__ import annotations

import csv
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from .config import AlarmConfig
from .paths import user_dir

CSV_HEADER = ["时间", "类别", "最高置信度", "命中数量", "抓拍图片"]


def _beep() -> None:
    """非阻塞地响一声系统提示音（Windows）。"""
    try:
        import winsound
        winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS | winsound.SND_ASYNC)
    except Exception:
        pass    # 非 Windows 或无声卡时静默跳过


@dataclass
class AlarmEvent:
    time_str: str
    cls_name: str
    max_conf: float
    count: int
    image_path: str


class AlarmManager:
    """把每帧的检测结果变成"一次报警事件"：去抖、留证、发声。

    用法：每帧调用一次 update()，返回当前是否处于报警状态。
    on_alarm 回调在真正触发通知时被调用（在推理线程里执行，别在里面碰 UI）。
    """

    def __init__(self, cfg: AlarmConfig, base_dir: Path | None = None,
                 on_alarm=None):
        self.cfg = cfg
        self.base_dir = base_dir or user_dir()
        self.on_alarm = on_alarm
        self.counter = 0
        self.is_alarming = False
        self._last_notify = 0.0
        self._seq = 0

    @property
    def snapshot_dir(self) -> Path:
        d = self.cfg.snapshot_dir
        p = Path(d)
        return p if p.is_absolute() else (self.base_dir / p)

    @property
    def events_csv(self) -> Path:
        p = Path(self.cfg.events_csv)
        return p if p.is_absolute() else (self.base_dir / p)

    def reset(self) -> None:
        self.counter = 0
        self.is_alarming = False

    def update(self, counts: dict[str, int], frame_bgr=None,
               detections=None) -> bool:
        """喂入一帧的统计结果。返回当前是否报警。"""
        if not self.cfg.enabled:
            return False

        hit = counts.get(self.cfg.alarm_class, 0) > 0
        self.counter = self.counter + 1 if hit else max(0, self.counter - 1)
        was = self.is_alarming
        self.is_alarming = self.counter >= self.cfg.confirm_frames

        now = time.time()
        rising = self.is_alarming and not was
        periodic = self.is_alarming and (now - self._last_notify) >= self.cfg.cooldown_sec
        if rising or periodic:
            self._notify(counts, frame_bgr, detections)
            self._last_notify = now
        return self.is_alarming

    def _notify(self, counts, frame_bgr, detections) -> None:
        self._seq += 1
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        cls = self.cfg.alarm_class
        hit_dets = [d for d in (detections or []) if d.cls_name == cls]
        max_conf = max((d.conf for d in hit_dets), default=0.0)
        count = counts.get(cls, 0)

        img_path = ""
        if self.cfg.snapshot and frame_bgr is not None:
            d = self.snapshot_dir
            d.mkdir(parents=True, exist_ok=True)
            fname = f"{time.strftime('%Y%m%d_%H%M%S')}_{self._seq:03d}.jpg"
            img_path = str(d / fname)
            cv2.imwrite(img_path, frame_bgr)

        event = AlarmEvent(ts, cls, float(max_conf), int(count), img_path)

        if self.cfg.snapshot or self.cfg.enabled:
            self._append_csv(event)
        if self.cfg.sound:
            threading.Thread(target=_beep, daemon=True).start()
        if self.on_alarm:
            try:
                self.on_alarm(event)
            except Exception:
                pass    # 回调异常不应该影响检测主流程

    def _append_csv(self, ev: AlarmEvent) -> None:
        path = self.events_csv
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists()
        # utf-8-sig 让 Excel 打开中文表头不乱码
        with path.open("a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(CSV_HEADER)
            w.writerow([ev.time_str, ev.cls_name, f"{ev.max_conf:.3f}",
                        ev.count, ev.image_path])
