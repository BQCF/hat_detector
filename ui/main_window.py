# ui/main_window.py
# 主窗口（现代科技风暗黑模式）
# 布局：左右 7:3；左侧纯黑视频卡片；右侧三张卡片（运行控制 / 实时性能 / 报警事件）。
#
# 第二步：已接入 core 的检测管线——
#   Detector 加载模型 → create_source 打开采集源 → InferPipeline 双线程跑推理，
#   主线程用 QTimer 轮询最新结果刷新画面与指标（轮询比每帧发 Qt 信号更稳，
#   30FPS 下不会把事件队列打爆）。
#   报警事件流与弹窗留在第三步。
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import cv2
from PySide6.QtCore import QEvent, QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (QApplication, QFileDialog, QHBoxLayout,
                               QInputDialog, QLabel, QMainWindow, QMessageBox,
                               QPushButton, QSizePolicy, QVBoxLayout, QWidget)

from core.alarm import AlarmEvent, AlarmManager
from core.config import AppConfig
from core.detector import Detector
from core.pipeline import InferPipeline, PipelineResult
from core.sources import create_source

from . import theme
from .settings_dialog import SettingsDialog
from .widgets import (Card, Divider, EventStream, MetricRow, SourceSegments,
                      StatusDot, Toast, VideoArea)


class _AlarmBridge(QObject):
    """报警回调发生在推理线程里，用信号排队回界面线程再动控件。"""

    triggered = Signal(object)


class MainWindow(QMainWindow):
    """桌面主窗口：负责启停采集/推理，并把结果刷到界面上。"""

    SOURCES = [
        ("camera", "摄像头"),
        ("video", "视频文件"),
        ("image", "图片"),
        ("rtsp", "RTSP"),
    ]

    def __init__(self, cfg: AppConfig):
        super().__init__()
        self.cfg = cfg
        self._running = False
        self.detector: Detector | None = None
        self.pipeline: InferPipeline | None = None
        self._last: PipelineResult | None = None
        self._model_label = "—"
        self._alarming = False

        # 报警回调在推理线程里触发 → 用信号排队回界面线程
        self._alarm_bridge = _AlarmBridge()
        self._alarm_bridge.triggered.connect(self._on_alarm)

        self.setWindowTitle("安全帽佩戴检测系统")
        self.resize(1380, 830)
        self.setMinimumSize(1120, 700)
        self.setStyleSheet(theme.stylesheet())

        self._build_ui()
        self._bind_shortcuts()
        self._sync_state()
        self._update_source_hint()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

    # ============================================================ 布局
    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)

        outer = QHBoxLayout(root)
        outer.setContentsMargins(theme.PAD, theme.PAD, theme.PAD, theme.PAD)
        outer.setSpacing(theme.GAP)

        self.video = VideoArea()
        outer.addWidget(self.video, 7)

        right = QWidget()
        right.setMinimumWidth(360)
        right.setMaximumWidth(460)
        col = QVBoxLayout(right)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(theme.GAP)
        self.card_control = self._build_control_card()
        self.card_perf = self._build_perf_card()
        col.addWidget(self.card_control, 0)
        col.addWidget(self.card_perf, 0)
        col.addWidget(self._build_event_card(), 1)
        outer.addWidget(right, 3)

        # 浮层提示（挂在视频区上，右上角浮出）
        self._toast = Toast(self.video)
        self.event_stream.opened.connect(self._open_image)
        self.video.installEventFilter(self)      # 视频区尺寸变化时重新定位浮层
        self._place_toast()

    # ---------------------------------------------------- 卡片 A：控制与状态
    def _build_control_card(self) -> Card:
        card = Card("运行控制", "Ready")

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(10)

        self.btn_start = QPushButton("开始检测")
        self.btn_start.setObjectName("PrimaryBtn")
        self.btn_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_start.setMinimumHeight(44)

        self.btn_stop = QPushButton("停止")
        self.btn_stop.setObjectName("DangerBtn")
        self.btn_stop.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_stop.setMinimumHeight(44)

        btn_row.addWidget(self.btn_start, 3)
        btn_row.addWidget(self.btn_stop, 2)
        card.body.addLayout(btn_row)

        card.body.addWidget(Divider())

        status_row = QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.setSpacing(8)
        self.status_dot = StatusDot()
        self.status_text = QLabel("当前状态：已停止")
        self.status_text.setObjectName("StatusText")
        status_row.addWidget(self.status_dot)
        status_row.addWidget(self.status_text)
        status_row.addStretch(1)
        card.body.addLayout(status_row)

        card.body.addWidget(Divider())

        src_label = QLabel("选择数据源")
        src_label.setObjectName("MetricLabel")
        card.body.addWidget(src_label)

        self.sources = SourceSegments(self.SOURCES, current=self.cfg.source_type)
        card.body.addWidget(self.sources)

        self.source_hint = QLabel("")
        self.source_hint.setObjectName("CardHint")
        self.source_hint.setWordWrap(True)
        card.body.addWidget(self.source_hint)

        ghost_row = QHBoxLayout()
        ghost_row.setContentsMargins(0, 0, 0, 0)
        ghost_row.setSpacing(4)
        self.btn_shot = QPushButton("截图")
        self.btn_events = QPushButton("打开事件表")
        self.btn_settings = QPushButton("设置")
        for b in (self.btn_shot, self.btn_events, self.btn_settings):
            b.setObjectName("GhostBtn")
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            ghost_row.addWidget(b)
        ghost_row.addStretch(1)
        card.body.addLayout(ghost_row)

        self.btn_start.clicked.connect(self._start)
        self.btn_stop.clicked.connect(lambda: self._stop())
        self.sources.changed.connect(self._on_source_changed)
        self.btn_shot.clicked.connect(self._screenshot)
        self.btn_events.clicked.connect(self._open_events)
        self.btn_settings.clicked.connect(self._open_settings)
        return card

    def _bind_shortcuts(self) -> None:
        """F5 开始 / F6 停止 / Ctrl+S 截图。"""
        for seq, slot in (("F5", self._start),
                          ("F6", lambda: self._stop()),
                          ("Ctrl+S", self._screenshot)):
            act = QAction(self)
            act.setShortcut(QKeySequence(seq))
            act.triggered.connect(slot)
            self.addAction(act)

    # ---------------------------------------------------- 卡片 B：实时性能
    def _build_perf_card(self) -> Card:
        card = Card("实时性能监测", "待机")
        self.metric_fps = MetricRow("帧率", "--", "FPS")
        self.metric_ms = MetricRow("单帧耗时", "--", "ms")
        self.metric_det = MetricRow("本帧目标", "--", "个")

        card.body.addWidget(self.metric_fps)
        card.body.addWidget(Divider())
        card.body.addWidget(self.metric_ms)
        card.body.addWidget(Divider())
        card.body.addWidget(self.metric_det)
        return card

    # ---------------------------------------------------- 卡片 C：报警事件
    def _build_event_card(self) -> Card:
        card = Card("报警事件", "双击查看截图")
        self.event_stream = EventStream()
        self.event_stream.setSizePolicy(QSizePolicy.Policy.Expanding,
                                        QSizePolicy.Policy.Expanding)
        card.body.addWidget(self.event_stream, 1)
        return card

    # ============================================================ 状态
    def _sync_state(self) -> None:
        running = self._running
        if running and self._alarming:
            self.status_dot.set_state("alarm")
            self.status_text.setText("当前状态：⚠ 检测到未佩戴安全帽")
            color = theme.DANGER
        elif running:
            self.status_dot.set_state("run")
            self.status_text.setText("当前状态：运行中")
            color = theme.OK
        else:
            self.status_dot.set_state("stop")
            self.status_text.setText("当前状态：已停止")
            color = theme.TEXT_DIM
        self.status_text.setStyleSheet(
            f"font-size:14px;font-weight:600;color:{color};")
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        self.video.set_running(running)

    def _set_loading(self, text: str) -> None:
        self.status_text.setText(text)
        self.status_text.setStyleSheet(
            f"font-size:14px;font-weight:600;color:{theme.ACCENT};")
        QApplication.processEvents()          # 让"加载中"先画出来再阻塞

    def _reset_metrics(self) -> None:
        self.metric_fps.set_value("--")
        self.metric_ms.set_value("--")
        self.metric_det.set_value("--")
        if self.card_perf.hint_label:
            self.card_perf.hint_label.setText("待机")

    # ============================================================ 数据源
    def _update_source_hint(self) -> None:
        kind = self.cfg.source_type
        value = str(self.cfg.source_value or "")
        if kind == "camera":
            text = f"摄像头 {value}" if value.isdigit() else "本地摄像头"
        elif kind in ("video", "image"):
            p = Path(value)
            text = p.name if value and p.exists() else "未选择文件 · 点「开始检测」时选择"
        elif kind == "rtsp":
            text = value if value.startswith("rtsp://") else "未填写 RTSP 地址"
        else:
            text = ""
        self.source_hint.setText(text)

    def _source_ready(self, kind: str) -> bool:
        value = str(self.cfg.source_value or "")
        if kind in ("video", "image"):
            return bool(value) and Path(value).exists()
        if kind == "rtsp":
            return value.startswith("rtsp://")
        return True

    def _pick_source(self, kind: str) -> bool:
        """弹窗让用户选具体的源；取消返回 False。"""
        if kind == "camera":
            cur = int(self.cfg.source_value) if str(self.cfg.source_value).isdigit() else 0
            idx, ok = QInputDialog.getInt(
                self, "选择摄像头", "摄像头序号（0 通常是内置摄像头）：", cur, 0, 8)
            if not ok:
                return False
            self.cfg.source_value = str(idx)
            return True

        if kind in ("video", "image"):
            if kind == "video":
                filt = "视频文件 (*.mp4 *.avi *.mkv *.mov *.webm);;所有文件 (*)"
                title = "选择视频文件"
            else:
                filt = "图片 (*.jpg *.jpeg *.png *.bmp *.webp);;所有文件 (*)"
                title = "选择图片"
            path, _ = QFileDialog.getOpenFileName(self, title, "", filt)
            if not path:
                return False
            self.cfg.source_value = path
            return True

        if kind == "rtsp":
            cur = self.cfg.source_value if str(self.cfg.source_value).startswith("rtsp://") else "rtsp://"
            text, ok = QInputDialog.getText(self, "RTSP 摄像头", "流地址：", text=cur)
            if not ok or not text.strip():
                return False
            self.cfg.source_value = text.strip()
            return True
        return False

    def _on_source_changed(self, kind: str) -> None:
        prev = self.cfg.source_type
        if kind == prev:
            self._update_source_hint()
            return
        if self._running:
            self._teardown(keep_view=False)
        if self._pick_source(kind):
            self.cfg.source_type = kind
        else:
            self.sources.set_current(prev)     # 用户取消 → 退回原来的选项
        self._update_source_hint()

    # ============================================================ 启动 / 停止
    def _start(self) -> None:
        if self._running:
            return

        kind = self.cfg.source_type
        # 视频/图片/RTSP 还没选好具体源时，先弹窗选
        if not self._source_ready(kind):
            if not self._pick_source(kind):
                return
            self._update_source_hint()

        self._teardown(keep_view=False)
        self._set_loading("当前状态：加载模型中…")

        try:
            self.detector = Detector(
                model_path=self.cfg.model_path, device=self.cfg.device,
                conf=self.cfg.conf, iou=self.cfg.iou, imgsz=self.cfg.imgsz,
                max_det=self.cfg.max_det, backend=self.cfg.backend)
            source = create_source(kind, str(self.cfg.source_value),
                                   self.cfg.camera_width, self.cfg.camera_height,
                                   self.cfg.video_loop)
            source.open()
            alarm = AlarmManager(self.cfg.alarm,
                                 on_alarm=self._alarm_bridge.triggered.emit)
            self.pipeline = InferPipeline(self.detector, source, alarm=alarm, dedup=True)
            self.pipeline.start()
        except Exception as exc:                            # noqa: BLE001
            self.detector = None
            self.pipeline = None
            self._sync_state()
            QMessageBox.warning(self, "启动失败",
                                f"{type(exc).__name__}\n\n{exc}")
            return

        self._model_label = self.detector.model_path.name
        self._running = True
        self._sync_state()
        if self.card_control.hint_label:
            self.card_control.hint_label.setText(
                f"{self.detector.backend_name} · {self.detector.device}")
        if self.card_perf.hint_label:
            self.card_perf.hint_label.setText("实时")
        self.video.set_tags(fps=0, model=self._model_label)
        self.timer.start(self.cfg.ui_interval_ms)

    def _stop(self, silent: bool = False) -> None:
        self._teardown(keep_view=False)

    def _teardown(self, keep_view: bool = False) -> None:
        """停采集/推理并复位界面；keep_view=True 时保留最后一帧（图片结果用）。"""
        self.timer.stop()
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None
        self.detector = None
        self._running = False
        self._alarming = False
        self._toast.hide()
        self._sync_state()
        if not keep_view:
            self._last = None
            self.video.reset_view()
            self._reset_metrics()

    # ============================================================ 每帧刷新
    def _tick(self) -> None:
        if self.pipeline is None:
            return

        if self.pipeline.error:
            err = self.pipeline.error
            self._teardown(keep_view=False)
            QMessageBox.warning(self, "检测出错", err)
            return

        res = self.pipeline.latest()
        if res is not None and (self._last is None or res.timestamp != self._last.timestamp):
            self._last = res
            self.video.set_frame(res.frame)
            # 单张图片只跑一帧，没有吞吐量可算，显示 -- 更准确
            self.metric_fps.set_value(f"{res.fps:.0f}" if res.fps > 0 else "--")
            self.metric_ms.set_value(f"{res.infer_ms:.1f}")
            self.metric_det.set_value(str(len(res.detections)))
            self.video.set_tags(fps=res.fps, model=self._model_label,
                                alarming=res.alarming)
            # 报警状态变了才刷状态栏，避免每帧重绘
            if bool(res.alarming) != self._alarming:
                self._alarming = bool(res.alarming)
                self._sync_state()

        # 采集源跑完（单张图片 / 不循环的视频）：保留最后一帧
        if self.pipeline.finished:
            self._teardown(keep_view=True)
            self.status_text.setText("当前状态：已结束（处理完毕）")
            self.status_text.setStyleSheet(
                f"font-size:14px;font-weight:600;color:{theme.TEXT_DIM};")

    # ============================================================ 次要操作
    def _screenshot(self) -> None:
        if self._last is None:
            QMessageBox.information(self, "截图", "当前没有画面，先开始检测。")
            return
        folder = self.cfg.resolve(self.cfg.alarm.snapshot_dir)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(str(path), self._last.frame)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _open_events(self) -> None:
        path = self.cfg.resolve(self.cfg.alarm.events_csv)
        if not path.exists():
            QMessageBox.information(self, "事件表", "还没有报警事件。")
            return
        subprocess.Popen(["explorer", "/select,", str(path)])

    def _open_settings(self) -> None:
        dlg = SettingsDialog(self.cfg, self)
        if dlg.exec():
            self._update_source_hint()
            if self.card_control.hint_label:
                self.card_control.hint_label.setText("已保存 · 下次开始生效")

    # ============================================================ 报警
    def _on_alarm(self, ev: AlarmEvent) -> None:
        """报警事件：写进事件流 + 浮层提示（本方法在界面线程执行）。"""
        self.event_stream.add_event(ev.time_str, ev.max_conf, ev.cls_name,
                                    ev.image_path)
        self._toast.popup(f"检测到 {ev.count} 处未佩戴安全帽",
                          f"{ev.time_str}　最高置信度 {ev.max_conf:.2f}")
        self._place_toast()

    def _open_image(self, path: str) -> None:
        """双击事件条目 → 用系统看图打开抓拍。"""
        if path and Path(path).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _place_toast(self) -> None:
        """把浮层提示放到视频区右上角。"""
        if not hasattr(self, "_toast"):
            return
        margin = 18
        self._toast.move(max(margin, self.video.width() - self._toast.width() - margin),
                         margin)

    def eventFilter(self, obj, event):                        # noqa: N802
        if obj is self.video and event.type() == QEvent.Type.Resize:
            self._place_toast()
        return super().eventFilter(obj, event)

    # ============================================================ 收尾
    def showEvent(self, event) -> None:                       # noqa: N802
        super().showEvent(event)
        theme.apply_dark_titlebar(self)

    def closeEvent(self, event) -> None:                      # noqa: N802
        self._teardown(keep_view=False)
        super().closeEvent(event)
