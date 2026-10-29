# ui/main_window.py
# 桌面主窗口：左边画面、右边信息栏、上方工具栏、下方状态栏。
# 线程安全：检测在后台线程跑，通过 Qt 信号把结果交给界面线程更新。
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import cv2
from PySide6.QtCore import QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (QApplication, QFileDialog, QInputDialog, QLabel,
                               QMainWindow, QMessageBox, QSplitter, QVBoxLayout,
                               QWidget)

from core.alarm import AlarmEvent, AlarmManager
from core.config import AppConfig, save_config
from core.detector import Detector, find_latest_model
from core.pipeline import InferPipeline, PipelineResult
from core.sources import create_source
from ui.settings_dialog import SettingsDialog
from ui.side_panel import SidePanel
from ui.video_view import VideoView


class _Bridge(QObject):
    """把后台线程的事件安全地送到界面线程（跨线程 emit 会自动排队）。"""
    alarm_happened = Signal(object)


class MainWindow(QMainWindow):
    def __init__(self, cfg: AppConfig):
        super().__init__()
        self.cfg = cfg
        self.detector: Detector | None = None
        self.pipeline: InferPipeline | None = None
        self.alarm: AlarmManager | None = None
        self._last_result: PipelineResult | None = None

        self._bridge = _Bridge()
        self._bridge.alarm_happened.connect(self._on_alarm_ui)

        self.setWindowTitle("安全帽佩戴检测系统")
        self.resize(1280, 780)
        self._build_ui()
        self._build_toolbar()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self._set_status("就绪。请选择采集源后点击「开始检测」。")

    # ---------------- UI 搭建 ----------------
    def _build_ui(self) -> None:
        self.video = VideoView("无画面\n\n点击上方按钮选择摄像头 / 视频 / 图片 / RTSP")
        self.panel = SidePanel()
        self.panel.open_path.connect(self._open_image_external)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.video)
        splitter.addWidget(self.panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([950, 330])

        container = QWidget()
        lay = QVBoxLayout(container)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.addWidget(splitter)
        self.setCentralWidget(container)

        self.status_label = QLabel()
        self.statusBar().addWidget(self.status_label, stretch=1)

    def _build_toolbar(self) -> None:
        tb = self.addToolBar("主工具栏")
        tb.setMovable(False)
        tb.setIconSize(tb.iconSize())

        self.act_camera = QAction("📷 摄像头", self)
        self.act_video = QAction("🎬 视频文件", self)
        self.act_image = QAction("🖼 图片", self)
        self.act_rtsp = QAction("📡 RTSP", self)
        self.act_start = QAction("▶ 开始检测", self)
        self.act_stop = QAction("⏹ 停止", self)
        self.act_shot = QAction("📸 截图", self)
        self.act_settings = QAction("⚙ 设置", self)
        self.act_events = QAction("📁 事件表", self)

        self.act_start.setShortcut(QKeySequence("F5"))
        self.act_stop.setShortcut(QKeySequence("F6"))
        self.act_shot.setShortcut(QKeySequence("Ctrl+S"))

        for a in (self.act_camera, self.act_video, self.act_image, self.act_rtsp):
            tb.addAction(a)
        tb.addSeparator()
        for a in (self.act_start, self.act_stop, self.act_shot):
            tb.addAction(a)
        tb.addSeparator()
        for a in (self.act_settings, self.act_events):
            tb.addAction(a)

        self.act_camera.triggered.connect(self._pick_camera)
        self.act_video.triggered.connect(self._pick_video)
        self.act_image.triggered.connect(self._pick_image)
        self.act_rtsp.triggered.connect(self._pick_rtsp)
        self.act_start.triggered.connect(self._start)
        self.act_stop.triggered.connect(self._stop)
        self.act_shot.triggered.connect(self._screenshot)
        self.act_settings.triggered.connect(self._show_settings)
        self.act_events.triggered.connect(self._open_events)

    # ---------------- 采集源选择 ----------------
    def _pick_camera(self) -> None:
        index, ok = QInputDialog.getInt(self, "摄像头", "摄像头序号：", 0, 0, 8)
        if ok:
            self.cfg.source_type = "camera"
            self.cfg.source_value = str(index)
            self._start()

    def _pick_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "", "视频文件 (*.mp4 *.avi *.mkv *.mov *.webm);;所有文件 (*)")
        if path:
            self.cfg.source_type = "video"
            self.cfg.source_value = path
            self._start()

    def _pick_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片 (*.jpg *.jpeg *.png *.bmp *.webp);;所有文件 (*)")
        if path:
            self.cfg.source_type = "image"
            self.cfg.source_value = path
            self._start()

    def _pick_rtsp(self) -> None:
        text, ok = QInputDialog.getText(
            self, "RTSP 流", "请输入 RTSP 地址：",
            text=self.cfg.source_value if self.cfg.source_type == "rtsp" else "rtsp://")
        if ok and text.strip():
            self.cfg.source_type = "rtsp"
            self.cfg.source_value = text.strip()
            self._start()

    # ---------------- 启动 / 停止 ----------------
    def _start(self) -> None:
        self._stop(silent=True)
        self._set_status("正在加载模型…")
        QApplication.processEvents()

        try:
            self.detector = Detector(
                model_path=self.cfg.model_path, device=self.cfg.device,
                conf=self.cfg.conf, iou=self.cfg.iou, imgsz=self.cfg.imgsz,
                max_det=self.cfg.max_det)
            self.alarm = AlarmManager(self.cfg.alarm,
                                      on_alarm=self._bridge.alarm_happened.emit)
            source = create_source(self.cfg.source_type, self.cfg.source_value,
                                   self.cfg.camera_width, self.cfg.camera_height,
                                   self.cfg.video_loop)
            source.open()
            self.pipeline = InferPipeline(self.detector, source,
                                          alarm=self.alarm, dedup=True)
            self.pipeline.start()
        except Exception as exc:                            # noqa: BLE001
            self._fail(f"无法启动检测：\n{type(exc).__name__}: {exc}")
            return

        self.panel.set_model(f"{self.detector.model_path.name}（{self.detector.device}）")
        self.panel.set_source(f"{self.cfg.source_type} → {self.cfg.source_value}")
        self.panel.set_state(True)
        self.timer.start(self.cfg.ui_interval_ms)
        self._set_status(f"检测运行中：{self.cfg.source_type} → {self.cfg.source_value}")

    def _stop(self, silent: bool = False) -> None:
        self.timer.stop()
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None
        if self.alarm is not None:
            self.alarm.reset()
        self.panel.set_state(False)
        if not silent:
            self._set_status("已停止。")

    # ---------------- 刷新 ----------------
    def _tick(self) -> None:
        if self.pipeline is None:
            return
        if self.pipeline.error:
            self._fail(f"检测出错：{self.pipeline.error}")
            self._stop(silent=True)
            return

        res = self.pipeline.latest()
        if res is None:
            if self.pipeline.finished:
                self._stop(silent=True)
                self._set_status(f"已处理完毕（{self.cfg.source_type}）。")
            return

        self._last_result = res
        self.video.set_frame(res.frame)
        self.panel.set_perf(res.fps, res.infer_ms, len(res.detections))
        self.panel.set_counts(res.counts)
        self.panel.set_state(True, res.alarming)

    # ---------------- 报警（在界面线程里执行） ----------------
    def _on_alarm_ui(self, ev: AlarmEvent) -> None:
        text = (f"{ev.time_str}   {ev.cls_name} ×{ev.count}   "
                f"conf {ev.max_conf:.2f}")
        self.panel.add_event(text, ev.image_path)
        self.statusBar().showMessage(
            f"⚠ 报警：{ev.time_str} 检测到 {ev.count} 处未佩戴安全帽", 8000)

    # ---------------- 其它动作 ----------------
    def _screenshot(self) -> None:
        if self._last_result is None:
            QMessageBox.information(self, "截图", "当前没有画面。")
            return
        d = self.cfg.resolve(self.cfg.alarm.snapshot_dir)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(str(path), self._last_result.frame)
        self._set_status(f"截图已保存：{path}")
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _show_settings(self) -> None:
        dlg = SettingsDialog(self.cfg, self)
        if dlg.exec():
            save_config(self.cfg)
            self._set_status("设置已保存。下次点「开始检测」时生效。")

    def _open_events(self) -> None:
        path = self.cfg.resolve(self.cfg.alarm.events_csv)
        if not path.exists():
            QMessageBox.information(self, "事件表", "还没有报警事件。")
            return
        self._open_in_explorer(path)

    def _open_image_external(self, path: str) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    @staticmethod
    def _open_in_explorer(path: Path) -> None:
        if sys.platform == "win32":
            subprocess.Popen(["explorer", "/select,", str(path)])
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent)))

    # ---------------- 小工具 ----------------
    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _fail(self, message: str) -> None:
        self._set_status("出错了，详见弹窗。")
        QMessageBox.warning(self, "出错", message)

    def closeEvent(self, event) -> None:                     # noqa: N802
        self._stop(silent=True)
        super().closeEvent(event)
