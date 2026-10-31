# ui/main_window.py
# 主窗口（现代科技风暗黑模式）。单路 / 多路共用一套 Stream 模型：
#   单路：右侧 SourceSegments 选源，开始后大画面直接显示；
#   多路：左侧频道列表（缩略图+状态），点击切大画面，每路各跑一个模型，后台并行。
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import cv2
from PySide6.QtCore import QEvent, QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices, QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QInputDialog,
                               QLabel, QMainWindow, QMessageBox, QPushButton,
                               QSizePolicy, QVBoxLayout, QWidget)

from core.alarm import AlarmEvent
from core.config import AppConfig, save_config
from core.pipeline import PipelineResult
from core.report import export_events_excel
from core.roi import RoiRegion
from core.sources import list_cameras
from core.stream import Stream

from . import theme
from .channel_panel import ChannelPanel
from .roi_editor import RoiEditor
from .settings_dialog import SettingsDialog
from .widgets import (Card, Divider, EventStream, MetricRow, SourceSegments,
                      StatusDot, Toast, VideoArea)


class _AlarmBridge(QObject):
    """报警回调在推理线程里触发，用信号排队回界面线程。载荷 = (来源名, AlarmEvent)。"""

    triggered = Signal(object)


class MainWindow(QMainWindow):
    """桌面主窗口：管理一路或多路 Stream，把结果刷到界面上。"""

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
        self._multi = bool(cfg.cameras)          # 多路模式：configs 里有 cameras 列表
        self._streams: dict[str, Stream] = {}
        self._selected: str | None = None
        self._last: PipelineResult | None = None
        self._model_label = "—"
        self._alarming = False
        self._tick_no = 0

        self._alarm_bridge = _AlarmBridge()
        self._alarm_bridge.triggered.connect(self._on_alarm)

        self.setWindowTitle("安全帽佩戴检测系统")
        self.resize(1440, 840)
        self.setMinimumSize(1180, 700)
        self.setStyleSheet(theme.stylesheet())

        self._build_ui()
        self._roi_editor = RoiEditor(self.video)
        self._roi_editor.finished.connect(self._on_roi_finished)
        self._roi_editor.cancelled.connect(lambda: None)
        self._populate_channels()
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

        if self._multi:
            self.channels = ChannelPanel()
            self.channels.selected.connect(self._select_stream)
            self.channels.remove_requested.connect(self._remove_stream)
            outer.addWidget(self.channels, 0)

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

        self._toast = Toast(self.video)
        self.event_stream.opened.connect(self._open_image)
        self.video.installEventFilter(self)
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

        if self._multi:
            add_row = QHBoxLayout()
            add_row.setSpacing(8)
            self.btn_add_cam = QPushButton("＋ 摄像头")
            self.btn_add_rtsp = QPushButton("＋ RTSP")
            for b in (self.btn_add_cam, self.btn_add_rtsp):
                b.setObjectName("SegBtn")
                b.setCursor(Qt.CursorShape.PointingHandCursor)
                add_row.addWidget(b)
            card.body.addLayout(add_row)
            self.source_hint = QLabel("")
            self.source_hint.setObjectName("CardHint")
            self.source_hint.setWordWrap(True)
            card.body.addWidget(self.source_hint)
            self.sources = None
        else:
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
        self.btn_events = QPushButton("事件表")
        self.btn_export = QPushButton("导出 Excel")
        self.btn_roi = QPushButton("画区域")
        self.btn_roi_clear = QPushButton("清区域")
        self.btn_settings = QPushButton("设置")
        for b in (self.btn_shot, self.btn_events, self.btn_export,
                  self.btn_roi, self.btn_roi_clear, self.btn_settings):
            b.setObjectName("GhostBtn")
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            ghost_row.addWidget(b)
        ghost_row.addStretch(1)
        card.body.addLayout(ghost_row)

        self.btn_start.clicked.connect(self._start)
        self.btn_stop.clicked.connect(lambda: self._stop())
        if self.sources is not None:
            self.sources.changed.connect(self._on_source_changed)
        else:
            self.btn_add_cam.clicked.connect(self._add_camera)
            self.btn_add_rtsp.clicked.connect(self._add_rtsp)
        self.btn_shot.clicked.connect(self._screenshot)
        self.btn_events.clicked.connect(self._open_events)
        self.btn_export.clicked.connect(self._export_excel)
        self.btn_roi.clicked.connect(self._enter_roi)
        self.btn_roi_clear.clicked.connect(self._clear_roi)
        self.btn_settings.clicked.connect(self._open_settings)
        return card

    def _bind_shortcuts(self) -> None:
        for seq, slot in (("F5", self._start),
                          ("F6", lambda: self._stop()),
                          ("Ctrl+S", self._screenshot)):
            act = QAction(self)
            act.setShortcut(QKeySequence(seq))
            act.triggered.connect(slot)
            self.addAction(act)

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

    def _build_event_card(self) -> Card:
        card = Card("报警事件", "双击查看截图")
        self.event_stream = EventStream()
        self.event_stream.setSizePolicy(QSizePolicy.Policy.Expanding,
                                        QSizePolicy.Policy.Expanding)
        card.body.addWidget(self.event_stream, 1)
        return card

    # ============================================================ 频道（多路）
    def _populate_channels(self) -> None:
        if not self._multi:
            return
        for c in self.cfg.cameras:
            self.channels.add_channel(c.get("name") or "")
        if self.cfg.cameras:
            self._selected = self.cfg.cameras[0].get("name") or ""

    def _select_stream(self, name: str) -> None:
        self._selected = name
        self._last = None

    def _add_camera(self) -> None:
        res = self._choose_camera()
        if res is None:
            return
        idx, label = res
        name = label.split("（")[0]          # 去掉分辨率后缀，作频道名
        self.cfg.cameras.append({"name": name, "source_type": "camera",
                                 "source_value": str(idx)})
        self._commit_cameras()

    def _choose_camera(self):
        """枚举可用摄像头并弹列表选择，返回 (index, 显示名) 或 None（取消）。"""
        cams = list_cameras()
        if cams:
            labels = [name for _, name in cams]
            cur = 0
            if str(self.cfg.source_value).isdigit():
                cur = next((k for k, (i, _) in enumerate(cams)
                            if i == int(self.cfg.source_value)), 0)
            label, ok = QInputDialog.getItem(self, "选择摄像头", "可用的本地摄像头：",
                                             labels, cur, False)
            if not ok:
                return None
            return cams[labels.index(label)]
        idx, ok = QInputDialog.getInt(self, "选择摄像头", "未枚举到摄像头，手动输入序号：", 0, 0, 8)
        if not ok:
            return None
        return idx, f"摄像头 {idx}"

    def _add_rtsp(self) -> None:
        text, ok = QInputDialog.getText(self, "添加 RTSP", "流地址：", text="rtsp://")
        if not ok or not text.strip():
            return
        default_name = f"RTSP {len(self.cfg.cameras) + 1}"
        name, _ = QInputDialog.getText(self, "给这路起名", "名称：", text=default_name)
        self.cfg.cameras.append({"name": (name.strip() or default_name),
                                 "source_type": "rtsp", "source_value": text.strip()})
        self._commit_cameras()

    def _commit_cameras(self) -> None:
        save_config(self.cfg)
        spec = self.cfg.cameras[-1]
        name = spec.get("name") or ""
        self.channels.add_channel(name)
        self._selected = self._selected or name
        if self._running:                      # 正在跑就立刻把这路也启动
            s = Stream(spec, self.cfg, self._alarm_bridge.triggered.emit)
            self._streams[s.name] = s
            s.start()

    def _remove_stream(self, name: str) -> None:
        self.cfg.cameras = [c for c in self.cfg.cameras if c.get("name") != name]
        save_config(self.cfg)
        if name in self._streams:
            self._streams[name].stop()
            del self._streams[name]
        self.channels.remove_channel(name)
        if self._selected == name:
            self._selected = next(iter(self._streams), None)
        if self._streams and not self._selected:
            self._selected = next(iter(self._streams))

    # ============================================================ 状态
    def _sync_state(self) -> None:
        running = self._running
        if running and self._alarming:
            self.status_dot.set_state("alarm")
            self.status_text.setText("当前状态：⚠ 检测到未佩戴安全帽")
            color = theme.DANGER
        elif running:
            self.status_dot.set_state("run")
            n = len(self._streams)
            self.status_text.setText(f"当前状态：运行中{(' · ' + str(n) + ' 路') if n > 1 else ''}")
            color = theme.OK
        else:
            self.status_dot.set_state("stop")
            self.status_text.setText("当前状态：已停止")
            color = theme.TEXT_DIM
        self.status_text.setStyleSheet(f"font-size:14px;font-weight:600;color:{color};")
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        self.video.set_running(running)

    def _set_loading(self, text: str) -> None:
        self.status_text.setText(text)
        self.status_text.setStyleSheet(f"font-size:14px;font-weight:600;color:{theme.ACCENT};")
        QApplication.processEvents()

    def _reset_metrics(self) -> None:
        self.metric_fps.set_value("--")
        self.metric_ms.set_value("--")
        self.metric_det.set_value("--")
        if self.card_perf.hint_label:
            self.card_perf.hint_label.setText("待机")

    def _update_channel_status(self) -> None:
        if not self._multi:
            return
        for name, s in self._streams.items():
            if s.error:
                self.channels.set_status(name, "stop", "启动失败")
            elif s.running and s.alarm is not None and s.alarm.is_alarming:
                self.channels.set_status(name, "alarm", "报警中")
            elif s.running:
                self.channels.set_status(name, "run", "运行中")
            else:
                self.channels.set_status(name, "stop", "已停止")

    # ============================================================ 数据源（单路）
    def _update_source_hint(self) -> None:
        if self._multi:
            self.source_hint.setText(f"{len(self.cfg.cameras)} 路 · 点击左侧频道切换大画面")
            return
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
        if kind == "camera":
            res = self._choose_camera()
            if res is None:
                return False
            self.cfg.source_value = str(res[0])
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
            self.sources.set_current(prev)
        self._update_source_hint()

    # ============================================================ 启动 / 停止
    def _stream_specs(self) -> list[dict]:
        if self._multi:
            return [dict(c) for c in self.cfg.cameras]
        return [{"name": self._source_name(),
                 "source_type": self.cfg.source_type,
                 "source_value": self.cfg.source_value}]

    def _source_name(self) -> str:
        kind = self.cfg.source_type
        val = str(self.cfg.source_value or "")
        if kind == "camera":
            return f"摄像头 {val}"
        if kind == "rtsp":
            return "RTSP"
        return Path(val).name or kind

    def _start(self) -> None:
        if self._running:
            return
        if not self._multi:
            kind = self.cfg.source_type
            if not self._source_ready(kind):
                if not self._pick_source(kind):
                    return
                self._update_source_hint()

        self._teardown(keep_view=False)
        self._set_loading("当前状态：加载模型中…")

        for spec in self._stream_specs():
            s = Stream(spec, self.cfg, self._alarm_bridge.triggered.emit)
            self._streams[s.name] = s
        for s in self._streams.values():
            s.start()

        if self._streams and all(s.error for s in self._streams.values()):
            first = next(iter(self._streams.values()))
            self._teardown(keep_view=False)
            self._sync_state()
            QMessageBox.warning(self, "启动失败", f"{first.error}")
            return

        self._running = True
        self._model_label = next((s.detector.model_path.name
                                  for s in self._streams.values() if s.detector), "—")
        self._selected = self._selected or next(iter(self._streams), None)
        self._sync_state()
        self._update_channel_status()
        if self.card_control.hint_label:
            self.card_control.hint_label.setText(
                next((s.detector.backend_name + " · " + s.detector.device
                      for s in self._streams.values() if s.detector), "—"))
        if self.card_perf.hint_label:
            self.card_perf.hint_label.setText("实时")
        self.video.set_tags(fps=0, model=self._model_label)
        self.timer.start(self.cfg.ui_interval_ms)

    def _stop(self, silent: bool = False) -> None:
        self._teardown(keep_view=False)

    def _teardown(self, keep_view: bool = False) -> None:
        self.timer.stop()
        for s in self._streams.values():
            s.stop()
        self._streams.clear()
        self._running = False
        self._alarming = False
        self._toast.hide()
        self._sync_state()
        self._update_channel_status()
        if not keep_view:
            self._last = None
            self.video.reset_view()
            self._reset_metrics()

    # ============================================================ 每帧刷新
    def _tick(self) -> None:
        if not self._streams:
            return
        self._tick_no += 1
        sel = self._streams.get(self._selected)

        for name, s in self._streams.items():
            res = s.latest()
            if res is None:
                if s.finished() and s is sel and not self._multi:
                    self._teardown(keep_view=True)
                    self.status_text.setText("当前状态：已结束（处理完毕）")
                    self.status_text.setStyleSheet(
                        f"font-size:14px;font-weight:600;color:{theme.TEXT_DIM};")
                continue
            if self._multi and self._tick_no % 6 == 0:
                self.channels.set_thumbnail(name, self._frame_to_pixmap(res.frame))
            if s is sel:
                if self._last is None or res.timestamp != self._last.timestamp:
                    self._last = res
                    self.video.set_frame(res.frame)
                    self.metric_fps.set_value(f"{res.fps:.0f}" if res.fps > 0 else "--")
                    self.metric_ms.set_value(f"{res.infer_ms:.1f}")
                    self.metric_det.set_value(str(len(res.detections)))
                    self.video.set_tags(fps=res.fps, model=self._model_label,
                                        alarming=res.alarming)
                    if bool(res.alarming) != self._alarming:
                        self._alarming = bool(res.alarming)
                        self._sync_state()

        self._update_channel_status()

    def _frame_to_pixmap(self, frame_bgr) -> QPixmap:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg.copy())

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
        path = self._events_csv_path()
        if path is None or not path.exists():
            QMessageBox.information(self, "事件表", "还没有报警事件。")
            return
        subprocess.Popen(["explorer", "/select,", str(path)])

    def _events_csv_path(self) -> Path | None:
        if self._multi:
            s = self._streams.get(self._selected)
            if s is not None and s.alarm is not None:
                return s.alarm.events_csv
            from core.stream import safe_dir
            return self.cfg.resolve(f"captures/{safe_dir(self._selected or '')}/events.csv")
        return self.cfg.resolve(self.cfg.alarm.events_csv)

    def _open_settings(self) -> None:
        dlg = SettingsDialog(self.cfg, self)
        if dlg.exec():
            self._update_source_hint()
            if self.card_control.hint_label:
                self.card_control.hint_label.setText("已保存 · 下次开始生效")

    # ============================================================ 危险区域
    def _enter_roi(self) -> None:
        rect = self.video.image_rect()
        if rect is None:
            QMessageBox.information(self, "画区域", "请先开始检测加载画面，再绘制危险区域。")
            return
        self._roi_editor.begin(rect, existing=self.cfg.roi.points)

    def _on_roi_finished(self, points) -> None:
        self.cfg.roi.points = [[float(x), float(y)] for x, y in points]
        self.cfg.roi.enabled = True
        save_config(self.cfg)
        for s in self._streams.values():
            if s.pipeline is not None:
                s.pipeline.roi = RoiRegion.from_config(self.cfg.roi)
        self._toast.popup("危险区域已设置", f"{len(points)} 个顶点 · 只统计区域内目标")

    def _clear_roi(self) -> None:
        self.cfg.roi.points = []
        self.cfg.roi.enabled = False
        save_config(self.cfg)
        for s in self._streams.values():
            if s.pipeline is not None:
                s.pipeline.roi = RoiRegion.from_config(self.cfg.roi)
        self._toast.popup("危险区域已清除", "现在统计全画面目标")

    # ============================================================ 导出 Excel
    def _export_excel(self) -> None:
        csv_path = self._events_csv_path()
        if csv_path is None or not csv_path.exists():
            QMessageBox.information(self, "导出 Excel", "还没有报警事件，先跑一会儿再导出。")
            return
        default = f"报警记录_{time.strftime('%Y%m%d_%H%M')}.xlsx"
        path, _ = QFileDialog.getSaveFileName(self, "导出 Excel", default,
                                              "Excel 工作簿 (*.xlsx)")
        if not path:
            return
        try:
            out = export_events_excel(csv_path, path)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(out)))
        except Exception as exc:                            # noqa: BLE001
            QMessageBox.warning(self, "导出失败", f"{type(exc).__name__}: {exc}")

    # ============================================================ 报警
    def _on_alarm(self, payload) -> None:
        name, ev = payload
        prefix = f"[{name}] " if self._multi else ""
        label = f"{name} · {ev.time_str}" if self._multi else ev.time_str
        self.event_stream.add_event(label, ev.max_conf, ev.cls_name, ev.image_path)
        self._toast.popup(f"{prefix}检测到 {ev.count} 处未佩戴安全帽",
                          f"{ev.time_str}　最高置信度 {ev.max_conf:.2f}")
        self._place_toast()

    def _open_image(self, path: str) -> None:
        if path and Path(path).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _place_toast(self) -> None:
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
