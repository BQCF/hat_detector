# ui/main_window.py
# 主窗口骨架（重构版 · 现代科技风暗黑模式）
# 布局：左右 7:3 分栏；左侧纯黑视频卡片；右侧三张卡片（控制 / 性能 / 报警事件）。
# 第一步只搭骨架与视觉，按钮仅切换状态；第二步再把 core 的检测逻辑接进来。
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QMainWindow, QPushButton,
                               QSizePolicy, QVBoxLayout, QWidget)

from core.config import AppConfig

from . import theme
from .widgets import (Card, Divider, EventStream, MetricRow, SourceSegments,
                      StatusDot, VideoArea)


class MainWindow(QMainWindow):
    """新的主窗口骨架。对外仍保留 _start / _stop，桌面入口无需改动。"""

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

        self.setWindowTitle("安全帽佩戴检测系统")
        self.resize(1380, 830)
        self.setMinimumSize(1120, 700)
        self.setStyleSheet(theme.stylesheet())

        self._build_ui()
        self._sync_state()

    # ============================================================ 布局
    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)

        outer = QHBoxLayout(root)
        outer.setContentsMargins(theme.PAD, theme.PAD, theme.PAD, theme.PAD)
        outer.setSpacing(theme.GAP)

        # ---- 左：视频区（7） ----
        self.video = VideoArea()
        outer.addWidget(self.video, 7)

        # ---- 右：三张卡片（3） ----
        right = QWidget()
        right.setMinimumWidth(360)
        right.setMaximumWidth(460)
        col = QVBoxLayout(right)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(theme.GAP)
        col.addWidget(self._build_control_card(), 0)
        col.addWidget(self._build_perf_card(), 0)
        col.addWidget(self._build_event_card(), 1)
        outer.addWidget(right, 3)

    # ---------------------------------------------------- 卡片 A：控制与状态
    def _build_control_card(self) -> Card:
        card = Card("运行控制", "Ready")

        # 两个大圆角按钮
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

        # 状态指示灯
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

        # 数据源切换
        src_label = QLabel("选择数据源")
        src_label.setObjectName("MetricLabel")
        card.body.addWidget(src_label)

        self.sources = SourceSegments(self.SOURCES, current=self.cfg.source_type)
        card.body.addWidget(self.sources)

        self.source_hint = QLabel(self._source_hint_text())
        self.source_hint.setObjectName("CardHint")
        self.source_hint.setWordWrap(True)
        card.body.addWidget(self.source_hint)

        # 次要操作
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

        # 事件绑定（第一步只切状态）
        self.btn_start.clicked.connect(self._start)
        self.btn_stop.clicked.connect(lambda: self._stop())
        self.sources.changed.connect(self._on_source_changed)
        return card

    # ---------------------------------------------------- 卡片 B：实时性能
    def _build_perf_card(self) -> Card:
        card = Card("实时性能监测", "GPU / CPU")
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
        """把 _running 同步到所有控件外观。"""
        running = self._running
        self.status_dot.set_state("run" if running else "stop")
        self.status_text.setText("当前状态：运行中" if running else "当前状态：已停止")
        self.status_text.setStyleSheet(
            f"font-size:14px;font-weight:600;"
            f"color:{theme.OK if running else theme.TEXT_DIM};")
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        self.video.set_running(running)
        if running:
            self.video.set_tags(fps=None, model=self.cfg.model_path or "best.onnx")

    def _source_hint_text(self) -> str:
        kind = self.sources.current_kind() if hasattr(self, "sources") else self.cfg.source_type
        return {
            "camera": "本地摄像头（可多路编号，默认 0）",
            "video": "视频文件（mp4 / avi / mkv，可循环播放）",
            "image": "单张图片检测",
            "rtsp": "网络摄像头 rtsp:// 流地址",
        }.get(kind, "")

    # ============================================================ 对外接口
    def _start(self) -> None:
        """第二步会在这里启动 core.InferPipeline；第一步只切状态。"""
        self._running = True
        self._sync_state()

    def _stop(self, silent: bool = False) -> None:
        self._running = False
        self._sync_state()

    def _on_source_changed(self, kind: str) -> None:
        self.cfg.source_type = kind
        self.source_hint.setText(self._source_hint_text())

    # ============================================================ 收尾
    def showEvent(self, event) -> None:                       # noqa: N802
        super().showEvent(event)
        theme.apply_dark_titlebar(self)     # 深色标题栏，和暗色界面统一

    def closeEvent(self, event) -> None:                      # noqa: N802
        self._stop(silent=True)
        super().closeEvent(event)
