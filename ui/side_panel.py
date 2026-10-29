# ui/side_panel.py
# 右侧信息栏：运行状态、性能、各类计数、报警事件列表。
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QGroupBox, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QSizePolicy, QVBoxLayout, QWidget)

STYLE_STATE_RUN = "color:#3fb950; font-size:17px; font-weight:700;"
STYLE_STATE_STOP = "color:#8b949e; font-size:17px; font-weight:700;"
STYLE_STATE_ALARM = "color:#ff4d4f; font-size:17px; font-weight:700;"


class SidePanel(QWidget):
    """右侧信息栏。双击事件条目时发出 open_path 信号。"""

    open_path = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(260)
        self.setMaximumWidth(380)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # ---- 状态 ----
        g1 = QGroupBox("运行状态")
        v1 = QVBoxLayout(g1)
        self.state_label = QLabel("🔴 已停止")
        self.state_label.setStyleSheet(STYLE_STATE_STOP)
        self.source_label = QLabel("采集源：—")
        self.source_label.setStyleSheet("color:#8b949e;")
        self.source_label.setWordWrap(True)
        self.model_label = QLabel("模型：—")
        self.model_label.setStyleSheet("color:#8b949e;")
        self.model_label.setWordWrap(True)
        for w in (self.state_label, self.source_label, self.model_label):
            v1.addWidget(w)
        root.addWidget(g1)

        # ---- 性能 ----
        g2 = QGroupBox("性能")
        v2 = QVBoxLayout(g2)
        self.fps_label = self._metric(v2, "帧率", "— FPS")
        self.ms_label = self._metric(v2, "单帧耗时", "— ms")
        self.det_label = self._metric(v2, "本帧目标", "— 个")

        # ---- 分类计数 ----
        self.counts_box = QGroupBox("检测计数")
        self.counts_layout = QVBoxLayout(self.counts_box)
        self.count_labels: dict[str, QLabel] = {}
        root.addWidget(g2)
        root.addWidget(self.counts_box)

        # ---- 事件列表 ----
        g4 = QGroupBox("报警事件（双击查看抓拍）")
        v4 = QVBoxLayout(g4)
        self.event_list = QListWidget()
        self.event_list.setStyleSheet(
            "QListWidget{background:#0d1117;border:1px solid #30363d;border-radius:6px;}"
            "QListWidget::item{padding:6px;}")
        self.event_list.itemDoubleClicked.connect(self._on_item_double_clicked)
        v4.addWidget(self.event_list)
        root.addWidget(g4, stretch=1)

        self.setStyleSheet("""
            QGroupBox { color:#8b949e; border:1px solid #30363d; border-radius:8px;
                        margin-top:10px; padding-top:8px; font-size:13px; }
            QGroupBox::title { subcontrol-origin: margin; left:10px; padding:0 4px; }
            QLabel { color:#e6edf3; font-size:13px; }
        """)

    # ---------------- 公共接口 ----------------
    def _metric(self, layout: QVBoxLayout, name: str, value: str) -> QLabel:
        row = QHBoxLayout()
        a = QLabel(name)
        a.setStyleSheet("color:#8b949e;")
        b = QLabel(value)
        b.setStyleSheet("color:#e6edf3; font-weight:600;")
        b.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignRight)
        row.addWidget(a)
        row.addWidget(b, stretch=1)
        layout.addLayout(row)
        return b

    def set_state(self, running: bool, alarming: bool = False) -> None:
        if alarming:
            self.state_label.setText("⚠ 报警中：检测到未佩戴安全帽")
            self.state_label.setStyleSheet(STYLE_STATE_ALARM)
        elif running:
            self.state_label.setText("🟢 检测运行中")
            self.state_label.setStyleSheet(STYLE_STATE_RUN)
        else:
            self.state_label.setText("🔴 已停止")
            self.state_label.setStyleSheet(STYLE_STATE_STOP)

    def set_source(self, text: str) -> None:
        self.source_label.setText(f"采集源：{text}")

    def set_model(self, text: str) -> None:
        self.model_label.setText(f"模型：{text}")

    def set_perf(self, fps: float, ms: float, n_det: int) -> None:
        self.fps_label.setText(f"{fps:.1f} FPS" if fps else "— FPS")
        self.ms_label.setText(f"{ms:.1f} ms" if ms else "— ms")
        self.det_label.setText(f"{n_det} 个")

    def set_counts(self, counts: dict[str, int]) -> None:
        """按类别刷新计数（类别由模型决定，控件动态创建）。"""
        for name, n in counts.items():
            if name not in self.count_labels:
                lbl = self._metric(self.counts_layout, name, str(n))
                self.count_labels[name] = lbl
            else:
                self.count_labels[name].setText(str(n))

    def add_event(self, text: str, path: str = "") -> None:
        item = QListWidgetItem(text)
        if path:
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setToolTip(f"抓拍：{path}")
        self.event_list.insertItem(0, item)      # 最新的排最上面

    # ---------------- 交互 ----------------
    def _on_item_double_clicked(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole) or ""
        if path and Path(path).exists():
            self.open_path.emit(path)
