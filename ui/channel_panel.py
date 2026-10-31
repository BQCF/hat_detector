# ui/channel_panel.py
# 多路摄像头：左侧「频道列表」，每个频道 = 缩略图 + 名称 + 状态点 + 移除按钮。
# 点击频道切换大画面；所有频道在后台同时跑。
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QPushButton, QVBoxLayout, QWidget)

from . import theme
from .widgets import StatusDot

THUMB_W, THUMB_H = 150, 84


class ChannelItem(QWidget):
    remove_clicked = Signal()

    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(8, 8, 8, 8)
        row.setSpacing(10)

        self.thumb = QLabel()
        self.thumb.setFixedSize(THUMB_W, THUMB_H)
        self.thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb.setText("无信号")
        self.thumb.setStyleSheet(
            f"background:{theme.BG};border:1px solid {theme.BORDER};"
            f"border-radius:6px;color:{theme.TEXT_FAINT};font-size:11px;")
        row.addWidget(self.thumb)

        # 删除按钮悬浮在缩略图右上角：不占布局宽度（窄面板里会被裁掉），
        # 高对比底色保证在深色界面和任意画面上都看得见。
        self.rm = QPushButton("×", self.thumb)
        self.rm.setToolTip("删除此路")
        self.rm.setFixedSize(20, 20)
        self.rm.move(THUMB_W - 24, 3)
        self.rm.setCursor(Qt.CursorShape.PointingHandCursor)
        self.rm.setStyleSheet(
            "QPushButton{color:#FFFFFF;background:rgba(0,0,0,160);"
            "border:1px solid rgba(255,255,255,80);border-radius:10px;"
            "font-size:14px;font-weight:700;padding:0;}"
            "QPushButton:hover{background:rgba(220,60,50,230);}")
        self.rm.clicked.connect(self.remove_clicked.emit)
        self.rm.raise_()

        col = QVBoxLayout()
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(6)
        top = QHBoxLayout()
        top.setSpacing(6)
        self.dot = StatusDot()
        self.dot.set_state("stop")
        self.name_label = QLabel(name)
        self.name_label.setObjectName("ChannelName")
        self.name_label.setStyleSheet(f"color:{theme.TEXT};font-size:13px;font-weight:600;")
        top.addWidget(self.dot)
        top.addWidget(self.name_label)
        top.addStretch(1)
        col.addLayout(top)
        self.hint = QLabel("—")
        self.hint.setStyleSheet(f"color:{theme.TEXT_DIM};font-size:11px;")
        col.addWidget(self.hint)
        col.addStretch(1)
        row.addLayout(col, 1)

    def set_thumbnail(self, pixmap: QPixmap) -> None:
        self.thumb.setPixmap(pixmap.scaled(
            THUMB_W, THUMB_H, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def set_status(self, state: str, hint: str = "") -> None:
        self.dot.set_state(state)
        if hint:
            self.hint.setText(hint)


class ChannelPanel(QWidget):
    """左侧频道列表：选中发 selected(name)，点 × 发 remove_requested(name)。"""

    selected = Signal(str)
    remove_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(260)
        self.setMaximumWidth(360)
        self.setStyleSheet(f"background:{theme.CARD};border:1px solid {theme.BORDER};border-radius:8px;")

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        self.list = QListWidget()
        self.list.setFrameShape(QFrame.Shape.NoFrame)
        self.list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.setStyleSheet(
            "QListWidget{background:transparent;border:none;}"
            "QListWidget::item{border-radius:6px;margin:2px 0;padding:0;}"
            "QListWidget::item:selected{background:rgba(30,136,229,0.25);}")
        self.list.currentItemChanged.connect(self._on_current_changed)
        root.addWidget(self.list, 1)

        self._items: dict[str, tuple[QListWidgetItem, ChannelItem]] = {}

    def _on_current_changed(self, cur, prev) -> None:          # noqa: N802
        if cur is None:
            return
        for name, (item, _) in self._items.items():
            if item is cur:
                self.selected.emit(name)
                break

    # ---------------- 对外 ----------------
    def names(self) -> list[str]:
        return list(self._items.keys())

    def add_channel(self, name: str) -> None:
        if name in self._items:
            return
        w = ChannelItem(name)
        w.remove_clicked.connect(lambda n=name: self.remove_requested.emit(n))
        item = QListWidgetItem(self.list)
        item.setSizeHint(w.sizeHint())
        self.list.addItem(item)
        self.list.setItemWidget(item, w)
        self._items[name] = (item, w)
        if len(self._items) == 1:
            self.list.setCurrentItem(item)

    def remove_channel(self, name: str) -> None:
        if name not in self._items:
            return
        item, _ = self._items.pop(name)
        self.list.takeItem(self.list.row(item))

    def set_thumbnail(self, name: str, pixmap: QPixmap) -> None:
        if name in self._items:
            self._items[name][1].set_thumbnail(pixmap)

    def set_status(self, name: str, state: str, hint: str = "") -> None:
        if name in self._items:
            self._items[name][1].set_status(state, hint)

    def select(self, name: str) -> None:
        if name in self._items:
            self.list.setCurrentItem(self._items[name][0])
