# ui/widgets.py
# 可复用界面组件：卡片、状态灯、指标行、数据源切换、视频区、报警条目。
# 这一步只管"长什么样"，不接业务逻辑（第二步再接）。
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (QButtonGroup, QFrame, QHBoxLayout, QLabel,
                               QPushButton, QScrollArea, QSizePolicy,
                               QStackedLayout, QVBoxLayout, QWidget)

from . import theme
from .video_view import VideoView


# ---------------------------------------------------------------- 基础容器
class Card(QFrame):
    """标准卡片：圆角 8px + 微弱边框 + 内边距。标题可选。"""

    def __init__(self, title: str = "", hint: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(16, 14, 16, 16)
        self.body.setSpacing(12)

        self.title_label: QLabel | None = None
        self.hint_label: QLabel | None = None     # 标题右侧的小字，外部可改
        if title:
            head = QHBoxLayout()
            head.setContentsMargins(0, 0, 0, 0)
            head.setSpacing(8)
            lab = QLabel(title)
            lab.setObjectName("CardTitle")
            self.title_label = lab
            head.addWidget(lab)
            if hint:
                h = QLabel(hint)
                h.setObjectName("CardHint")
                self.hint_label = h
                head.addStretch(1)
                head.addWidget(h)
            self.body.addLayout(head)


class Divider(QFrame):
    """1px 分隔线。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Divider")
        self.setFixedHeight(1)


# ---------------------------------------------------------------- 状态灯
class StatusDot(QWidget):
    """会发光的状态小圆点：绿=运行中，红=已停止/报警。"""

    COLORS = {"run": theme.OK, "stop": theme.DANGER, "alarm": theme.DANGER}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(18, 18)
        self._color = QColor(self.COLORS["stop"])

    def set_state(self, state: str) -> None:
        self._color = QColor(self.COLORS.get(state, self.COLORS["stop"]))
        self.update()

    def paintEvent(self, event) -> None:                      # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx, cy = self.width() / 2, self.height() / 2

        # 由外向内三层递减透明度，形成柔和光晕
        for scale, alpha in ((2.0, 16), (1.5, 28), (1.05, 45)):
            glow = QColor(self._color)
            glow.setAlpha(alpha)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(glow)
            painter.drawEllipse(QPointF(cx, cy), 7.0 * scale / 2, 7.0 * scale / 2)

        painter.setBrush(self._color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(cx, cy), 3.6, 3.6)


# ---------------------------------------------------------------- 指标行
class MetricRow(QWidget):
    """大号数字 + 小字单位，左侧配标签。"""

    def __init__(self, label: str, value: str = "—", unit: str = "", parent=None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 4, 0, 4)
        row.setSpacing(6)

        self._label = QLabel(label)
        self._label.setObjectName("MetricLabel")

        self._value = QLabel(value)
        self._value.setObjectName("MetricValue")

        self._unit = QLabel(unit)
        self._unit.setObjectName("MetricUnit")

        row.addWidget(self._label)
        row.addStretch(1)
        row.addWidget(self._value, 0, Qt.AlignmentFlag.AlignBottom)
        row.addWidget(self._unit, 0, Qt.AlignmentFlag.AlignBottom)

    def set_value(self, value: str, unit: str | None = None) -> None:
        self._value.setText(value)
        if unit is not None:
            self._unit.setText(unit)


# ---------------------------------------------------------------- 数据源切换
class SourceSegments(QWidget):
    """数据源切换按钮组（摄像头 / 视频文件 / 图片 / RTSP）。"""

    changed = Signal(str)

    def __init__(self, options: list[tuple[str, str]], current: str = "", parent=None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: dict[str, QPushButton] = {}

        for kind, text in options:
            btn = QPushButton(text)
            btn.setObjectName("SegBtn")
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.clicked.connect(lambda _=False, k=kind: self.changed.emit(k))
            self._group.addButton(btn)
            row.addWidget(btn)
            self._buttons[kind] = btn

        if current in self._buttons:
            self._buttons[current].setChecked(True)
        elif self._buttons:
            next(iter(self._buttons.values())).setChecked(True)

    def current_kind(self) -> str:
        for kind, btn in self._buttons.items():
            if btn.isChecked():
                return kind
        return ""

    def set_current(self, kind: str) -> None:
        if kind in self._buttons:
            self._buttons[kind].setChecked(True)


# ---------------------------------------------------------------- 视频区
class _Placeholder(QWidget):
    """无画面时的占位：半透明现代摄像头图标 + 提示文字。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._subtitle = "在右侧选择数据源后点击「开始检测」"

    def set_subtitle(self, text: str) -> None:
        self._subtitle = text
        self.update()

    def paintEvent(self, event) -> None:                      # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2 - 18

        accent = QColor(theme.ACCENT)
        soft = QColor(accent)
        soft.setAlpha(20)
        stroke = QColor(accent)
        stroke.setAlpha(120)

        # 取景四角（AI 扫描框）
        bw, bh = 118, 88
        rect = QRectF(cx - bw / 2, cy - bh / 2, bw, bh)
        pen = QPen(stroke)
        pen.setWidth(2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        seg = 16
        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            x = rect.left() if dx < 0 else rect.right()
            y = rect.top() if dy < 0 else rect.bottom()
            painter.drawLine(QPointF(x, y), QPointF(x - dx * seg, y))
            painter.drawLine(QPointF(x, y), QPointF(x, y - dy * seg))

        # 摄像头机身
        body = QRectF(cx - 48, cy - 26, 96, 62)
        painter.setBrush(soft)
        painter.setPen(QPen(stroke, 2))
        painter.drawRoundedRect(body, 12, 12)

        # 顶部取景凸起
        painter.drawRoundedRect(QRectF(cx - 16, cy - 38, 32, 14), 6, 6)

        # 镜头
        painter.setBrush(QColor(accent.red(), accent.green(), accent.blue(), 40))
        painter.drawEllipse(QPointF(cx, cy + 5), 17, 17)
        painter.drawEllipse(QPointF(cx, cy + 5), 7, 7)

        # 文字提示
        painter.setPen(QColor(theme.TEXT_DIM))
        hint_font = QFont(theme.FONT_FAMILY, 12)
        hint_font.setBold(True)
        painter.setFont(hint_font)
        painter.drawText(QRectF(0, cy + 76, w, 24),
                         Qt.AlignmentFlag.AlignHCenter, "等待视频信号")
        painter.setPen(QColor(theme.TEXT_FAINT))
        painter.setFont(QFont(theme.FONT_FAMILY, 10))
        painter.drawText(QRectF(0, cy + 102, w, 22),
                         Qt.AlignmentFlag.AlignHCenter,
                         self._subtitle)


class VideoArea(QFrame):
    """左侧主显示区：纯黑无边框卡片，内含占位层与视频层，左上角悬浮状态标签。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("VideoCard")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setMinimumSize(520, 360)

        self._stack = QStackedLayout(self)
        self._stack.setContentsMargins(0, 0, 0, 0)
        self._placeholder = _Placeholder()
        self.view = VideoView()
        self._stack.addWidget(self._placeholder)
        self._stack.addWidget(self.view)
        self._stack.setCurrentWidget(self._placeholder)

        # 左上角悬浮标签组（半透明）
        self._tags = QWidget(self)
        tag_row = QHBoxLayout(self._tags)
        tag_row.setContentsMargins(0, 0, 0, 0)
        tag_row.setSpacing(8)
        self.tag_state = QLabel("● 运行中")
        self.tag_fps = QLabel("FPS —")
        self.tag_model = QLabel("模型 —")
        for t in (self.tag_state, self.tag_fps, self.tag_model):
            t.setObjectName("OverlayTag")
            tag_row.addWidget(t)
        self._tags.hide()

    # ---- 对外 ----
    def set_running(self, running: bool) -> None:
        self._tags.setVisible(running)
        self._tags.raise_()                # 切到视频层后要重新提到最上层
        self._placeholder.set_subtitle(
            "正在连接数据源…" if running else "在右侧选择数据源后点击「开始检测」")

    def set_tags(self, fps: float | None = None,
                 model: str | None = None,
                 alarming: bool = False) -> None:
        if fps is not None:
            self.tag_fps.setText(f"FPS {fps:.0f}" if fps else "FPS —")
        if model is not None:
            self.tag_model.setText(f"模型 {model}")
        self.tag_state.setText("● 报警中" if alarming else "● 运行中")
        color = theme.DANGER if alarming else theme.OK
        self.tag_state.setStyleSheet(
            f"color:{color};background:rgba(0,0,0,140);"
            f"border:1px solid rgba(255,255,255,30);border-radius:6px;padding:4px 10px;")
        self._tags.adjustSize()          # 文案变了要重新量尺寸，否则会被截断
        self._tags.raise_()

    def set_frame(self, frame_bgr) -> None:
        """第二步接入：把 BGR 帧喂进来就切到视频层。"""
        self.view.set_frame(frame_bgr)
        self._stack.setCurrentWidget(self.view)
        self._tags.raise_()              # 视频层会盖住悬浮标签，必须重新置顶

    def reset_view(self) -> None:
        self.view.clear_frame()
        self._stack.setCurrentWidget(self._placeholder)
        self._tags.raise_()

    # ---- 布局 ----
    def resizeEvent(self, event) -> None:                     # noqa: N802
        super().resizeEvent(event)
        self._tags.adjustSize()
        self._tags.move(16, 16)


# ---------------------------------------------------------------- 报警条目
class EventRow(QFrame):
    """一条报警记录：缩略图 + 时间戳 + 置信度 + 状态徽章。未佩戴用橙红。"""

    def __init__(self, time_text: str, conf: float, kind: str = "head",
                 parent=None):
        super().__init__(parent)
        self.setObjectName("EventRow")
        row = QHBoxLayout(self)
        row.setContentsMargins(8, 8, 10, 8)
        row.setSpacing(10)

        # 截图缩略图占位（第三步换成真实抓拍）
        thumb = QLabel()
        thumb.setFixedSize(64, 44)
        thumb.setStyleSheet(
            f"background:{theme.BG};border:1px solid {theme.BORDER};"
            f"border-radius:6px;color:{theme.TEXT_FAINT};font-size:10px;")
        thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thumb.setText("截图")
        row.addWidget(thumb)

        col = QVBoxLayout()
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(4)
        t = QLabel(time_text)
        t.setObjectName("EventTime")
        c = QLabel(f"置信度 {conf:.2f}")
        c.setObjectName("EventConf")
        col.addWidget(t)
        col.addWidget(c)
        row.addLayout(col, 1)

        is_bad = kind == "head"
        badge = QLabel("未佩戴" if is_bad else "已佩戴")
        badge.setObjectName("BadgeDanger" if is_bad else "BadgeOk")
        row.addWidget(badge, 0, Qt.AlignmentFlag.AlignVCenter)


class EventStream(QScrollArea):
    """报警事件流：竖向滚动列表，新事件插在最上面。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._host = QWidget()
        self._list = QVBoxLayout(self._host)
        self._list.setContentsMargins(0, 0, 6, 0)
        self._list.setSpacing(8)

        self._empty = QLabel("暂无报警事件")
        self._empty.setObjectName("EmptyHint")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty.setMinimumHeight(90)
        self._list.addWidget(self._empty)
        self._list.addStretch(1)

        self.setWidget(self._host)

    def add_event(self, time_text: str, conf: float, kind: str = "head") -> EventRow:
        self._empty.hide()
        row = EventRow(time_text, conf, kind)
        # 插在最后那个 stretch 之前 → 新的在上
        self._list.insertWidget(1, row)
        return row

    def clear(self) -> None:
        while self._list.count() > 2:
            item = self._list.takeAt(1)
            if item.widget():
                item.widget().deleteLater()
        self._empty.show()
