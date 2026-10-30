# ui/roi_editor.py
# 危险区域绘制层：盖在视频区上，左键加顶点、右键/回车闭合（>=3 点）、Esc 取消。
# 存的是归一化坐标（0~1），换分辨率也不用重画。
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget


class RoiEditor(QWidget):
    """透明覆盖层，负责收集多边形顶点并回传归一化坐标。"""

    finished = Signal(list)     # [[nx, ny], ...]
    cancelled = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pts: list[tuple[float, float]] = []
        self._rect = QRectF()
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.hide()

    def begin(self, image_rect: QRectF, existing=None) -> None:
        self._pts = [(float(x), float(y)) for x, y in (existing or [])]
        self._rect = image_rect
        p = self.parent()
        if p is not None:
            self.setGeometry(0, 0, p.width(), p.height())
        self.show()
        self.raise_()
        self.setFocus()
        self.update()

    # ---------------- 坐标映射 ----------------
    def _to_norm(self, pos):
        r = self._rect
        if r.width() <= 0 or r.height() <= 0:
            return None
        nx = (pos.x() - r.x()) / r.width()
        ny = (pos.y() - r.y()) / r.height()
        return max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny))

    # ---------------- 交互 ----------------
    def mousePressEvent(self, event) -> None:                 # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            p = self._to_norm(event.position())
            if p is not None:
                self._pts.append(p)
                self.update()
        elif event.button() == Qt.MouseButton.RightButton:
            self._finish()

    def keyPressEvent(self, event) -> None:                   # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self.cancel()
        elif event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._finish()

    def _finish(self) -> None:
        if len(self._pts) >= 3:
            self.finished.emit([[x, y] for x, y in self._pts])
        else:
            self.cancelled.emit()
        self.hide()

    def cancel(self) -> None:
        self.cancelled.emit()
        self.hide()

    # ---------------- 绘制 ----------------
    def paintEvent(self, event) -> None:                      # noqa: N802
        if self._rect.width() <= 0:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        danger = QColor("#FF5A3C")
        accent = QColor("#1E88E5")

        pts = [QPointF(self._rect.x() + x * self._rect.width(),
                       self._rect.y() + y * self._rect.height())
               for x, y in self._pts]

        # 闭合后的半透明填充
        if len(pts) >= 3:
            fill = QColor(danger)
            fill.setAlpha(40)
            painter.setBrush(fill)
            painter.setPen(QPen(danger, 2))
            painter.drawPolygon(pts)
        elif len(pts) >= 2:                                    # 未闭合只画边
            painter.setPen(QPen(danger, 2))
            for i in range(len(pts) - 1):
                painter.drawLine(pts[i], pts[i + 1])

        # 顶点
        for pt in pts:
            painter.setBrush(accent)
            painter.setPen(QPen(QColor("#0F1115"), 1))
            painter.drawEllipse(pt, 4, 4)

        # 提示
        painter.setPen(QColor("#FFFFFF"))
        hint = ("右键 / 回车闭合　·　Esc 取消" if len(self._pts) >= 3
                else "左键添加顶点（至少 3 个）　·　Esc 取消")
        painter.drawText(QRectF(14, 12, self.width() - 28, 24),
                         Qt.AlignmentFlag.AlignLeft, hint)
