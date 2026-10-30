# ui/video_view.py
# 画面控件：把 BGR 帧画到窗口上，等比缩放、不变形。
from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel


class VideoView(QLabel):
    """显示视频帧的 QLabel：自动等比缩放，窗口怎么拉都不变形。"""

    def __init__(self, placeholder: str = "无画面", parent=None):
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._placeholder = placeholder
        self.setMinimumSize(480, 360)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(
            "background:#000000; color:#5A6473; border:none;"
            "border-radius:8px; font-size:15px;")
        self.setText(self._placeholder)

    def set_frame(self, frame_bgr: np.ndarray) -> None:
        """接收一帧 BGR 图像并显示。"""
        if frame_bgr is None or frame_bgr.size == 0:
            return
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg.copy())   # copy() 脱离 numpy 缓冲
        self._apply()

    def clear_frame(self) -> None:
        self._pixmap = None
        self.setPixmap(QPixmap())
        self.setText(self._placeholder)

    def _apply(self) -> None:
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        self.setPixmap(scaled)

    def image_rect(self):
        """当前画面在控件里实际占据的矩形（等比缩放后居中）。没有画面返回 None。"""
        if self._pixmap is None:
            return None
        iw, ih = self._pixmap.width(), self._pixmap.height()
        lw, lh = self.width(), self.height()
        if iw <= 0 or ih <= 0 or lw <= 0 or lh <= 0:
            return None
        scale = min(lw / iw, lh / ih)
        dw, dh = iw * scale, ih * scale
        from PySide6.QtCore import QRectF
        return QRectF((lw - dw) / 2, (lh - dh) / 2, dw, dh)

    def resizeEvent(self, event) -> None:                 # noqa: N802
        super().resizeEvent(event)
        self._apply()
