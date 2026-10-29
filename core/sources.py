# core/sources.py
# 采集源：摄像头 / 视频文件 / 图片 / RTSP 流，统一成同一个 read() 接口。
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FrameSource:
    """采集源基类：open() / read() / release()。read() 返回 (是否成功, BGR 帧)。"""

    def __init__(self, value: str = ""):
        self.value = value
        self.cap: cv2.VideoCapture | None = None
        self._image: np.ndarray | None = None

    def open(self) -> "FrameSource":
        raise NotImplementedError

    def read(self) -> tuple[bool, np.ndarray | None]:
        raise NotImplementedError

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    @property
    def fps(self) -> float:
        return 0.0

    @property
    def frame_size(self) -> tuple[int, int]:
        return (0, 0)

    @property
    def is_live(self) -> bool:
        """是否为实时源（摄像头 / RTSP）。"""
        return False


class CameraSource(FrameSource):
    def __init__(self, index: str = "0", width: int = 1280, height: int = 720):
        super().__init__(index)
        self.index = int(index) if str(index).isdigit() else 0
        self.req_w, self.req_h = width, height

    def open(self) -> "CameraSource":
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.index)     # 退回默认后端再试
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {self.index}（可能被其他程序占用）")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.req_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.req_h)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)        # 只留最新一帧，降低延迟
        return self

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.cap is None:
            return False, None
        ok, frame = self.cap.read()
        return (ok, frame) if ok else (False, None)

    @property
    def fps(self) -> float:
        return float(self.cap.get(cv2.CAP_PROP_FPS)) if self.cap else 0.0

    @property
    def frame_size(self) -> tuple[int, int]:
        if not self.cap:
            return (0, 0)
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def is_live(self) -> bool:
        return True


class VideoSource(FrameSource):
    def __init__(self, path: str, loop: bool = True):
        super().__init__(path)
        self.loop = loop

    def open(self) -> "VideoSource":
        if not Path(self.value).exists():
            raise FileNotFoundError(f"视频文件不存在：{self.value}")
        self.cap = cv2.VideoCapture(self.value)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频：{self.value}")
        return self

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.cap is None:
            return False, None
        ok, frame = self.cap.read()
        if not ok and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)    # 播完回到开头
            ok, frame = self.cap.read()
        return (ok, frame) if ok else (False, None)

    @property
    def fps(self) -> float:
        return float(self.cap.get(cv2.CAP_PROP_FPS)) if self.cap else 0.0

    @property
    def frame_size(self) -> tuple[int, int]:
        if not self.cap:
            return (0, 0)
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


class RtspSource(VideoSource):
    """RTSP/IP 摄像头：和视频文件同一个接口，但不循环、只留最新帧。"""

    def open(self) -> "RtspSource":
        self.cap = cv2.VideoCapture(self.value, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.value)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法连接 RTSP 流：{self.value}")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return self

    @property
    def is_live(self) -> bool:
        return True


class ImageSource(FrameSource):
    """单张图片：read() 只成功一次，之后返回 False（表示"这一帧处理完了"）。"""

    def open(self) -> "ImageSource":
        if not Path(self.value).exists():
            raise FileNotFoundError(f"图片不存在：{self.value}")
        img = cv2.imread(self.value)
        if img is None:
            raise RuntimeError(f"图片无法解码：{self.value}")
        self._image = img
        self._done = False
        return self

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._image is None or self._done:
            return False, None
        self._done = True
        return True, self._image.copy()

    @property
    def frame_size(self) -> tuple[int, int]:
        return (self._image.shape[1], self._image.shape[0]) if self._image is not None else (0, 0)


def create_source(source_type: str, value: str,
                  camera_width: int = 1280, camera_height: int = 720,
                  video_loop: bool = True) -> FrameSource:
    """按配置创建采集源。"""
    t = (source_type or "").lower()
    if t == "camera":
        return CameraSource(value, camera_width, camera_height)
    if t == "video":
        return VideoSource(value, loop=video_loop)
    if t == "image":
        return ImageSource(value)
    if t == "rtsp":
        return RtspSource(value, loop=False)
    raise ValueError(f"未知的采集源类型：{source_type}（应为 camera/video/image/rtsp）")


def enumerate_cameras(max_index: int = 6) -> list[int]:
    """探测可用摄像头序号（打开后立即释放）。"""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            found.append(i)
        cap.release()
    return found
