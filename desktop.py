#!/usr/bin/env python
# desktop.py
# 桌面程序入口：双击或 `python desktop.py` 启动。
#
# 用法：
#   python desktop.py                              # 用 configs/default.yaml 的设置启动
#   python desktop.py --config my.yaml             # 指定配置文件
#   python desktop.py --source 0                   # 直接开摄像头 0 并自动开始检测
#   python desktop.py --source demo.mp4            # 直接播视频文件
#   python desktop.py --source rtsp://192.168.1.10/stream
import argparse
import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from core.config import load_config
from ui.main_window import MainWindow

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def guess_source_type(value: str) -> str:
    """从命令行给的路径猜采集源类型。"""
    if value.startswith("rtsp://") or value.startswith("rtsps://"):
        return "rtsp"
    if value.isdigit():
        return "camera"
    ext = Path(value).suffix.lower()
    if ext in IMG_EXTS:
        return "image"
    return "video"


def main() -> int:
    parser = argparse.ArgumentParser(description="安全帽佩戴检测系统（桌面版）")
    parser.add_argument("--config", default="", help="配置文件路径（默认 configs/default.yaml）")
    parser.add_argument("--source", default="",
                        help="采集源：摄像头序号 / 视频 / 图片 / rtsp:// 地址；给了就自动开始检测")
    args = parser.parse_args()

    cfg = load_config(args.config or None)
    if args.source:
        cfg.source_type = guess_source_type(args.source)
        cfg.source_value = args.source

    app = QApplication(sys.argv)
    app.setApplicationName("安全帽佩戴检测系统")
    app.setFont(QFont("Microsoft YaHei", 10))
    app.setStyleSheet("""
        QMainWindow { background:#161b22; }
        QToolBar { background:#0d1117; border:none; padding:6px; spacing:6px; }
        QToolBar QToolButton { color:#e6edf3; padding:6px 12px; border-radius:6px; }
        QToolBar QToolButton:hover { background:#21262d; }
        QStatusBar { background:#0d1117; color:#8b949e; }
        QMessageBox { background:#161b22; }
    """)

    win = MainWindow(cfg)
    win.show()
    if args.source:
        # 给界面一帧时间画出来，再自动开始检测
        QTimer.singleShot(300, win._start)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
