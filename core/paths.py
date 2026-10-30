# core/paths.py
# 路径解析：区分「只读资源」和「可写数据」，开发环境与打包后都正确。
#
# 开发环境（python desktop.py）：两者都是项目根目录 D:\projects\hat_detector
# 打包后（dist/hat_detector/hat_detector.exe）：
#   resource_dir() = dist/hat_detector/_internal   ← 打包进去的模型等只读资源
#   user_dir()     = dist/hat_detector             ← 抓拍、配置等会写的东西，放 exe 旁边
from __future__ import annotations

import sys
from pathlib import Path


def is_frozen() -> bool:
    """是否运行在 PyInstaller 打包后的环境。"""
    return bool(getattr(sys, "frozen", False))


def resource_dir() -> Path:
    """只读资源目录（模型、内置数据）。打包后是 _internal。"""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent.parent


def user_dir() -> Path:
    """可写目录（配置、抓拍、事件表）。打包后放 exe 旁边，重装程序不会丢。"""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent
