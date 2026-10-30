# ui/theme.py
# 设计令牌（Design Tokens）与全局样式：现代科技风 · 暗黑模式。
# 换配色 / 圆角 / 间距只改这里，所有界面自动跟随。
from __future__ import annotations

from PySide6.QtGui import QFont

# ---------------- 字体 ----------------
FONT_FAMILY = "Microsoft YaHei"

# ---------------- 颜色 ----------------
BG = "#0F1115"            # 全局背景：深灰底色
CARD = "#171B22"          # 卡片背景
CARD_SOFT = "#1C2129"     # 卡片内悬浮/输入区
BORDER = "#252B36"        # 微弱边框

ACCENT = "#1E88E5"        # 强调色：深天蓝（按钮、选中态、进度）
ACCENT_HOVER = "#42A5F5"
ACCENT_PRESS = "#1565C0"

DANGER = "#FF5A3C"        # 危险色：橙红（未佩戴报警）
DANGER_HOVER = "#FF7A63"
DANGER_PRESS = "#E64A2E"

OK = "#00E676"            # 正常色：亮绿（已佩戴 / 运行中）

TEXT = "#E6EAF2"          # 主文字
TEXT_DIM = "#8B95A7"      # 次要文字
TEXT_FAINT = "#5A6473"    # 说明/占位文字
VIDEO_BG = "#000000"      # 视频卡片：纯黑无边框

# ---------------- 尺寸 ----------------
RADIUS = 8                # 卡片圆角 8px
GAP = 16                  # 卡片间距 15~20px
PAD = 18                  # 窗口外边距


def font(size: int = 13, bold: bool = False, family: str = FONT_FAMILY) -> QFont:
    """按设计规范取字体：标题 16-18 加粗，正文 12-14。"""
    f = QFont(family, size)
    f.setBold(bold)
    return f


def apply_dark_titlebar(window) -> None:
    """Windows 深色标题栏：不然顶上一条白条会破坏整体暗色观感。"""
    import sys
    if sys.platform != "win32":
        return
    try:
        import ctypes
        hwnd = int(window.winId())
        value = ctypes.c_int(1)
        for attr in (20, 19):     # DWMWA_USE_IMMERSIVE_DARK_MODE，新旧系统编号不同
            if ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    hwnd, attr, ctypes.byref(value), ctypes.sizeof(value)) == 0:
                break
    except Exception:
        pass                      # 非 Win11 或调用失败就保持系统默认，不影响功能


_QSS = """
/* ---------- 基础 ---------- */
QWidget {
    font-family: "@FONT@";
    font-size: 13px;
    color: @TEXT@;
}
QWidget#Root { background: @BG@; }

/* ---------- 卡片 ---------- */
QFrame#Card {
    background: @CARD@;
    border: 1px solid @BORDER@;
    border-radius: @RADIUS@px;
}
QFrame#VideoCard {
    background: @VIDEO_BG@;
    border: none;
    border-radius: @RADIUS@px;
}
QFrame#Divider {
    background: @BORDER@;
    border: none;
    max-height: 1px;
    min-height: 1px;
}

/* ---------- 文字层级 ---------- */
QLabel#CardTitle   { font-size: 17px; font-weight: 700; color: @TEXT@; }
QLabel#CardHint    { font-size: 12px; color: @TEXT_FAINT@; }
QLabel#MetricLabel { font-size: 13px; color: @TEXT_DIM@; }
QLabel#MetricValue { font-size: 28px; font-weight: 700; color: @TEXT@; }
QLabel#MetricUnit  { font-size: 12px; color: @TEXT_DIM@; }
QLabel#StatusText  { font-size: 14px; font-weight: 600; color: @TEXT_DIM@; }

/* 视频区左上角悬浮标签（半透明） */
QLabel#OverlayTag {
    font-size: 12px;
    color: #E6EAF2;
    background: rgba(0, 0, 0, 140);
    border: 1px solid rgba(255, 255, 255, 30);
    border-radius: 6px;
    padding: 4px 10px;
}

/* ---------- 主要按钮 ---------- */
QPushButton#PrimaryBtn {
    background: @ACCENT@;
    color: #FFFFFF;
    border: none;
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 14px;
    font-weight: 700;
}
QPushButton#PrimaryBtn:hover   { background: @ACCENT_HOVER@; }
QPushButton#PrimaryBtn:pressed { background: @ACCENT_PRESS@; }
QPushButton#PrimaryBtn:disabled { background: #2A3038; color: @TEXT_FAINT@; }

QPushButton#DangerBtn {
    background: @DANGER@;
    color: #FFFFFF;
    border: none;
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 14px;
    font-weight: 700;
}
QPushButton#DangerBtn:hover   { background: @DANGER_HOVER@; }
QPushButton#DangerBtn:pressed { background: @DANGER_PRESS@; }
QPushButton#DangerBtn:disabled { background: #2A3038; color: @TEXT_FAINT@; }

/* ---------- 数据源切换按钮组 ---------- */
QPushButton#SegBtn {
    background: transparent;
    color: @TEXT_DIM@;
    border: 1px solid @BORDER@;
    border-radius: 8px;
    padding: 8px 4px;
    font-size: 12px;
}
QPushButton#SegBtn:hover   { background: @CARD_SOFT@; color: @TEXT@; }
QPushButton#SegBtn:checked {
    background: @ACCENT@;
    color: #FFFFFF;
    border-color: @ACCENT@;
    font-weight: 600;
}

/* ---------- 次要按钮 ---------- */
QPushButton#GhostBtn {
    background: transparent;
    color: @TEXT_DIM@;
    border: none;
    border-radius: 6px;
    padding: 7px 12px;
    font-size: 12px;
}
QPushButton#GhostBtn:hover { background: @CARD_SOFT@; color: @TEXT@; }

/* ---------- 滚动区 ---------- */
QScrollArea { background: transparent; border: none; }
QScrollArea > QWidget > QWidget { background: transparent; }
QScrollBar:vertical { background: transparent; width: 8px; margin: 2px 0; }
QScrollBar::handle:vertical {
    background: #2C333D;
    border-radius: 4px;
    min-height: 28px;
}
QScrollBar::handle:vertical:hover { background: #3C4552; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }

/* ---------- 报警条目 ---------- */
QFrame#EventRow {
    background: @CARD_SOFT@;
    border: 1px solid @BORDER@;
    border-radius: 8px;
}
QLabel#EventTime { font-size: 12px; color: @TEXT@; }
QLabel#EventConf { font-size: 11px; color: @TEXT_DIM@; }
QLabel#BadgeDanger {
    font-size: 11px;
    font-weight: 700;
    color: #FFD9D2;
    background: rgba(255, 90, 60, 60);
    border: 1px solid rgba(255, 90, 60, 160);
    border-radius: 6px;
    padding: 2px 8px;
}
QLabel#BadgeOk {
    font-size: 11px;
    font-weight: 700;
    color: #CFFBE4;
    background: rgba(0, 230, 118, 45);
    border: 1px solid rgba(0, 230, 118, 130);
    border-radius: 6px;
    padding: 2px 8px;
}
QLabel#EmptyHint { font-size: 12px; color: @TEXT_FAINT@; }

/* ---------- 浮层提示（非阻塞弹窗） ---------- */
QFrame#Toast {
    background: rgba(38, 20, 17, 240);
    border: 1px solid @DANGER@;
    border-radius: 10px;
}
QLabel#ToastTitle { font-size: 13px; font-weight: 700; color: #FFD9D2; }
QLabel#ToastSub   { font-size: 11px; color: @TEXT_DIM@; }
QPushButton#ToastClose {
    background: transparent;
    color: @TEXT_DIM@;
    border: none;
    border-radius: 4px;
    font-size: 14px;
    padding: 0;
}
QPushButton#ToastClose:hover { background: rgba(255, 255, 255, 25); color: @TEXT@; }
"""


def stylesheet() -> str:
    """把设计令牌塞进 QSS（按 token 长度倒序替换，避免 ACCENT 把 ACCENT_HOVER 截断）。"""
    tokens = {
        "FONT": FONT_FAMILY,
        "BG": BG, "CARD": CARD, "CARD_SOFT": CARD_SOFT, "BORDER": BORDER,
        "ACCENT_HOVER": ACCENT_HOVER, "ACCENT_PRESS": ACCENT_PRESS, "ACCENT": ACCENT,
        "DANGER_HOVER": DANGER_HOVER, "DANGER_PRESS": DANGER_PRESS, "DANGER": DANGER,
        "OK": OK,
        "TEXT_DIM": TEXT_DIM, "TEXT_FAINT": TEXT_FAINT, "TEXT": TEXT,
        "VIDEO_BG": VIDEO_BG, "RADIUS": str(RADIUS),
    }
    qss = _QSS
    for k in sorted(tokens, key=len, reverse=True):
        qss = qss.replace(f"@{k}@", tokens[k])
    return qss
