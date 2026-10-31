# 临时：用 Win32 PrintWindow 抓窗口内容（不抢前台、不改 z 序，不打扰用户）
import ctypes
import ctypes.wintypes as wt
import sys
import time

from PIL import Image

TITLE = sys.argv[1]
OUT = sys.argv[2]
WAIT = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
user32.SetProcessDPIAware()


def find_window(deadline):
    while time.time() < deadline:
        found = []

        @ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)
        def _cb(hwnd, lparam):
            if user32.IsWindowVisible(hwnd):
                n = user32.GetWindowTextLengthW(hwnd)
                if n:
                    buf = ctypes.create_unicode_buffer(n + 1)
                    user32.GetWindowTextW(hwnd, buf, n + 1)
                    if TITLE == buf.value or TITLE in buf.value:
                        found.append(hwnd)
            return True

        user32.EnumWindows(_cb, 0)
        if found:
            return found[0]
        time.sleep(0.3)
    return None


class RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", ctypes.c_uint32), ("biWidth", ctypes.c_long),
                ("biHeight", ctypes.c_long), ("biPlanes", ctypes.c_uint16),
                ("biBitCount", ctypes.c_uint16), ("biCompression", ctypes.c_uint32),
                ("biSizeImage", ctypes.c_uint32), ("biXPelsPerMeter", ctypes.c_long),
                ("biYPelsPerMeter", ctypes.c_long), ("biClrUsed", ctypes.c_uint32),
                ("biClrImportant", ctypes.c_uint32)]


hwnd = find_window(time.time() + WAIT)
if not hwnd:
    print("没等到窗口:", TITLE)
    sys.exit(1)

rect = RECT()
user32.GetWindowRect(hwnd, ctypes.byref(rect))
w, h = rect.right - rect.left, rect.bottom - rect.top

hdc = user32.GetWindowDC(hwnd)
mem = gdi32.CreateCompatibleDC(hdc)
bmp = gdi32.CreateCompatibleBitmap(hdc, w, h)
gdi32.SelectObject(mem, bmp)

# PW_RENDERFULLCONTENT = 2，能抓到被遮挡/未激活的窗口
ok = user32.PrintWindow(hwnd, mem, 2)

bi = BITMAPINFOHEADER()
bi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
bi.biWidth = w
bi.biHeight = -h            # 负数 = 自上而下
bi.biPlanes = 1
bi.biBitCount = 32
bi.biCompression = 0
buf = ctypes.create_string_buffer(w * h * 4)
gdi32.GetDIBits(mem, bmp, 0, h, buf, ctypes.byref(bi), 0)

img = Image.frombuffer("RGB", (w, h), buf, "raw", "BGRX", 0, 1)
img.save(OUT)

gdi32.DeleteObject(bmp)
gdi32.DeleteDC(mem)
user32.ReleaseDC(hwnd, hdc)

# 全黑说明 PrintWindow 没画面，提示一下
extrema = img.convert("L").getextrema()
print(f"已保存: {OUT}  {w}x{h}  PrintWindow={ok}  亮度范围={extrema}")
