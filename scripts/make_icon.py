#!/usr/bin/env python
# scripts/make_icon.py
# 生成程序图标：暗色圆角底 + 安全帽 + 检测取景角标，配色与界面设计令牌一致。
# 输出 assets/app.ico（多尺寸）与 assets/app_256.png（预览用）。
#
# 用法：python scripts/make_icon.py
from pathlib import Path

from PIL import Image, ImageDraw

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "assets"

# 与 ui/theme.py 保持一致
BG = (16, 20, 26, 255)          # #10141A 深底
BORDER = (44, 51, 63, 255)      # 描边
ACCENT = (30, 136, 229, 255)    # #1E88E5
ACCENT_LIGHT = (66, 165, 245, 255)
OK = (0, 230, 118, 255)         # #00E676

S = 1024                        # 先画大图再缩小，边缘更平滑


def build() -> Image.Image:
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # 圆角底板
    d.rounded_rectangle([6, 6, S - 7, S - 7], radius=230, fill=BG,
                        outline=BORDER, width=10)

    # 帽顶：半椭圆，底部正好接在帽檐上（pieslice 180~360 只画上半部分）
    dome = [int(S * 0.170), int(S * 0.318), int(S * 0.830), int(S * 0.958)]
    d.pieslice(dome, 180, 360, fill=ACCENT)

    # 帽檐：比帽顶宽、更扁，安全帽的标志性轮廓
    brim = [int(S * 0.085), int(S * 0.620), int(S * 0.915), int(S * 0.748)]
    d.rounded_rectangle(brim, radius=56, fill=ACCENT)
    # 帽檐下沿加深，做出厚度
    d.rounded_rectangle([brim[0], brim[3] - 40, brim[2], brim[3]], radius=26,
                        fill=(21, 101, 192, 255))

    # 帽顶高光：沿左上方内缘的一道弧
    hl = [int(S * 0.228), int(S * 0.358), int(S * 0.772), int(S * 0.933)]
    d.arc(hl, 200, 285, fill=ACCENT_LIGHT, width=26)

    return img


def preview_strip(big: Image.Image) -> Image.Image:
    """按实际显示尺寸拼一条预览，方便检查小尺寸下是否还认得出。"""
    sizes = [128, 64, 48, 32, 16]
    pad = 18
    width = sum(sizes) + pad * (len(sizes) + 1)
    height = max(sizes) + pad * 2
    strip = Image.new("RGB", (width, height), (32, 32, 32))   # 模拟深色任务栏
    x = pad
    for s in sizes:
        icon = big.resize((s, s), Image.LANCZOS)
        strip.paste(icon, (x, (height - s) // 2), icon)
        x += s + pad
    return strip


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    big = build()
    big.resize((256, 256), Image.LANCZOS).save(OUT_DIR / "app_256.png")
    preview_strip(big).save(OUT_DIR / "app_preview.png")

    ico = OUT_DIR / "app.ico"
    big.save(ico, format="ICO",
             sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
    print(f"已生成: {ico}  ({ico.stat().st_size/1024:.1f} KB)")
    print(f"预览图: {OUT_DIR / 'app_256.png'}")
    print(f"小尺寸预览: {OUT_DIR / 'app_preview.png'}")


if __name__ == "__main__":
    main()
