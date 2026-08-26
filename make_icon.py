# -*- coding: utf-8 -*-
"""生成可爱卡通小猫图标 cat_icon.ico（多尺寸，供窗口与 exe 使用）。"""
import os
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "models", "cat_icon.ico")

INK = "#4a2f1a"
ORANGE = "#f5c56b"
EAR = "#e8a33d"
PINK = "#f2b8c0"
BLUSH = "#f4b6c0"


def draw_cat(size):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    s = size / 256.0
    def p(*flat):
        return [(int(flat[i] * s), int(flat[i + 1] * s)) for i in range(0, len(flat), 2)]
    def L(*args):  # 缩放后的矩形
        return [int(v * s) for v in args]

    # 耳朵
    d.polygon(p(50, 50, 60, 12, 100, 40), fill=EAR, outline=INK, width=max(1, int(6 * s)))
    d.polygon(p(206, 50, 196, 12, 156, 40), fill=EAR, outline=INK, width=max(1, int(6 * s)))
    d.polygon(p(62, 44, 68, 24, 90, 40), fill=PINK)
    d.polygon(p(194, 44, 188, 24, 166, 40), fill=PINK)
    # 头
    d.ellipse(L(36, 32, 220, 216), fill=ORANGE, outline=INK, width=max(1, int(6 * s)))
    # 眼睛 + 高光
    d.ellipse(L(70, 76, 104, 110), fill="#2a2118")
    d.ellipse(L(152, 76, 186, 110), fill="#2a2118")
    d.ellipse(L(84, 82, 96, 94), fill="#ffffff")
    d.ellipse(L(160, 82, 172, 94), fill="#ffffff")
    # 鼻子 + 嘴
    d.polygon(p(122, 116, 134, 116, 128, 126), fill="#e8847f")
    d.arc(L(106, 118, 124, 138), start=200, end=340, fill=INK, width=max(1, int(4 * s)))
    d.arc(L(132, 118, 150, 138), start=200, end=340, fill=INK, width=max(1, int(4 * s)))
    # 腮红
    d.ellipse(L(60, 112, 84, 128), fill=BLUSH)
    d.ellipse(L(172, 112, 196, 128), fill=BLUSH)
    # 胡须
    for y in (100, 112, 124):
        d.line(L(28, y, 58, y), fill=INK, width=max(1, int(4 * s)))
        d.line(L(228, y, 198, y), fill=INK, width=max(1, int(4 * s)))
    return img


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    icon = draw_cat(256)
    icon.save(OUT, format="ICO", sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print("已生成:", OUT)
