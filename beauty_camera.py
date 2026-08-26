# -*- coding: utf-8 -*-
"""
美颜相机 · 桌面小程序
功能：
  - 实时摄像头预览（镜像，像照镜子）
  - 人脸识别（YuNet 深度学习检测器）
  - 美颜：美白 / 瘦脸 / 大眼（程度可调）
  - 画质增强：清晰度 / 锐化 / 去模糊 / 亮度 / 对比度 / 饱和度
  - 摄影常用滤镜（精选 20 种风格）
  - 相机专业参数：快门(曝光) / ISO(增益) / 白平衡 / 聚焦，支持 手动 / 自动
  - 一键拍照，保存到 exe 同级的「照片」文件夹
"""

import os
import sys
import time
import threading

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

# ----------------------------------------------------------------------------
# 资源路径（兼容 PyInstaller 单文件打包）
# ----------------------------------------------------------------------------
def resource_path(rel):
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel)


def get_save_dir():
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    d = os.path.join(base, "照片")
    os.makedirs(d, exist_ok=True)
    return d


def imread_unicode(path):
    """读取图片（支持含中文的路径，cv2.imread 在 Windows 上不支持中文路径）。"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_unicode(path, img, quality=95):
    """写入图片（支持含中文的路径，cv2.imwrite 会静默失败）。"""
    try:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return False
        with open(path, "wb") as f:
            f.write(buf.tobytes())
        return True
    except Exception:
        return False


MODEL_PATH = resource_path(os.path.join("models", "face_detection_yunet_2023mar.onnx"))


def build_pixel_cat(width=96):
    """生成滤镜预览背景：一只像素小猫（低分辨率绘制 + 最近邻放大，自带马赛克颗粒感）。"""
    pal = {  # RGB
        ".": (245, 234, 208),  # 奶油底
        "k": (58, 42, 26),     # 深棕描边
        "o": (240, 162, 74),   # 橘色皮毛
        "w": (251, 243, 230),  # 白色口鼻
        "p": (242, 184, 192),  # 粉色内耳 / 鼻子
        "e": (42, 33, 24),     # 眼睛
    }
    left = [
        "...kk...",
        "..kook..",
        ".kooopk.",
        ".koooooo",
        ".kooeooo",
        ".kooeooo",
        ".koooooo",
        ".kooowww",
        ".kooowwp",
        ".kooowww",
        ".koooooo",
        "..kooooo",
        "...koooo",
    ]
    rows = [r + r[::-1] for r in left]  # 左右镜像，保证 16 宽对称
    h, w = len(rows), len(rows[0])      # 13 x 16
    img = np.zeros((h, w, 3), np.uint8)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            img[y, x] = pal[ch]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    th = max(1, int(width * h / w))
    return cv2.resize(img, (width, th), interpolation=cv2.INTER_NEAREST)

# ---- 复古主题配色 ----
WOOD_DARK = "#5a3a22"    # 深木板
WOOD_MID = "#8a5a2f"     # 中木板
WOOD_LIGHT = "#b98a5e"   # 浅木板
CREAM = "#f3e5c3"        # 奶油底
PAPER = "#efe0bb"        # 羊皮纸
INK = "#4a2f1a"          # 深棕文字
GOLD = "#e0b052"         # 复古金
FILM_DARK = "#2a2118"    # 胶片深色

# ---- 主题（白天复古 / 深夜）----
PALETTE = {
    "day": {
        "bg": WOOD_DARK, "panel": PAPER, "frame": CREAM, "fg": INK,
        "accent": GOLD, "trough": WOOD_LIGHT, "btn": GOLD, "btn2": WOOD_LIGHT,
        "tab": WOOD_LIGHT, "tab_sel": GOLD, "tab_active": WOOD_MID,
        "film": FILM_DARK, "sub": WOOD_LIGHT, "desc": "#6b4a2f",
        "spark": GOLD, "sprocket": CREAM, "video": "#000000",
    },
    "night": {
        "bg": "#0e1420", "panel": "#161e30", "frame": "#121a28", "fg": "#c9d4e8",
        "accent": "#8faee0", "trough": "#2a3550", "btn": "#3a4a6a", "btn2": "#2a3550",
        "tab": "#22304a", "tab_sel": "#5a7ab0", "tab_active": "#33507a",
        "film": "#0a0e18", "sub": "#6b7da8", "desc": "#8ba0c8",
        "spark": "#dbe6ff", "sprocket": "#3d4a68", "video": "#000000",
    },
}

# ----------------------------------------------------------------------------
# 滤镜库：精选摄影常用滤镜
#   参数：temp 色温 / tint 色调 / hue 色相偏移 / sat 饱和度 / gamma 曲线 / contrast 对比度 / vignette 暗角
# ----------------------------------------------------------------------------
def build_filters():
    return [
        {"name": "原图", "identity": True,
         "desc": "不添加任何滤镜，展示摄像头原始画面。"},
        {"name": "自然", "temp": 0, "tint": 0, "hue": 0, "sat": 1.05, "gamma": 1.00, "contrast": 1.02, "vignette": 0.00,
         "desc": "中性微调，真实自然，接近原片质感。"},
        {"name": "鲜艳", "temp": 5, "tint": 0, "hue": 0, "sat": 1.35, "gamma": 1.05, "contrast": 1.10, "vignette": 0.00,
         "desc": "提升饱和度与对比度，色彩更鲜活明快。"},
        {"name": "黑白", "temp": 0, "tint": 0, "hue": 0, "sat": 0.00, "gamma": 1.05, "contrast": 1.20, "vignette": 0.15,
         "desc": "经典黑白，去除色彩，强调明暗与质感。"},
        {"name": "人像", "temp": 15, "tint": 8, "hue": 0, "sat": 1.10, "gamma": 0.95, "contrast": 0.95, "vignette": 0.10,
         "desc": "暖调柔肤，肤色红润通透，适合拍人。"},
        {"name": "日系", "temp": -10, "tint": 0, "hue": 0, "sat": 0.85, "gamma": 0.88, "contrast": 0.88, "vignette": 0.05,
         "desc": "明亮低饱和，通透清新，文艺日系风。"},
        {"name": "清新", "temp": -15, "tint": 0, "hue": 0, "sat": 1.05, "gamma": 0.92, "contrast": 0.95, "vignette": 0.00,
         "desc": "明亮冷调，清爽干净，适合风景与人像。"},
        {"name": "胶片", "temp": 10, "tint": 0, "hue": 0, "sat": 0.95, "gamma": 1.05, "contrast": 1.10, "vignette": 0.35,
         "desc": "模拟胶片颗粒与暗角，复古耐看。"},
        {"name": "复古", "temp": 25, "tint": -5, "hue": 0, "sat": 0.75, "gamma": 0.90, "contrast": 0.90, "vignette": 0.40,
         "desc": "暖黄褪色，怀旧复古质感。"},
        {"name": "褪色", "temp": 5, "tint": 0, "hue": 0, "sat": 0.60, "gamma": 0.85, "contrast": 0.85, "vignette": 0.45,
         "desc": "低饱和低对比，柔和怀旧褪色感。"},
        {"name": "暖阳", "temp": 35, "tint": -3, "hue": 10, "sat": 1.10, "gamma": 0.95, "contrast": 1.00, "vignette": 0.10,
         "desc": "温暖金黄，如夕阳下的柔和光线。"},
        {"name": "冷调", "temp": -30, "tint": 0, "hue": 0, "sat": 1.00, "gamma": 1.00, "contrast": 1.00, "vignette": 0.00,
         "desc": "偏冷色调，干净清冷。"},
        {"name": "蓝调", "temp": -40, "tint": -5, "hue": 0, "sat": 0.95, "gamma": 1.00, "contrast": 1.05, "vignette": 0.20,
         "desc": "蓝色氛围，静谧忧郁的电影感。"},
        {"name": "青橙", "temp": 15, "tint": -10, "hue": 0, "sat": 1.15, "gamma": 1.08, "contrast": 1.15, "vignette": 0.15,
         "desc": "电影青橙色调，暗部偏青、亮部偏橙。"},
        {"name": "黄昏", "temp": 30, "tint": 15, "hue": 0, "sat": 1.05, "gamma": 0.95, "contrast": 1.00, "vignette": 0.25,
         "desc": "橙紫暮色，浪漫黄昏氛围。"},
        {"name": "高对比", "temp": 0, "tint": 0, "hue": 0, "sat": 1.15, "gamma": 1.15, "contrast": 1.35, "vignette": 0.15,
         "desc": "增强明暗对比，锐利立体有冲击力。"},
        {"name": "柔和", "temp": 5, "tint": 0, "hue": 0, "sat": 0.90, "gamma": 0.90, "contrast": 0.90, "vignette": 0.10,
         "desc": "柔化高光阴影，温柔梦幻。"},
        {"name": "梦幻", "temp": 10, "tint": 5, "hue": 0, "sat": 0.80, "gamma": 0.85, "contrast": 0.82, "vignette": 0.30,
         "desc": "低对比柔光，朦胧梦幻。"},
        {"name": "银盐", "temp": 0, "tint": 0, "hue": 0, "sat": 0.00, "gamma": 1.15, "contrast": 1.40, "vignette": 0.30,
         "desc": "高反差黑白，颗粒感强，银盐胶片味。"},
        {"name": "暗角", "temp": 0, "tint": 0, "hue": 0, "sat": 1.00, "gamma": 1.00, "contrast": 1.00, "vignette": 0.60,
         "desc": "四周压暗，突出中心主体。"},
    ]


FILTERS = build_filters()

# ----------------------------------------------------------------------------
# 图像处理：滤镜
# ----------------------------------------------------------------------------
def _gamma_lut(gamma):
    return np.clip(((np.arange(256, dtype=np.float32) / 255.0) ** (1.0 / gamma)) * 255.0, 0, 255).astype(np.uint8)


_GAMMA_CACHE = {}


def _cached_gamma(gamma):
    g = round(gamma, 2)
    if g not in _GAMMA_CACHE:
        _GAMMA_CACHE[g] = _gamma_lut(g)
    return _GAMMA_CACHE[g]


_VIGNETTE_CACHE = {}


def _vignette_mask(h, w, strength):
    key = (h, w, round(strength, 2))
    if key not in _VIGNETTE_CACHE:
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = w / 2.0, h / 2.0
        d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) / (np.sqrt(cx * cx + cy * cy))
        mask = 1.0 - strength * np.clip(d - 0.45, 0.0, 1.0)
        _VIGNETTE_CACHE[key] = np.clip(mask, 0.0, 1.0)[..., None].astype(np.float32)
    return _VIGNETTE_CACHE[key]


def apply_filter(bgr, f):
    if f.get("identity"):
        return bgr
    img = bgr.astype(np.float32)

    # 白平衡（色温 / 色调）-> 通道增益
    temp, tint = f["temp"], f["tint"]
    r_gain = 1.0 + temp / 220.0 - tint / 320.0
    g_gain = 1.0 + tint / 420.0
    b_gain = 1.0 - temp / 220.0 - tint / 320.0
    img[..., 2] *= r_gain
    img[..., 1] *= g_gain
    img[..., 0] *= b_gain

    # 色相偏移 + 饱和度
    hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
    if f["hue"]:
        hsv[..., 0] = (hsv[..., 0].astype(np.int32) + int(f["hue"] // 2)) % 180
        hsv[..., 0] = hsv[..., 0].astype(np.uint8)
    hsv[..., 1] = np.clip(hsv[..., 1].astype(np.float32) * f["sat"], 0, 255).astype(np.uint8)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # gamma（查表，快）
    g = f["gamma"]
    if g != 1.0:
        lut = _cached_gamma(g)
        out = cv2.LUT(out, lut)

    # 对比度 + 亮度
    out = cv2.convertScaleAbs(out, alpha=f["contrast"], beta=f.get("bright", 0))

    # 暗角
    if f["vignette"] > 0:
        out = np.clip(out.astype(np.float32) * _vignette_mask(out.shape[0], out.shape[1], f["vignette"]), 0, 255).astype(np.uint8)

    return out


# ----------------------------------------------------------------------------
# 图像处理：美颜（美白 / 瘦脸 / 大眼）与画质增强
# ----------------------------------------------------------------------------
def whiten(frame, strength):
    """美白：轻微降饱和 + 提亮。"""
    if strength <= 0:
        return frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1].astype(np.float32) * (1.0 - 0.25 * strength / 100.0), 0, 255).astype(np.uint8)
    hsv[..., 2] = np.clip(hsv[..., 2].astype(np.float32) + 12 * strength / 100.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


_GRID_CACHE = {}


def _base_grid(h, w):
    """缓存基础坐标网格，避免每帧重新 mgrid 分配大数组。"""
    key = (h, w)
    if key not in _GRID_CACHE:
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        _GRID_CACHE[key] = (ys, xs)
    return _GRID_CACHE[key]


def build_beauty_warp(frame, faces, slim, eye):
    """构造瘦脸 + 大眼的形变场，返回 map_x / map_y（供 cv2.remap 使用）。
    在 1/2 分辨率上计算位移场再放大，位移场平滑、精度足够，可大幅降低耗时。"""
    h, w = frame.shape[:2]
    sh, sw = h // 2, w // 2
    ys, xs = _base_grid(sh, sw)
    dx = np.zeros((sh, sw), np.float32)
    dy = np.zeros((sh, sw), np.float32)

    for face in faces:
        fx, fy, fw, fh = face[0] * 0.5, face[1] * 0.5, face[2] * 0.5, face[3] * 0.5
        cx = fx + fw / 2.0
        jaw_cy = fy + fh * 0.72  # 下巴大致位置

        if slim > 0:
            # 脸内椭圆软掩码：把形变限制在脸内，避免背景被扭曲
            m = np.zeros((sh, sw), np.float32)
            cv2.ellipse(m, (int(cx), int(fy + fh * 0.5)),
                        (int(max(3, fw * 0.5)), int(max(3, fh * 0.55))), 0, 0, 360, 1.0, -1)
            m = cv2.GaussianBlur(m, (0, 0), max(3.0, fw * 0.08))
            X = xs - cx
            Y = ys - jaw_cy
            g = np.exp(-(X * X) / (2 * (fw * 0.45) ** 2) - (Y * Y) / (2 * (fh * 0.40) ** 2))
            dx += X * g * m * (slim / 100.0) * 0.28

        if eye > 0:
            lm = face[4:14].reshape(5, 2) * 0.5  # 右眼 左眼 鼻尖 右嘴角 左嘴角
            for ei in (0, 1):
                ex, ey = lm[ei]
                if ex <= 0 or ey <= 0:
                    continue
                R = max(fw * 0.16, 5.0)
                D = np.sqrt((xs - ex) ** 2 + (ys - ey) ** 2)
                wt = np.clip(1.0 - D / R, 0.0, 1.0) ** 2
                k = (eye / 100.0) * 0.30
                dx += -(xs - ex) * wt * k
                dy += -(ys - ey) * wt * k

    # 位移场放大回原尺寸（×2 换算坐标单位），并与全尺寸基础网格合成
    dx = cv2.resize(dx, (w, h), interpolation=cv2.INTER_LINEAR) * 2.0
    dy = cv2.resize(dy, (w, h), interpolation=cv2.INTER_LINEAR) * 2.0
    fys, fxs = _base_grid(h, w)
    return (fxs + dx).astype(np.float32), (fys + dy).astype(np.float32)


def enhance(frame, clarity, sharpen, brightness, contrast, saturation):
    """画质增强：清晰度 / 锐化（反锐化掩模）、亮度、对比度、饱和度。"""
    if clarity > 0 or sharpen > 0:
        if clarity > 0:
            blur = cv2.GaussianBlur(frame, (0, 0), 3.0)
            frame = cv2.addWeighted(frame, 1.0 + clarity / 100.0 * 0.9, blur, -clarity / 100.0 * 0.9, 0)
        if sharpen > 0:
            blur2 = cv2.GaussianBlur(frame, (0, 0), 1.2)
            frame = cv2.addWeighted(frame, 1.0 + sharpen / 100.0 * 0.6, blur2, -sharpen / 100.0 * 0.6, 0)
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if brightness != 0 or contrast != 0:
        alpha = 1.0 + contrast / 100.0
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=brightness)
    if saturation != 0:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1].astype(np.float32) * (1.0 + saturation / 100.0), 0, 255).astype(np.uint8)
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return frame


# ----------------------------------------------------------------------------
# 相机：采集 + 参数（硬件尽力 + 软件兜底）
# ----------------------------------------------------------------------------
class Camera:
    def __init__(self):
        self.cap = None
        self.width = 640
        self.height = 480
        self.auto_exposure = True
        self.auto_gain = 1.0       # 软件自动曝光增益（反馈控制）
        self.manual_gain = 0.0     # 手动软件增益（EV）
        self.night = False         # 深夜模式（低照度增强）
        self._luma_wgt = None      # 中心加权矩阵缓存
        self._luma_wsum = 1.0
        self.error = None

    def open(self):
        self.error = None
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap.release()
            self.error = "无法打开摄像头"
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap = cap
        self._set_auto(True)
        return True

    def read(self):
        if self.cap is None:
            return False, None
        return self.cap.read()

    # ---- 硬件参数（能设就设，设不了忽略）----
    def _try_set(self, prop, val):
        try:
            self.cap.set(prop, val)
        except Exception:
            pass

    def _set_auto(self, auto):
        self.auto_exposure = auto
        if self.cap is None:
            return
        if auto:
            self._try_set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
            self._try_set(cv2.CAP_PROP_AUTO_WB, 1.0)
            self._try_set(cv2.CAP_PROP_AUTOFOCUS, 1.0)
        else:
            self._try_set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self._try_set(cv2.CAP_PROP_AUTO_WB, 0.0)
            self._try_set(cv2.CAP_PROP_AUTOFOCUS, 0.0)

    def set_mode(self, auto):
        self._set_auto(auto)

    def set_exposure(self, v):
        """v: 0..100 -> 快门(曝光)时间映射。"""
        if not self.auto_exposure:
            self._try_set(cv2.CAP_PROP_EXPOSURE, max(-13.0, -13.0 + v / 100.0 * 9.0))

    def set_iso(self, v):
        """v: 0..100 -> ISO(增益)。硬件失败时软件增益兜底。"""
        self.manual_gain = v / 100.0 * 0.6
        if not self.auto_exposure:
            self._try_set(cv2.CAP_PROP_GAIN, v / 100.0 * 250.0)

    def set_wb(self, v):
        """v: 0..100 -> 白平衡色温。"""
        if not self.auto_exposure:
            self._try_set(cv2.CAP_PROP_WB_TEMPERATURE, 3000 + v / 100.0 * 5000.0)

    def set_focus(self, v):
        """v: 0..100 -> 对焦距离（0=近，100=远）。"""
        if not self.auto_exposure:
            self._try_set(cv2.CAP_PROP_FOCUS, v / 100.0)

    # ---- 软件自动曝光反馈 ----
    def auto_expose(self, frame, face=None):
        if self.night:
            # 深夜模式：暗光下整体提亮，目标亮度更高、增益上限更大，保证夜间清晰度
            base = self._center_luma(frame)
            target = 0.55
            boost = float(np.clip(1.0 + (target - base) * 2.0, 1.0, 2.2))
            self.auto_gain = float(np.clip(self.auto_gain + (boost - self.auto_gain) * 0.35, 1.0, 2.2))
            return self.auto_gain

        # 硬件自动曝光已处理整体亮度；这里仅在背光（人脸明显比背景暗）时小幅提亮，避免整体过曝
        base = self._center_luma(frame)
        boost = 1.0
        if face is not None:
            x, y, w, h = (int(v) for v in face[:4])
            x, y = max(0, x), max(0, y)
            x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
            if x2 - x > 10 and y2 - y > 10:
                face_luma = cv2.cvtColor(frame[y:y2, x:x2], cv2.COLOR_BGR2GRAY).mean() / 255.0
                if face_luma < base - 0.10:
                    boost = float(np.clip(1.0 + (base - face_luma - 0.10) * 1.0, 1.0, 1.25))
        self.auto_gain = float(np.clip(self.auto_gain + (boost - self.auto_gain) * 0.25, 1.0, 1.25))
        return self.auto_gain

    def _center_luma(self, frame):
        g = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 48))
        if self._luma_wgt is None:
            h, w = g.shape
            ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
            self._luma_wgt = np.exp(-((xs - w / 2) ** 2 + (ys - h / 2) ** 2) / (2 * (min(w, h) * 0.4) ** 2))
            self._luma_wsum = self._luma_wgt.sum()
        return float((g.astype(np.float32) * self._luma_wgt).sum() / self._luma_wsum) / 255.0

    def apply_gain(self, frame):
        """应用软件增益：自动模式用自动增益，手动模式用手动 ISO 增益（二者不叠加）。"""
        if self.auto_exposure:
            g = self.auto_gain
        else:
            g = 1.0 + self.manual_gain
        g = max(0.3, min(3.0, g))
        if abs(g - 1.0) < 0.01:
            return frame
        return cv2.convertScaleAbs(frame, alpha=g, beta=0)

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# ----------------------------------------------------------------------------
# 人脸检测
# ----------------------------------------------------------------------------
class FaceDetector:
    def __init__(self, model_path):
        self.detector = cv2.FaceDetectorYN.create(
            model_path, "", (320, 320), score_threshold=0.5, nms_threshold=0.3, top_k=5000
        )
        self.frame_skip = 0

    def detect(self, frame):
        h, w = frame.shape[:2]
        # 缩小到短边 320 再检测，显著降低耗时；坐标随后放大回原分辨率
        scale = 320.0 / max(w, h)
        if scale < 1.0:
            sw, sh = int(w * scale), int(h * scale)
            small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        else:
            small, sw, sh = frame, w, h
        self.detector.setInputSize((sw, sh))
        _, faces = self.detector.detect(small)
        if faces is None:
            return []
        rx, ry = w / float(sw), h / float(sh)
        result = []
        for f in faces:
            # cv2 5.0 输出： [x,y,w,h, 5个关键点(10坐标), score]，置信度在最后一位(14)
            if float(f[14]) < 0.5:
                continue
            f = f.copy()
            f[0], f[1], f[2], f[3] = f[0] * rx, f[1] * ry, f[2] * rx, f[3] * ry
            f[4:14:2] *= rx
            f[5:14:2] *= ry
            result.append(f)
        return result


# ----------------------------------------------------------------------------
# 主界面
# ----------------------------------------------------------------------------
class BeautyCameraApp:
    def __init__(self, root):
        self.root = root
        self.night = False  # 深夜模式开关
        root.title("神猫相机")
        root.configure(bg=self._c("bg"))

        self.camera = Camera()
        self.detector = None
        if os.path.exists(MODEL_PATH):
            try:
                self.detector = FaceDetector(MODEL_PATH)
            except Exception as e:
                print("人脸检测器加载失败：", e)
                self.detector = None

        # 处理参数（美颜 / 增强 / 滤镜）
        self.par = {
            "whiten": 25, "slim": 35, "eye": 25,
            "clarity": 0, "sharpen": 0, "brightness": 0, "contrast": 0, "saturation": 0,
            "filter": 0,
        }
        self.last_faces = []
        self._frame_no = 0
        self._fps = 0.0
        self._last_t = time.time()

        # 滤镜画廊相关状态
        self._bg = None            # 预览背景图（BGR）
        self._thumb_np = []        # 各滤镜的 numpy 缩略图
        self._thumb_photos = []    # 各滤镜的 PhotoImage（防回收）
        self._thumb_w = 0          # 缩略图单元宽（含间距）
        self._filter_sel = None    # 当前高亮框（画布物品 id）

        # 动画状态（猫眨眼 / 闪光粒子）
        self._blink_wait = 40      # 距下次眨眼倒计时（帧）
        self._blink_left = 0       # 闭眼剩余帧数
        self._sparks = []          # 漂浮闪光粒子

        self._build_ui()
        self._animate()
        self._start_camera()

    # ---- UI ----
    def _build_ui(self):
        self._setup_style()
        main = tk.Frame(self.root, bg=self._c("bg"))
        main.pack(fill="both", expand=True, padx=6, pady=6)
        self.main = main

        # 顶部：标题栏（白天木板 + 醒猫；深夜夜空 + 睡猫）
        self._build_header(main)

        # 底部：胶片画廊
        self._build_filter_gallery(main)

        # 中部：预览（相框）+ 控制面板
        top = tk.Frame(main, bg=self._c("bg"))
        top.pack(fill="both", expand=True, pady=(6, 0))

        left = tk.Frame(top, bg=self._c("frame"), padx=8, pady=8)
        left.pack(side="left", fill="both", expand=True)
        self.video_label = tk.Label(left, bg=self._c("video"))
        self.video_label.pack(fill="both", expand=True)

        right = tk.Frame(top, bg=self._c("panel"), width=340, padx=8, pady=8)
        right.pack(side="right", fill="both", padx=(8, 0))
        right.pack_propagate(False)

        btn_bar = tk.Frame(right, bg=self._c("panel"))
        btn_bar.pack(fill="x", pady=(0, 6))
        tk.Button(btn_bar, text="📷 拍照", command=self.capture, bg=self._c("btn"), fg=self._c("fg"),
                  relief="raised", bd=2, font=("Microsoft YaHei", 10, "bold")).pack(side="left", fill="x", expand=True, padx=2)
        tk.Button(btn_bar, text="打开照片", command=self.open_photos, bg=self._c("btn2"), fg=self._c("fg"),
                  relief="raised", bd=2).pack(side="left", fill="x", expand=True, padx=2)
        # 深夜模式切换
        nb_txt = "☀️ 白天模式" if self.night else "🌙 深夜模式"
        tk.Button(right, text=nb_txt, command=self._toggle_night, bg=self._c("btn2"), fg=self._c("fg"),
                  relief="raised", bd=2).pack(fill="x", pady=(0, 6))
        self.status_var = tk.StringVar(value="初始化中…")
        tk.Label(right, textvariable=self.status_var, bg=self._c("panel"), fg=self._c("fg")).pack(anchor="w", pady=(0, 4))

        nb = ttk.Notebook(right)
        nb.pack(fill="both", expand=True)

        # 美颜
        tab = tk.Frame(nb, bg=self._c("panel")); nb.add(tab, text="美颜")
        self._slider(tab, "美白", "whiten", 0, 100)
        self._slider(tab, "瘦脸", "slim", 0, 100)
        self._slider(tab, "大眼", "eye", 0, 100)

        # 画质
        tab = tk.Frame(nb, bg=self._c("panel")); nb.add(tab, text="画质")
        self._slider(tab, "清晰度", "clarity", 0, 100)
        self._slider(tab, "锐化", "sharpen", 0, 100)
        self._slider(tab, "亮度", "brightness", -100, 100)
        self._slider(tab, "对比度", "contrast", -100, 100)
        self._slider(tab, "饱和度", "saturation", -100, 100)

        # 相机
        tab = tk.Frame(nb, bg=self._c("panel")); nb.add(tab, text="相机")
        self.mode_var = tk.StringVar(value="自动")
        mb = tk.Frame(tab, bg=self._c("panel"))
        mb.pack(fill="x", pady=2)
        tk.Radiobutton(mb, text="自动", variable=self.mode_var, value="自动", command=self._on_mode,
                       bg=self._c("panel"), fg=self._c("fg"), activebackground=self._c("panel"), activeforeground=self._c("fg"),
                       selectcolor=self._c("trough"), font=("Microsoft YaHei", 9)).pack(side="left")
        tk.Radiobutton(mb, text="手动", variable=self.mode_var, value="手动", command=self._on_mode,
                       bg=self._c("panel"), fg=self._c("fg"), activebackground=self._c("panel"), activeforeground=self._c("fg"),
                       selectcolor=self._c("trough"), font=("Microsoft YaHei", 9)).pack(side="left", padx=8)
        self._slider(tab, "快门(曝光)", "exposure", 0, 100)
        self._slider(tab, "ISO(增益)", "iso", 0, 100)
        self._slider(tab, "白平衡", "wb", 0, 100)
        self._slider(tab, "聚焦", "focus", 0, 100)
        self._on_mode()  # 初始化滑杆启用/禁用状态

    def _c(self, role):
        """按当前主题取色。"""
        return PALETTE["night" if self.night else "day"][role]

    def _toggle_night(self):
        """切换深夜模式：重建界面（保留相机与已生成的滤镜缩略图）。"""
        self.night = not self.night
        self.root.configure(bg=self._c("bg"))
        self._thumb_photos = []
        if hasattr(self, "main"):
            self.main.destroy()
        self._build_ui()  # 内部会复用已有缩略图并重建画廊
        # 相机随主题切换到夜间参数
        self.camera.night = self.night

    def _setup_style(self):
        """ttk 主题（Notebook 标签 / 滚动条），随主题配色。"""
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TNotebook", background=self._c("panel"), borderwidth=0)
        style.configure("TNotebook.Tab", background=self._c("tab"), foreground=self._c("fg"),
                        padding=[14, 6], font=("Microsoft YaHei", 9, "bold"))
        style.map("TNotebook.Tab",
                  background=[("selected", self._c("tab_sel")), ("active", self._c("tab_active"))],
                  foreground=[("selected", self._c("fg"))])
        style.configure("Horizontal.TScrollbar", background=self._c("trough"),
                        troughcolor=self._c("panel"), bordercolor=self._c("panel"),
                        arrowcolor=self._c("fg"))

    # ---- 标题栏 / 装饰 / 动画（白天复古木板 / 深夜夜空） ----
    def _build_header(self, parent):
        c = tk.Canvas(parent, height=132, bg=self._c("bg"), highlightthickness=0)
        c.pack(fill="x")
        self.header = c
        if self.night:
            self._draw_night_sky(c)
            self._draw_moon(c)
            self._draw_lamp(c)
            self._cat = self._draw_sleeping_cat(c)
            c.create_text(178, 40, text="神猫相机", fill=self._c("fg"), anchor="w",
                          font=("Microsoft YaHei", 26, "bold"))
            c.create_text(180, 70, text="SHENMAO · NIGHT MODE", fill=self._c("sub"), anchor="w",
                          font=("Georgia", 10, "italic"))
            self._init_stars(c)
        else:
            self._draw_wood(c)
            self._cat = self._draw_cat(c)
            c.create_text(178, 40, text="神猫相机", fill=GOLD, anchor="w",
                          font=("Microsoft YaHei", 26, "bold"))
            c.create_text(180, 70, text="SHENMAO · RETRO CAMERA", fill=WOOD_LIGHT, anchor="w",
                          font=("Georgia", 10, "italic"))
            self._draw_camera(c)
            self._draw_film(c)
            self._init_sparks(c)

    # ---- 白天装饰 ----
    def _draw_wood(self, c, w=2000):
        """横向木板 + 木纹，铺满标题栏背景。"""
        plank = 22
        for i in range(0, 132, plank):
            shade = WOOD_MID if (i // plank) % 2 == 0 else "#7c4f29"
            c.create_rectangle(0, i, w, i + plank, fill=shade, outline="")
            c.create_line(0, i, w, i, fill=WOOD_DARK, width=2)
            for g in range(4):
                gy = i + 5 + g * 4
                c.create_line(0, gy, w, gy + 2, fill="#6b4423", width=1)

    def _draw_cat(self, c):
        """可爱卡通小猫（返回睁眼/闭眼物品 id，供眨眼动画）。"""
        ink = INK
        c.create_polygon(40, 48, 50, 12, 78, 34, fill="#e8a33d", outline=ink, width=2)
        c.create_polygon(130, 48, 120, 12, 92, 34, fill="#e8a33d", outline=ink, width=2)
        c.create_polygon(48, 42, 55, 22, 68, 34, fill="#f2b8c0", outline="")
        c.create_polygon(122, 42, 115, 22, 102, 34, fill="#f2b8c0", outline="")
        c.create_oval(32, 30, 138, 118, fill="#f5c56b", outline=ink, width=3)
        eye_l = c.create_oval(58, 64, 82, 88, fill="#2a2118", outline="")
        eye_r = c.create_oval(98, 64, 122, 88, fill="#2a2118", outline="")
        hi_l = c.create_oval(70, 68, 78, 76, fill="#ffffff", outline="")
        hi_r = c.create_oval(110, 68, 118, 76, fill="#ffffff", outline="")
        close_l = c.create_line(58, 76, 82, 76, fill="#2a2118", width=3, state="hidden")
        close_r = c.create_line(98, 76, 122, 76, fill="#2a2118", width=3, state="hidden")
        c.create_polygon(82, 90, 88, 90, 85, 96, fill="#e8847f", outline="")
        c.create_arc(74, 90, 84, 100, start=200, extent=140, style="arc", outline=ink, width=2)
        c.create_arc(86, 90, 96, 100, start=200, extent=140, style="arc", outline=ink, width=2)
        c.create_oval(50, 88, 66, 98, fill="#f4b6c0", outline="")
        c.create_oval(104, 88, 120, 98, fill="#f4b6c0", outline="")
        for y in (74, 84, 94):
            c.create_line(26, y, 50, y, fill=ink, width=2)
            c.create_line(120, y, 144, y, fill=ink, width=2)
        return {"open": [eye_l, eye_r, hi_l, hi_r], "closed": [close_l, close_r]}

    def _draw_camera(self, c, x0=900):
        """复古相机剪影装饰。"""
        ink = INK
        c.create_rectangle(x0, 20, x0 + 150, 100, fill=WOOD_LIGHT, outline=ink, width=2)
        c.create_rectangle(x0 + 10, 12, x0 + 58, 36, fill=WOOD_MID, outline=ink, width=2)
        c.create_oval(x0 + 45, 46, x0 + 105, 106, fill=FILM_DARK, outline=ink, width=3)
        c.create_oval(x0 + 58, 59, x0 + 92, 93, fill="#6b4423", outline=ink, width=2)
        c.create_oval(x0 + 68, 69, x0 + 82, 83, fill=GOLD, outline="")
        c.create_oval(x0 + 120, 76, x0 + 146, 100, fill=GOLD, outline=ink, width=2)

    def _draw_film(self, c, x0=880):
        """一条带齿孔的胶片装饰。"""
        y0 = 106
        c.create_rectangle(x0, y0, x0 + 200, y0 + 16, fill=FILM_DARK, outline="")
        for i in range(10):
            px = x0 + 4 + i * 20
            c.create_rectangle(px, y0 + 1, px + 8, y0 + 7, fill=CREAM, outline="")
            c.create_rectangle(px, y0 + 9, px + 8, y0 + 15, fill=CREAM, outline="")

    def _init_sparks(self, c):
        self._sparks = []
        for i in range(6):
            x = 210 + i * 95
            y = 30 + (i * 37) % 85
            it = c.create_oval(x - 3, y - 3, x + 3, y + 3, fill=GOLD, outline="")
            self._sparks.append([it, float(x), float(y), 0.5 + (i % 3) * 0.3, i])

    # ---- 深夜装饰 ----
    def _draw_night_sky(self, c, w=2000):
        """深夜天空：上深下略亮的新变，模拟微弱天光。"""
        for i in range(132):
            t = i / 132.0
            col = "#%02x%02x%02x" % (int(14 + t * 7), int(20 + t * 9), int(32 + t * 13))
            c.create_line(0, i, w, i, fill=col)

    def _draw_moon(self, c):
        """淡淡的月亮（右上角）：柔光晕 + 浅色月盘，不太亮。"""
        cx, cy, r = 985, 38, 22
        for rr, col in ((r + 16, "#161f30"), (r + 10, "#1a2438"), (r + 5, "#232f46")):
            c.create_oval(cx - rr, cy - rr, cx + rr, cy + rr, fill=col, outline="")
        c.create_oval(cx - r, cy - r, cx + r, cy + r, fill="#9fb4cc", outline="")
        c.create_oval(cx - r + 6, cy - r + 5, cx + r - 4, cy + r - 4, fill="#c3d4e6", outline="")

    def _draw_lamp(self, c):
        """路灯：细杆 + 灯罩 + 微弱暖光晕（不太亮）。"""
        x = 890
        c.create_line(x, 132, x, 58, fill="#2c3a58", width=5)
        c.create_line(x - 24, 58, x + 6, 58, fill="#2c3a58", width=5)
        for rr, col in ((34, "#141c2c"), (26, "#182133"), (18, "#1d2940")):
            c.create_oval(x + 4 - rr, 58 - rr, x + 4 + rr, 58 + rr, fill=col, outline="")
        c.create_polygon(x - 6, 58, x + 14, 58, x + 10, 40, x - 2, 40, fill="#2c3a58", outline="")
        c.create_oval(x - 1, 44, x + 9, 54, fill="#e6c27a", outline="")

    def _draw_sleeping_cat(self, c):
        """睡眠中的小猫：蜷缩身体 + 闭眼 + 'Z z z' 呼吸符号。"""
        ink = self._c("sub")
        c.create_oval(30, 58, 152, 122, fill="#c99a56", outline=ink, width=2)   # 蜷缩身体
        c.create_oval(36, 40, 120, 96, fill="#e8b878", outline=ink, width=2)    # 头
        c.create_polygon(44, 46, 54, 20, 70, 40, fill="#c99a56", outline=ink, width=2)
        c.create_polygon(112, 46, 102, 20, 86, 40, fill="#c99a56", outline=ink, width=2)
        close_l = c.create_arc(52, 62, 74, 74, start=200, extent=140, style="arc", outline=ink, width=2)
        close_r = c.create_arc(86, 62, 108, 74, start=200, extent=140, style="arc", outline=ink, width=2)
        c.create_polygon(78, 78, 84, 78, 81, 84, fill="#d8907f", outline="")
        c.create_oval(48, 74, 60, 82, fill="#d99a94", outline="")
        c.create_oval(100, 74, 112, 82, fill="#d99a94", outline="")
        for y in (70, 78, 86):
            c.create_line(32, y, 48, y, fill=ink, width=1)
            c.create_line(112, y, 128, y, fill=ink, width=1)
        c.create_arc(122, 88, 158, 128, start=180, extent=180, style="arc", outline=ink, width=3)
        zz = c.create_text(138, 30, text="z Z z", fill=self._c("sub"), anchor="w",
                           font=("Georgia", 12, "italic"))
        return {"open": [], "closed": [close_l, close_r], "zz": zz}

    def _init_stars(self, c):
        self._sparks = []
        for i in range(7):
            x = 210 + i * 90
            y = 18 + (i * 53) % 90
            it = c.create_oval(x - 2, y - 2, x + 2, y + 2, fill=self._c("spark"), outline="")
            self._sparks.append([it, float(x), float(y), 0.4 + (i % 3) * 0.2, i])

    def _animate(self):
        """标题栏动态效果：白天小猫眨眼 + 粒子；深夜星星闪烁（猫沉睡，不眨眼）。"""
        if not self.night:
            if self._blink_left > 0:
                self._blink_left -= 1
                self._set_blink(True)
            else:
                self._set_blink(False)
                self._blink_wait -= 1
                if self._blink_wait <= 0:
                    self._blink_left = 4
                    self._blink_wait = 55
        self._move_sparks()
        self.root.after(50, self._animate)

    def _set_blink(self, closed):
        c = self.header
        if not hasattr(self, "_cat"):
            return
        for i in self._cat["open"]:
            c.itemconfigure(i, state="hidden" if closed else "normal")
        for i in self._cat["closed"]:
            c.itemconfigure(i, state="normal" if closed else "hidden")

    def _move_sparks(self):
        c = self.header
        if not self._sparks:
            return
        bright = self._c("spark")
        dim = "#f7d98b" if not self.night else self._c("sub")
        for s in self._sparks:
            it, x, y, vy, _ = s
            y -= vy
            if y < 6:
                y = 120
            s[2] = y
            c.coords(it, x - 3, y - 3, x + 3, y + 3)
            c.itemconfigure(it, fill=bright if (int(y) // 8) % 2 else dim)

    def _slider(self, parent, label, key, lo, hi, format_func=None):
        row = tk.Frame(parent, bg=self._c("panel"))
        row.pack(fill="x", pady=3)
        tk.Label(row, text=label, width=8, bg=self._c("panel"), fg=self._c("fg"),
                 font=("Microsoft YaHei", 9)).pack(side="left", anchor="w")
        val_lbl = tk.Label(row, text=str(self.par.get(key, 0)), width=5,
                           bg=self._c("trough"), fg=self._c("fg"), relief="groove", bd=1,
                           font=("Microsoft YaHei", 9, "bold"))
        val_lbl.pack(side="right")
        s = tk.Scale(
            row, from_=lo, to=hi, orient="horizontal", showvalue=False,
            command=lambda v, k=key, vl=val_lbl, f=format_func: self._on_slider(k, v, vl, f),
            bg=self._c("panel"), fg=self._c("fg"), troughcolor=self._c("trough"),
            highlightthickness=0, bd=0, sliderrelief="raised",
            sliderlength=16, activebackground=self._c("accent"), repeatdelay=200,
        )
        s.set(self.par.get(key, 0))
        s.pack(side="left", fill="x", expand=True, padx=6)
        self.__dict__["slider_" + key] = s

    def _on_slider(self, key, v, val_lbl, format_func):
        try:
            iv = int(float(v))
        except ValueError:
            return
        if key in ("exposure", "iso", "wb", "focus"):
            # 相机参数：立即下发
            self._apply_cam_param(key, iv)
            val_lbl.config(text=("%s" % format_func(iv)) if format_func else str(iv))
            return
        self.par[key] = iv
        val_lbl.config(text=("%s" % format_func(iv)) if format_func else str(iv))

    def _on_mode(self):
        auto = self.mode_var.get() == "自动"
        self.camera.set_mode(auto)
        # 自动模式下禁用曝光/ISO/白平衡/聚焦滑杆，切到手动才可调
        st = "disabled" if auto else "normal"
        for k in ("exposure", "iso", "wb", "focus"):
            self.__dict__["slider_" + k].configure(state=st)

    def _apply_cam_param(self, key, v):
        if key == "exposure":
            self.camera.set_exposure(v)
        elif key == "iso":
            self.camera.set_iso(v)
        elif key == "wb":
            self.camera.set_wb(v)
        elif key == "focus":
            self.camera.set_focus(v)

    # ---- 滤镜预览与介绍 ----
    def _load_bg(self):
        """滤镜预览背景：像素小猫（缓存）。"""
        if self._bg is None:
            self._bg = build_pixel_cat()
        return self._bg

    def _build_filter_gallery(self, parent):
        """底部滤镜画廊：胶片条样式（齿孔 + 横向滚动缩略图）。"""
        bar = tk.Frame(parent, bg=self._c("panel"), padx=6, pady=6)
        bar.pack(side="bottom", fill="x", pady=(8, 0))

        self.filter_name = tk.StringVar(value=FILTERS[self.par["filter"]]["name"])
        self.filter_desc = tk.StringVar(value=FILTERS[self.par["filter"]]["desc"])
        tk.Label(bar, textvariable=self.filter_name, bg=self._c("panel"), fg=self._c("fg"),
                 font=("Microsoft YaHei", 10, "bold")).pack(anchor="w")
        tk.Label(bar, textvariable=self.filter_desc, bg=self._c("panel"), fg=self._c("desc"),
                 wraplength=1000, justify="left").pack(anchor="w", pady=(0, 6))

        self._make_sprockets(bar)
        self.filter_canvas = tk.Canvas(bar, height=84, bg=self._c("film"), highlightthickness=0)
        self.filter_canvas.pack(fill="x")
        self._make_sprockets(bar)

        hbar = ttk.Scrollbar(bar, orient="horizontal", command=self.filter_canvas.xview, style="Horizontal.TScrollbar")
        self.filter_canvas.configure(xscrollcommand=hbar.set)
        hbar.pack(fill="x")
        self.filter_canvas.bind("<Button-1>", self._on_filter_click)
        self.filter_canvas.bind("<MouseWheel>", self._on_filter_wheel)

        # 缩略图：仅在尚未生成时后台生成；已有则直接绘制（深夜切换时复用）
        if not self._thumb_np:
            def worker():
                bg = self._load_bg()
                h, w = bg.shape[:2]
                tw = 96
                small = cv2.resize(bg, (tw, max(1, int(tw * h / w))))
                self._thumb_np = [apply_filter(small, FILTERS[i]) for i in range(len(FILTERS))]
            threading.Thread(target=worker, daemon=True).start()
            self.root.after(30, self._populate_filter_gallery)
        else:
            self._populate_filter_gallery()

    def _make_sprockets(self, parent):
        """胶片齿孔条（静态装饰，随宽度自适应填充）。"""
        c = tk.Canvas(parent, height=12, bg=self._c("film"), highlightthickness=0)
        c.pack(fill="x")
        c.bind("<Configure>", lambda e, cc=c: self._fill_sprockets(cc, e.width))
        return c

    def _fill_sprockets(self, c, w):
        c.delete("all")
        for x in range(4, w, 20):
            c.create_rectangle(x, 1, x + 10, 11, fill=self._c("sprocket"), outline="")

    def _populate_filter_gallery(self):
        if len(self._thumb_np) < len(FILTERS):
            self.root.after(30, self._populate_filter_gallery)
            return
        self.filter_canvas.delete("all")
        self._thumb_photos = []
        self._filter_sel = None
        tw = self._thumb_np[0].shape[1] + 4
        self._thumb_w = tw
        for i, a in enumerate(self._thumb_np):
            rgb = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self._thumb_photos.append(photo)
            self.filter_canvas.create_image(i * tw + 2, 2, image=photo, anchor="nw")
        self.filter_canvas.configure(scrollregion=(0, 0, len(FILTERS) * tw, self._thumb_np[0].shape[0] + 4))
        self._highlight_filter(self.par["filter"])

    def _on_filter_click(self, event):
        if not self._thumb_w:
            return
        idx = int(self.filter_canvas.canvasx(event.x) // self._thumb_w)
        if 0 <= idx < len(FILTERS):
            self._apply_filter(idx)

    def _on_filter_wheel(self, event):
        self.filter_canvas.xview_scroll(-3 if event.delta > 0 else 3, "units")

    def _apply_filter(self, idx):
        self.par["filter"] = idx
        self.filter_name.set(FILTERS[idx]["name"])
        self.filter_desc.set(FILTERS[idx]["desc"])
        self._highlight_filter(idx)

    def _highlight_filter(self, idx):
        if not self._thumb_w:
            return
        if self._filter_sel is None:
            self._filter_sel = self.filter_canvas.create_rectangle(0, 0, 0, 0, outline="#ff5f5f", width=3)
        x0 = idx * self._thumb_w
        self.filter_canvas.coords(self._filter_sel, x0 + 1, 1, x0 + self._thumb_w - 1, self._thumb_np[idx].shape[0] + 3)
        self.filter_canvas.tag_raise(self._filter_sel)

    # ---- 采集与处理主循环 ----
    def _start_camera(self):
        if not self.camera.open():
            self.status_var.set("摄像头打开失败：" + (self.camera.error or "未知原因"))
            messagebox.showerror("错误", "无法打开摄像头。\n请确认摄像头未被其他程序占用。")
            return
        self.status_var.set("运行中…")
        self._loop()

    def _loop(self):
        ok, frame = self.camera.read()
        if not ok:
            self.root.after(20, self._loop)
            return
        frame = cv2.flip(frame, 1)  # 镜像

        # 人脸检测（每 2 帧一次，复用结果；先检测供测光用）
        self._frame_no += 1
        if self.detector is not None and self._frame_no % 2 == 1:
            try:
                self.last_faces = self.detector.detect(frame)
            except Exception:
                self.last_faces = []

        # 软件曝光（自动模式：对人脸测光以应对背光）
        if self.camera.auto_exposure:
            self.camera.auto_expose(frame, self.last_faces[0] if self.last_faces else None)
        frame = self.camera.apply_gain(frame)

        # 美颜
        frame = whiten(frame, self.par["whiten"])
        if self.par["slim"] > 0 or self.par["eye"] > 0:
            map_x, map_y = build_beauty_warp(frame, self.last_faces, self.par["slim"], self.par["eye"])
            frame = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        # 滤镜
        frame = apply_filter(frame, FILTERS[self.par["filter"]])

        # 画质增强
        frame = enhance(
            frame, self.par["clarity"], self.par["sharpen"],
            self.par["brightness"], self.par["contrast"], self.par["saturation"],
        )

        # FPS
        now = time.time()
        dt = now - self._last_t
        if dt > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)
        self._last_t = now
        nface = len(self.last_faces)
        self.status_var.set("FPS %.1f ｜ 人脸 %d ｜ %dx%d" % (self._fps, nface, frame.shape[1], frame.shape[0]))

        self._show(frame)
        self.root.after(15, self._loop)

    def _show(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

    # ---- 拍照 ----
    def capture(self):
        # 从当前预览拿最新帧太绕，这里直接再读一帧并套用同样处理
        ok, frame = self.camera.read()
        if not ok or frame is None:
            messagebox.showerror("错误", "无法读取画面")
            return
        frame = cv2.flip(frame, 1)
        if self.camera.auto_exposure:
            self.camera.auto_expose(frame)
        frame = self.camera.apply_gain(frame)
        if self.detector is not None:
            try:
                self.last_faces = self.detector.detect(frame)
            except Exception:
                pass
        frame = whiten(frame, self.par["whiten"])
        if self.par["slim"] > 0 or self.par["eye"] > 0:
            map_x, map_y = build_beauty_warp(frame, self.last_faces, self.par["slim"], self.par["eye"])
            frame = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        frame = apply_filter(frame, FILTERS[self.par["filter"]])
        frame = enhance(frame, self.par["clarity"], self.par["sharpen"],
                        self.par["brightness"], self.par["contrast"], self.par["saturation"])

        fname = "照片_%s.jpg" % time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(get_save_dir(), fname)
        if imwrite_unicode(path, frame):
            self.status_var.set("已保存：%s" % fname)
        else:
            self.status_var.set("保存失败")

    def open_photos(self):
        os.startfile(get_save_dir())

    def on_close(self):
        self.camera.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 1.2)
    except Exception:
        pass
    # 窗口图标（运行源码时；打包后由 exe 图标承载）
    try:
        ico = resource_path(os.path.join("models", "cat_icon.ico"))
        if os.path.exists(ico):
            root.iconbitmap(ico)
    except Exception:
        pass
    root.geometry("1150x780")
    root.minsize(1000, 660)
    app = BeautyCameraApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
