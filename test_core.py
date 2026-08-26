# -*- coding: utf-8 -*-
"""无头测试：验证检测器加载、滤镜、美颜 warp 不崩溃。"""
import os
import numpy as np
import cv2
import beauty_camera as bc

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

# 1) 检测器
det = bc.FaceDetector(os.path.join(HERE, "models", "face_detection_yunet_2023mar.onnx"))
print("FaceDetector 创建 OK")

# 造一张带渐变背景的图
frame = np.zeros((480, 640, 3), np.uint8)
frame[:] = (90, 120, 200)

# 2) 所有滤镜（含 360 种生成 + identity）
for f in bc.FILTERS:
    out = bc.apply_filter(frame, f)
assert out.shape == frame.shape and out.dtype == np.uint8
print("滤镜数量：", len(bc.FILTERS), "，apply_filter 全部通过")

# 3) 美白（假人脸框）
fake_faces = [np.array([200, 100, 180, 220, 0.9, 260, 200, 320, 200, 290, 260, 250, 280, 330, 280], np.float32)]
out = bc.whiten(frame, 30)
print("美白 OK，shape=", out.shape)

# 4) 瘦脸 + 大眼 warp
map_x, map_y = bc.build_beauty_warp(frame, fake_faces, 40, 30)
warped = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
print("瘦脸/大眼 warp OK，shape=", warped.shape)

# 5) 增强
out = bc.enhance(frame, 30, 20, 10, 10, 15)
print("增强 OK")

print("\n全部核心逻辑测试通过 ✅")
