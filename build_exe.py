# -*- coding: utf-8 -*-
"""打包脚本：生成单文件 exe。运行 `python build_exe.py`。"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

cmd = [
    sys.executable, "-m", "PyInstaller",
    "--noconfirm",
    "--clean",
    "--onefile",
    "--windowed",
    "--name", "神猫相机",
    "--icon", os.path.join(HERE, "models", "cat_icon.ico"),
    "--add-data", os.path.join(HERE, "models") + os.pathsep + "models",
    os.path.join(HERE, "beauty_camera.py"),
]

print("执行：", " ".join(cmd))
subprocess.check_call(cmd, cwd=HERE)
print("\n打包完成！exe 位于 dist/ 目录下。")
