"""npyファイルの点群データをplyファイルに変換するモジュール.

成功していない。
"""
import os
import open3d as o3d
import numpy as np

SCRIPT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR_PATH = os.path.dirname(SCRIPT_DIR_PATH)
PLY_DIR_PATH = os.path.join(PROJECT_DIR_PATH, "predict_points")


def npy_to_ply(file_dir, npy_file_name, save_dir=None):
    # パス設定
    if save_dir is None:
        save_dir = file_dir
    npy_path = os.path.join(file_dir, npy_file_name)
    save_path = os.path.join(save_dir, npy_file_name)

    print(f"npy_path : {npy_path}")
    print(f"save_path: {save_path}")

    # 読み込み
    points = np.load(npy_path)
    # 保存
    o3d.io.write_point_cloud(save_path, points)
