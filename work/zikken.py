import numpy as np
import open3d as o3d
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import cv2
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import tripy


# from pyntcloud import PyntCloud
from scipy.spatial import Delaunay

SCRIPT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR_PATH = os.path.dirname(SCRIPT_DIR_PATH)
PLY_DIR_PATH = os.path.join(PROJECT_DIR_PATH, "predict_points")


def zikken():
    # 仮の点群データ（x, y, z座標）
    points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0.5, 0.5, 1]
    ])

    # Delaunay 三角形分割
    triangles = Delaunay(points).simplices

    # 三角形メッシュを描画
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for triangle in triangles:
        poly3d = [[points[i, 0], points[i, 1], points[i, 2]] for i in triangle]
        ax.add_collection3d(Poly3DCollection(
            [poly3d], facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


if __name__ == "__main__":
    print("実験")
    zikken()
    print("完了")
