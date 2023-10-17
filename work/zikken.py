"""Open3Dで3D点群をメッシュ(ポリゴン)に変換するプログラム.
plyファイルからmeshを生成する.
参考URL
    Open3DとPythonによる実装
    https://tecsingularity.com/open3d/bpa/
    PLYファイルについて
    https://programming-surgeon.com/imageanalysis/ply-python/
@author kawanoichi
実行コマンド
$ python3 mesh.py
"""
import numpy as np
import os
from matplotlib import pyplot as plt
import open3d as o3d


SCRIPT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR_PATH = os.path.dirname(SCRIPT_DIR_PATH)
PLY_DIR_PATH = os.path.join(PROJECT_DIR_PATH, "predict_points")


def show_point(self, ptCloud) -> None:
    """点群を表示する関数.
    
    Args:
        ptCloud(np.ndarray): 点群
    """
    # figureを生成
    fig = plt.figure()

    ax1 = fig.add_subplot(111, projection='3d')
    ax1.scatter(ptCloud[:, 0],
                ptCloud[:, 1],
                ptCloud[:, 2],
                c='b')
    ax1.set_xlabel('X')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')

    plt.show()


def main(self):
    """点群をメッシュ化し、表示する関数."""
    # 点群データの読み込み
    ptCloud = np.load(self.point_file)
    print("ptCloud.shape", ptCloud.shape)

    # print("ptCloud[0]", ptCloud[0][0])
    # print("ptCloud[0]", ptCloud[0][1])
    # print("ptCloud[0]", ptCloud[0][2])
    # print("ptCloud[0]", ptCloud[0, 0])

    self.show_point(ptCloud,
                    show_xyz=True,
                    show_xy=True,
                    show_xz=True,
                    show_yz=True)
    normals = self.normals()
    self.plot_normals(ptCloud, normals)

if __name__ == "__main__":
    print("-- 実験 --")
    print(f"SCRIPT_DIR_PATH : {SCRIPT_DIR_PATH}")
    print(f"PROJECT_DIR_PATH: {PROJECT_DIR_PATH}")
    print(f"PLY_DIR_PATH    : {PLY_DIR_PATH}")

    # main()
    print("終了")
