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
# from sklearn.cluster import KMeans

import rotate_coordinate as rotate

SCRIPT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR_PATH = os.path.dirname(SCRIPT_DIR_PATH)
PLY_DIR_PATH = os.path.join(PROJECT_DIR_PATH, "predict_points")


class MakeSurface:
    def __init__(self, point_file: str) -> None:
        """コンストラクタ.

        Args:
            point_file (str): 点群ファイル(.npy)
        """
        self.point_file = point_file
        self.vectors_26 = np.array([])
        self.groupe = None
        self.fig = plt.figure()  # 表示するグラフ

        # 画像の存在チェック
        if not os.path.isfile(self.point_file):
            raise FileNotFoundError("No file '%s'" % self.point_file)

        # 26方位ベクトルの作成
        self.vector_26()
        upper_vector = np.array([0, 1, 0])
        self.upper_vector_index = np.where(
            np.all(self.vectors_26 == upper_vector, axis=1))
        if len(self.upper_vector_index) != 1:
            print("Error: number of upper vector is not 1")
            exit()

    def vector_26(self):
        """26方位ベクトル作成関数."""
        kinds_of_coodinate = [-1, 0, 1]

        # 26方位のベクトル(終点座標)を作成
        for x in kinds_of_coodinate:
            for y in kinds_of_coodinate:
                for z in kinds_of_coodinate:
                    if not x == y == z == 0:
                        append_coordinate = np.array([x, y, z])
                        self.vectors_26 = np.append(
                            self.vectors_26, append_coordinate, axis=0)
        self.vectors_26 = self.vectors_26.reshape(
            (len(kinds_of_coodinate) ^ 3)-1, 3)

    def angle_between_vectors(self, vector_a, vector_b):
        """ベクトル間のなす角を求める関数.

        ベクトルAとベクトルBのなす角を求める.

        Args:
            vector_a: ベクトルA
            vector_b: ベクトルB
        Return:
            theta_deg: なす角
        """
        dot_product = np.dot(vector_a, vector_b)
        magnitude_a = np.linalg.norm(vector_a)
        magnitude_b = np.linalg.norm(vector_b)

        cos_theta = dot_product / (magnitude_a * magnitude_b)

        # acosはarccosine関数で、cosの逆関数です。
        theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))

        # 弧度法から度数法に変換
        return np.degrees(theta_rad)

    def show_point(self, ax, points) -> None:
        """点群を表示する関数.

        Args:
            ptCloud(np.ndarray): 点群
        """
        ax.set(xlabel='x', ylabel='y', zlabel='z')
        ax.scatter(points[:, 0],
                   points[:, 1],
                   points[:, 2],
                   c='b')

    def show_normals(self, ax, points, normals) -> None:
        """点群と法線ベクトルを表示する関数.

        Args:
            points(np.ndarray): 点群
        """

        ax.set(xlabel='x', ylabel='y', zlabel='z')

        # 点をプロット
        ax.scatter(points[:, 0], points[:, 1],
                   points[:, 2], c='b', marker='o', label='Points')

        # 法線ベクトルをプロット
        scale = 0.1  # 矢印のスケール
        for i in range(len(points)):
            if points[i, 0] < -0.05:
                ax.quiver(points[i, 0], points[i, 1], points[i, 2],
                          normals[i, 0]*scale, normals[i, 1]*scale, normals[i, 2]*scale, color='r', length=1.0, normalize=True)

    def edit_normals(self, points: np.ndarray) -> None:
        """法線ベクトルに関連する関数.

        Args:
            points(np.ndarray): 点群
        """
        # Open3DのPointCloudに変換
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        # print("point_cloud\n", point_cloud.points)

        # 法線情報を計算
        point_cloud.estimate_normals()

        # 法線情報にアクセス
        normals = np.asarray(point_cloud.normals)
        # print("normals.shape", normals.shape)

        # 似た方角を向いたベクトルをグループ分け
        self.groupe = np.zeros(normals.shape[0])
        for i, normal in enumerate(normals):
            min_theta = 180  # 比較するためのなす角
            for j, vector26 in enumerate(self.vectors_26):
                angle = int(self.angle_between_vectors(normal, vector26))
                if angle < min_theta:
                    self.groupe[i] = j
                    min_theta = angle

        grope_upper_points = points[np.where(
            self.groupe == self.upper_vector_index[0])]
        grope_upper_normals = normals[np.where(
            self.groupe == self.upper_vector_index[0])]

        # グラフの追加
        ax = self.fig.add_subplot(2, 2, 3, projection='3d')
        self.show_normals(ax, grope_upper_points, grope_upper_normals)

    def main(self) -> None:
        """点群をメッシュ化し、表示する関数."""
        # 点群データの読み込み
        ptCloud = np.load(self.point_file)

        # グラフの追加
        ax1 = self.fig.add_subplot(2, 2, 1, projection='3d')
        self.show_point(ax1, ptCloud)

        # 飛行機の向きを調整
        ptCloud2 = ptCloud.copy()
        for i, point in enumerate(ptCloud2):
            ptCloud2[i] = rotate.rotate_around_x_axis(point, 90, reverse=False)
            ptCloud2[i] = rotate.rotate_around_y_axis(point, 90, reverse=False)

        # グラフの追加
        ax2 = self.fig.add_subplot(2, 2, 2, projection='3d')
        self.show_point(ax2, ptCloud2)

        # 法線ベクトルの作成・編集
        ptCloud = self.edit_normals(ptCloud2)

        plt.show()

        """
        # Poissonサーフェスリコンストラクションを適用
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=9)

        # 表面を可視化
        o3d.visualization.draw_geometries([mesh])
        # """


if __name__ == "__main__":
    import time
    start = time.time()

    # 設定の出力
    line = "-" * len(f"SCRIPT_DIR_PATH  : {PLY_DIR_PATH}")
    print(f"{line}")
    print(f"SCRIPT_DIR_PATH  : {SCRIPT_DIR_PATH}")
    print(f"PROJECT_DIR_PATH : {PROJECT_DIR_PATH}")
    print(f"PLY_DIR_PATH     : {PLY_DIR_PATH}")
    print(f"{line}")

    ply_file_path = os.path.join(PLY_DIR_PATH, "e50_p2048_airplane_01png.npy")
    # ply_file_path = os.path.join(PLY_DIR_PATH, "e50_p2048_airplane2_15png.npy")
    ms = MakeSurface(ply_file_path)
    ms.main()

    # 処理時間計測用
    execute_time = time.time() - start
    print(f"実行時間: {str(execute_time)[:5]}s")

    print("終了")
