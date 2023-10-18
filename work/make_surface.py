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

        # 画像の存在チェック
        if not os.path.isfile(self.point_file):
            raise FileNotFoundError("No file '%s'" % self.point_file)

    def vector_26(self):
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
        print("self.vectors_26.shape", self.vectors_26.shape)

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

    def show_normals(self, ptCloud, normals) -> None:
        """点群と法線ベクトルを表示する関数.

        Args:
            ptCloud(np.ndarray): 点群
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 点をプロット
        ax.scatter(ptCloud[:, 0], ptCloud[:, 1],
                   ptCloud[:, 2], c='b', marker='o', label='Points')

        # 法線ベクトルをプロット
        scale = 0.1  # 矢印のスケール
        for i in range(len(ptCloud)):
            if ptCloud[i, 0] < -0.05:
                ax.quiver(ptCloud[i, 0], ptCloud[i, 1], ptCloud[i, 2],
                          normals[i, 0]*scale, normals[i, 1]*scale, normals[i, 2]*scale, color='r', length=1.0, normalize=True)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

        plt.show()

    def edit_normals(self, points: np.ndarray) -> None:
        """法線ベクトルを表示する関数.

        Args:
            points(np.ndarray): 点群
        """
        # Open3DのPointCloudに変換
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        print("point_cloud\n", point_cloud.points)

        # 法線情報を計算
        point_cloud.estimate_normals()

        # 法線情報にアクセス
        normals = np.asarray(point_cloud.normals)
        print("normals.shape", normals.shape)

        # 類似性の閾値
        threshold = 0.9

        # クラスタリングを行う
        labels = []
        current_label = 0

        for i in range(len(normals)):
            if i == 0:
                labels.append(current_label)
            else:
                # 類似性の計算（内積）
                similarity = np.dot(normals[i], normals[i - 1])
                if i < 10:
                    print(similarity)

                # 閾値を超えたら新しいクラスタとしてラベルを増やす
                if similarity < threshold:
                    current_label += 1
                labels.append(current_label)

        print("number of labels is ", len(labels))

        for i, label in enumerate(labels):
            if i < 10:
                print(label)
        """
        # クラスタごとに点を分ける
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(points[i])

        # 各クラスタの点群を表示
        for label, cluster_points in clusters.items():
            cluster_points = np.asarray(cluster_points)
            cluster_cloud = o3d.geometry.PointCloud()
            cluster_cloud.points = o3d.utility.Vector3dVector(cluster_points)
            o3d.visualization.draw_geometries([cluster_cloud])
        # """

    def main(self) -> None:
        """点群をメッシュ化し、表示する関数."""
        # 点群データの読み込み
        ptCloud = np.load(self.point_file)
        print("ptCloud.shape", ptCloud.shape)
        # print("ptCloud.shape", type(ptCloud))

        # 入力点群を表示する
        # self.show_point(ptCloud)

        # 飛行機の向きを調整
        ptCloud2 = ptCloud.copy()
        for i, point in enumerate(ptCloud2):
            ptCloud2[i] = rotate.rotate_around_x_axis(point, 90, reverse=False)
            ptCloud2[i] = rotate.rotate_around_y_axis(point, 90, reverse=False)

        # 入力点群を表示する
        # self.show_point(ptCloud2)

        # 法線ベクトルの作成・編集
        # ptCloud = self.edit_normals(points=ptCloud)

        self.vector_26()

        """
        # Poissonサーフェスリコンストラクションを適用
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=9)

        # 表面を可視化
        o3d.visualization.draw_geometries([mesh])
        # """


if __name__ == "__main__":
    print(f"SCRIPT_DIR_PATH : {SCRIPT_DIR_PATH}")
    print(f"PROJECT_DIR_PATH: {PROJECT_DIR_PATH}")
    print(f"PLY_DIR_PATH    : {PLY_DIR_PATH}")

    ply_file_path = os.path.join(PLY_DIR_PATH, "e50_p2048_airplane_01png.npy")
    # ply_file_path = os.path.join(PLY_DIR_PATH, "e50_p2048_airplane2_15png.npy")
    ms = MakeSurface(ply_file_path)
    ms.main()
    print("終了")
