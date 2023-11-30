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
import open3d as o3d
import os

SCRIPT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR_PATH = os.path.dirname(SCRIPT_DIR_PATH)
PLY_DIR_PATH = os.path.join(PROJECT_DIR_PATH, "predict_points")


class MakeSurface:
    def __init__(self, point_file):
        self.point_file = point_file

        if not os.path.isfile(self.point_file):
            raise FileNotFoundError("No file '%s'" % self.point_file)

    def make_surface(self):
        """点群をメッシュ化し、表示する関数."""
        # 点群データの読み込み
        ptCloud = o3d.io.read_point_cloud(self.point_file)

        # 法線の計算
        # radius:半径, max_nn:検索を行う近隣の数
        ptCloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # 点の法線の方向一貫性の考慮
        ptCloud.orient_normals_consistent_tangent_plane(60)

        # 近傍距離を計算
        distances = ptCloud.compute_nearest_neighbor_distance()

        # 法線の表示
        # o3d.visualization.draw_geometries([ptCloud], point_show_normal=True)

        # 近傍距離の平均
        avg_dist = np.mean(distances)

        # 半径
        radius = 2*avg_dist

        # [半径,直径]
        radii = [radius, radius * 2]

        # 三角形メッシュを計算する
        # o3d.utility.DoubleVector:numpy配列をopen3D形式に変換
        recMeshBPA = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            ptCloud, o3d.utility.DoubleVector(radii))

        # geometry.Geometry オブジェクトのリストを描画する関数
        o3d.visualization.draw_geometries([recMeshBPA])


if __name__ == "__main__":
    print(f"SCRIPT_DIR_PATH:  {SCRIPT_DIR_PATH}")
    print(f"PROJECT_DIR_PATH: {PROJECT_DIR_PATH}")
    print(f"PLY_DIR_PATH:     {PLY_DIR_PATH}")

    ply_file_path = os.path.join(PLY_DIR_PATH, "e50_p2048_airplane2_15png.ply")
    ms = MakeSurface(ply_file_path)
    ms.make_surface()
    print("終了")
