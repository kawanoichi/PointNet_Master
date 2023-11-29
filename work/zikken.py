import numpy as np
import open3d as o3d
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import cv2

# from pyntcloud import PyntCloud
from scipy.spatial import Delaunay


SCRIPT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR_PATH = os.path.dirname(SCRIPT_DIR_PATH)
PLY_DIR_PATH = os.path.join(PROJECT_DIR_PATH, "predict_points")


def create_mesh():
    # # npyファイルから点群データを読み込む
    # point_cloud_data = np.load("path/to/point_cloud.npy")

    # # Numpy配列からOpen3Dの点群オブジェクトを作成
    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(point_cloud_data)

    # # メッシュ生成（Poisson Surface Reconstructionを使用）
    # mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud)

    # # メッシュの保存
    # o3d.io.write_triangle_mesh("path/to/output_mesh.ply", mesh)

    # 仮の点群データを生成
    np.random.seed(0)
    points = np.random.rand(1024, 3)

    # デラウン三角形分割を行い、メッシュ情報を生成
    triangulation = Delaunay(points[:, :2])
    faces = triangulation.simplices

    # 法線ベクトルの生成（仮のデータなので簡単に単位法線）
    normals = np.cross(points[faces[:, 1]] - points[faces[:, 0]], points[faces[:, 2]] - points[faces[:, 0]])
    normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]

    # 座標と法線ベクトルをopen3dの形式に変換
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)

    # PLYファイルに保存
    save_path = os.path.join(PROJECT_DIR_PATH, "ply_data", 'zikken.ply')
    o3d.io.write_triangle_mesh(save_path, mesh, write_ascii=True)

def zikken():
    plt.ioff() 
    # 画像の読み込み
    img = cv2.imread('work/test.png') 
    # カラーデータの色空間の変換 
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) 

    # 画像の表示
    plt.show()


if __name__ == "__main__":
    print("実験")
    # create_mesh()
    zikken()
    print("完了")
    




