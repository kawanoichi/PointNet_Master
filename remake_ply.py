"""plyファイルのメッシュの再構築を行う.

面の頂点インデックスがバラバラなときに使用する。
実行コマンド
$ python3 remake_ply.py -m point_cloud.ply
"""

import pyvista as pv
import numpy as np
import argparse


def remake_ply(path):
    """メッシュの再構築を行う"""
    save_path = "remake_" + path

    # PLYファイルから頂点と面のデータを読み取ります。
    mesh = pv.read(path)

    # 頂点座標を取得します。
    vertices = mesh.points

    # 面の頂点インデックスを取得します。
    faces = mesh.faces.reshape((-1, 4))[:, 1:]
    print("1")
    # 頂点インデックスを正しい順序に並べ替えます。
    # ここでは、隣接する面の法線ベクトルを計算して、頂点の向きを修正しています。
    normals = mesh.cell_normals
    for i, face in enumerate(faces):
        normal = normals[i]
        centroid = np.mean(vertices[face], axis=0)
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]
        cross_product = np.cross(v1 - v0, v2 - v0)
        dot_product = np.dot(normal, cross_product)
        if dot_product < 0:
            faces[i] = face[::-1]
    print("2")

    # 新しいメッシュを作成します。
    new_mesh = pv.PolyData(vertices, faces)
    print("3")

    # 新しいメッシュをPLYファイルとして出力します。
    print("save_path : ", save_path)
    new_mesh.save(save_path)  # コアダンプしてしまう。
    print("4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用例\n"
                                                 " 指定した.plyファイルのメッシュのメッシュの差構築を行う\n"
                                                 " $ python remake_ply.py -mesh point_cloud.ply\n",
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("-m", "--mesh", type=str, help="メッシュ化を行う")
    args = parser.parse_args()

    # 指定していないときは'point_cloud.ply'のメッシュ化を行う
    if args.mesh is None:
        remake_ply("メッシュの再構築を行いたいファイルを指定してください")
        exit(0)

    remake_ply(args.mesh)

    print("終了")
