import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from plyfile import PlyData, PlyElement
import open3d as o3d

from information import PLY_DIR_PATH


# PLYファイルを読み込む
def read_ply(file_path):
    with open(file_path, 'rb') as f:
        plydata = PlyData.read(f)

    # PLYファイルのデータ構造に合わせて変更
    if 'vertex' in plydata:
        return plydata['vertex'].data
    elif 'vertices' in plydata:
        return plydata['vertices'].data
    else:
        raise ValueError('PLY file does not contain vertex data.')
    

# マーチングキューブ法による表面抽出
def marching_cubes(data, threshold=0):
    verts, faces, _, _ = measure.marching_cubes(data, threshold)
    return verts, faces

# PLYファイルを保存する
def save_ply(file_path, verts, faces):
    vertex = np.array([(v[0], v[1], v[2]) for v in verts], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    face = np.array([(f[0], f[1], f[2]) for f in faces], dtype=[('vertex_indices', 'i4', (3,))])

    el_verts = PlyElement.describe(vertex, 'vertex')
    el_faces = PlyElement.describe(face, 'face')

    PlyData([el_verts, el_faces], text=True).write(file_path)


def main(ply_file_path, output_ply_path):
    # PLYファイルを読み込む
    ptCloud = o3d.io.read_point_cloud(ply_file_path)
    # >>> PointCloud with 8 points.

    np_ptcloud = np.asarray(ptCloud.points)

    # PLYファイルから取得したデータが3DのNumPy配列であることを確認
    if np_ptcloud.shape[1] != 3:
        raise ValueError('Input mesh should be a 3D numpy array.')

    # 3Dデータの取得（ここではx, y, z座標のみを取得しています）
    data = np.array([(v[0], v[1], v[2]) for v in np_ptcloud], dtype=float)
    print("type(data)", type(data))
    print("data.shape", data.shape)


    # マーチングキューブ法で表面抽出
    verts, faces = marching_cubes(data)

    # 結果の描画
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='viridis')

    # PLYファイルに保存
    save_ply(output_ply_path, verts, faces)

    plt.show()

if __name__ == "__main__":
    ply_file = "point_cloud.ply"
    ply_file_path = os.path.join(PLY_DIR_PATH, ply_file)
    output_ply_path = os.path.join(PLY_DIR_PATH, "marching_"+ply_file)
    main(ply_file_path, output_ply_path)
    print("終了")
