"""拡張子を変換するファイル.
python3 convert_extension.py
"""
import numpy as np
import os

def make_ply(coodinate_file, save_ply):
    # パラメータファイルが存在するかの確認
    if not os.path.isfile(save_ply):
        raise FileNotFoundError("No file '%s'" % save_ply)

    """npファイルをplyファイルに変換する."""
    point_data = np.load(coodinate_file)
    vertices, vertex_indices = np.unique(point_data, return_inverse=True, axis=0)
    # 1024
    # index_triangle = vertex_indices[1:].reshape([-1, 3])
    # 2048
    index_triangle = vertex_indices[2:].reshape([-1, 3])
    # print(index_triangle) 


    """Extract point and face data from .stl file"""

    """write header"""
    # element vertexとelement faceの部分はstlからの値を代入
    header = (
    'ply\n\
    format ascii 1.0\n\
    element vertex {}\n\
    property float x\n\
    property float y\n\
    property float z\n\
    property float quality\n\
    element face {}\n\
    property list uchar int vertex_indices\n\
    end_header\n').format(len(vertices),len(index_triangle)
    )

    with open(save_ply, 'w') as f:
        f.write(header)

    """write text for point data"""
    # y座標をqualityに設定し、縦１列の行列にする
    qualities = np.array([vertices[:, 1]]).T
    # x,y,z座標の後ろにqualityを付け加える
    vertices_text = np.concatenate([vertices, qualities], axis=1)

    # 'a'モードで点情報を上書き追記する
    with open(save_ply, 'a') as f:
        np.savetxt(f, vertices_text, fmt='%.5f')

    """write text for face data"""
    first_indices = np.full([len(index_triangle), 1], 3)
    # 三角形である'3'を先頭に記載してから三角形の点インデックスを記載する
    face_txt =np.concatenate([first_indices, index_triangle], axis=1)

    #上書きで三角形の情報を追記する
    with open(save_ply, 'a') as f:
        np.savetxt(f, face_txt, fmt='%i')


if __name__ == "__main__":    
    # coodinate_file = "predict_points/e50_p1024_lamp_01png.npy"
    # coodinate_file = "predict_points/e50_p1024_airplane_01png.npy"
    coodinate_file = "predict_points/e50_p2048_airplane_01png.npy"
    save_file = coodinate_file[:-3]+"ply"
    make_ply(coodinate_file, save_file)

    print("終了")