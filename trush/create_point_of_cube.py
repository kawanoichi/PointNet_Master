"""立方体を表すplyファイルを作成するモジュール."""
import os
from information import PLY_DIR_PATH


def create_cube_ply(file_path):
    """mesh(表面情報)を含んだplyファイルの作成"""
    vertices = [
        (-1, -1, -1),  # 0
        (1, -1, -1),   # 1
        (1, 1, -1),    # 2
        (-1, 1, -1),   # 3
        (-1, -1, 1),   # 4
        (1, -1, 1),    # 5
        (1, 1, 1),     # 6
        (-1, 1, 1)     # 7
    ]

    faces = [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (2, 3, 7, 6),
        (0, 4, 7, 3),
        (1, 5, 6, 2)
    ]

    with open(file_path, 'w') as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write("element vertex {}\n".format(len(vertices)))
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("element face {}\n".format(len(faces)))
        file.write("property list uchar int vertex_indices\n")
        file.write("end_header\n")

        # Write vertices
        for vertex in vertices:
            file.write("{} {} {}\n".format(vertex[0], vertex[1], vertex[2]))

        # Write faces
        for face in faces:
            file.write("4 {} {} {} {}\n".format(
                face[0], face[1], face[2], face[3]))


def create_point_cloud_ply(file_path):
    """点群情報だけのplyファイルの作成."""
    vertices = [
        (-1, -1, -1),  # 0
        (1, -1, -1),   # 1
        (1, 1, -1),    # 2
        (-1, 1, -1),   # 3
        (-1, -1, 1),   # 4
        (1, -1, 1),    # 5
        (1, 1, 1),     # 6
        (-1, 1, 1)     # 7
    ]

    with open(file_path, 'w') as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write("element vertex {}\n".format(len(vertices)))
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("end_header\n")

        # Write vertices
        for vertex in vertices:
            file.write("{} {} {}\n".format(vertex[0], vertex[1], vertex[2]))


if __name__ == "__main__":
    print("開始")

    # メッシュ情報を含んだ立方体のplyファイルを作成
    file_path = os.path.join(PLY_DIR_PATH, "cube.ply")
    create_cube_ply(file_path)

    # 立方体の点群情報だけのplyファイルを作成
    file_path = os.path.join(PLY_DIR_PATH, "point_cloud.ply")
    create_point_cloud_ply(file_path)

    print("完了")
