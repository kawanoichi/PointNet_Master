"""26方位のベクトルを作成・表示するモジュール
方位の数(-1, 0, 1) の [x, y, z]
26 = 3^3 - 1
※ -1は[0, 0, 0]のパターン
"""
import numpy as np
import os
from matplotlib import pyplot as plt


SCRIPT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR_PATH = os.path.dirname(SCRIPT_DIR_PATH)
PLY_DIR_PATH = os.path.join(PROJECT_DIR_PATH, "predict_points")


def show_vector(vectors, start=[0, 0, 0]):
    """ある点からのベクトルを描画する関数.
    Args:
        vectors: ベクトルの終点座標を格納した配列
        start: ベクトルの始点
    """
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.set(xlabel='x', ylabel='y', zlabel='z')
    ax.axis('equal')  # 正方形に設定

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)

    for vector in vectors:
        ax.quiver(start[0], start[1], start[2],
                  vector[0], vector[1], vector[2],
                  color='blue', length=1, arrow_length_ratio=0.2)
    plt.show()


def angle_between_vectors(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    magnitude_a = np.linalg.norm(vector_a)
    magnitude_b = np.linalg.norm(vector_b)

    cos_theta = dot_product / (magnitude_a * magnitude_b)

    # acosはarccosine関数で、cosの逆関数です。
    theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    # 弧度法から度数法に変換
    theta_deg = np.degrees(theta_rad)

    return theta_deg


def main():
    """点群をメッシュ化し、表示する関数."""
    #                     x   y   z
    vectors = np.array([])
    kinds_of_coodinate = [-1, 0, 1]

    # 26方位のベクトル(終点座標)を作成
    for x in kinds_of_coodinate:
        for y in kinds_of_coodinate:
            for z in kinds_of_coodinate:
                if not x == y == z == 0:
                    append_coordinate = np.array([x, y, z])
                    vectors = np.append(vectors, append_coordinate, axis=0)
    vectors = vectors.reshape((len(kinds_of_coodinate) ^ 3)-1, 3)
    print("vectors.shape", vectors.shape)
    show_vector(vectors)

    comparison_vector = np.array([0,  1,  0])
    for vector in vectors:
        angle = int(angle_between_vectors(comparison_vector, vector))
        print(f"vector1: {comparison_vector}")
        print(f"vector2: {vector} ")
        print(f"between the vectors is: {angle}")


if __name__ == "__main__":
    print("-- 14方位ベクトル表示モジュール --")
    print(f"SCRIPT_DIR_PATH : {SCRIPT_DIR_PATH}")
    print(f"PROJECT_DIR_PATH: {PROJECT_DIR_PATH}")
    print(f"PLY_DIR_PATH    : {PLY_DIR_PATH}")

    main()
    print("終了")
