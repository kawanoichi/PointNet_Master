import numpy as np


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


# 例として、ベクトルAとベクトルBを定義
vector_a = np.array([1, 2, 3])
vector_b = np.array([4, 5, 6])

# なす角を計算
angle = angle_between_vectors(vector_a, vector_b)

print(f"The angle between the vectors is: {angle} degrees")
