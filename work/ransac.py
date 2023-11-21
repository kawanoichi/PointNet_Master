import numpy as np
import os
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

SCRIPT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR_PATH = os.path.dirname(SCRIPT_DIR_PATH)
PLY_DIR_PATH = os.path.join(PROJECT_DIR_PATH, "predict_points")


def make_data():
    # ダミーの3D点群データを生成
    np.random.seed(42)
    x = 10 * np.random.rand(100)
    y = 2 * x - 1 + np.random.normal(0, 1, 100)
    z = 3 * x + 5 * np.random.normal(0, 1, 100)

    # ノイズを追加
    y[::2] += 10 * np.random.normal(0, 1, 50)

    # 3D点群データを整形
    data = np.column_stack((x, y, z))
    print(f"data.shape: {data.shape}")

    return data


def ransac(data):
    # RANSACRegressorを設定
    model = RANSACRegressor()

    # モデルを適合
    model.fit(data[:, :2], data[:, 2])

    # 推定されたモデルのパラメータ
    inlier_mask = model.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    # 結果の可視化
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[inlier_mask, 0], data[inlier_mask, 1],
               data[inlier_mask, 2], color='blue', marker='.', label='Inliers')
    ax.scatter(data[outlier_mask, 0], data[outlier_mask, 1],
               data[outlier_mask, 2], color='red', marker='.', label='Outliers')

    # 推定された平面を可視化
    xx, yy = np.meshgrid(np.linspace(data[:, 0].min(), data[:, 0].max(), 10),
                         np.linspace(data[:, 1].min(), data[:, 1].max(), 10))
    zz = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.plot_surface(xx, yy, zz, alpha=0.5, color='green', label='RANSAC Model')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()


if __name__ == "__main__":
    # data = make_data()
    file_name = "e50_p2048_airplane_01png.npy"

    # 点群データの読み込み
    point_path = os.path.join(PLY_DIR_PATH, file_name)

    # 画像の存在チェック
    if not os.path.isfile(point_path):
        raise FileNotFoundError("No file '%s'" % point_path)

    # 点群データの読み込み
    data = np.load(point_path)

    # 面推測
    ransac(data)
