"""Ransac処理を行う関数のバックアップ
うまくいかなかったが、念のために残しておく
MakeSurfaceにそのまま追加するだけでクラス関数として使用できるはず
"""

import numpy as np
import os
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

SCRIPT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR_PATH = os.path.dirname(SCRIPT_DIR_PATH)
PLY_DIR_PATH = os.path.join(PROJECT_DIR_PATH, "predict_points")


def ransac_2d(self, data, title="Ransac 2d"):
    """面を生成するRANSACを実行する関数(2D).
    Args:
        data(np.ndarray): 点群
    """
    x = data[:, 0]
    y = data[:, 1]
    # RANSACRegressorを設定
    model = RANSACRegressor(min_samples=20,
                            residual_threshold=0.1,
                            max_trials=200)

    # モデルを適合
    model.fit(data[:, 0].reshape(-1, 1), data[:, 1])

    # 推定されたモデルのパラメータ
    inlier_mask = model.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    ax = self.fig.add_subplot(self.fig_vertical,
                              self.fig_horizontal,
                              self.graph_num)
    self.graph_num += 1

    plt.title(title)
    plt.scatter(x[inlier_mask], y[inlier_mask],
                color='blue', marker='.', label='Inliers')
    plt.scatter(x[outlier_mask], y[outlier_mask],
                color='red', marker='.', label='Outliers')
    plt.plot(x, model.predict(x.reshape(-1, 1)),
             color='green', linewidth=2, label='RANSAC Model')
    plt.legend(loc='lower right')
    ax.set(xlabel='x', ylabel='y')


def ransac_3d(self, data, title="Ransac 3d"):
    """面を生成するRANSACを実行する関数(3D).
    Args:
        data(np.ndarray): 点群
    """
    # RANSACRegressorを設定
    model = RANSACRegressor()

    # モデルを適合
    model.fit(data[:, :2], data[:, 2])

    # 推定されたモデルのパラメータ
    inlier_mask = model.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    # 結果の可視化
    plt.title(title)
    ax = self.fig.add_subplot(self.fig_vertical,
                              self.fig_horizontal,
                              self.graph_num,
                              projection='3d')
    self.graph_num += 1
    ax.scatter(data[inlier_mask, 0], data[inlier_mask, 1],
               data[inlier_mask, 2], color='blue', marker='.', label='Inliers')
    ax.scatter(data[outlier_mask, 0], data[outlier_mask, 1],
               data[outlier_mask, 2], color='red', marker='.', label='Outliers')

    # 推定された平面を可視化
    xx, yy = np.meshgrid(np.linspace(data[:, 0].min(), data[:, 0].max(), 10),
                         np.linspace(data[:, 1].min(), data[:, 1].max(), 10))
    zz = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.plot_surface(xx, yy, zz, alpha=0.5,
                    color='green', label='RANSAC Model')

    ax.set(xlabel='x', ylabel='y', zlabel='z')
    ax.legend()
