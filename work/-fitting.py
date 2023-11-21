import numpy as np


def fit_plane(point_cloud):
    """
    入力
        point_cloud : xyzのリスト　numpy.array型
    出力
        plane_v : 法線ベクトルの向き(単位ベクトル)
        com : 重心　近似平面が通る点
    """

    com = np.sum(point_cloud, axis=0) / len(point_cloud)
    # 重心を計算
    q = point_cloud - com
    # 重心を原点に移動し、同様に全点群を平行移動する  pythonのブロードキャスト機能使用
    Q = np.dot(q.T, q)
    # 3x3行列を計算する 行列計算で和の形になるため総和になる
    la, vectors = np.linalg.eig(Q)
    # 固有値、固有ベクトルを計算　固有ベクトルは縦のベクトルが横に並ぶ形式
    plane_v = vectors.T[np.argmin(la)]
    # 固有値が最小となるベクトルの成分を抽出

    return plane_v, com
