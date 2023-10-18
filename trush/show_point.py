from matplotlib import pyplot as plt


def show_point(ptCloud) -> None:
    """点群を表示する関数.

    Args:
        ptCloud(np.ndarray): 点群
    """
    # figureを生成
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.set(xlabel='x', ylabel='y', zlabel='z')
    ax.scatter(ptCloud[:, 0],
               ptCloud[:, 1],
               ptCloud[:, 2],
               c='b')

    plt.show()


def show_normals(ptCloud, normals) -> None:
    """点群と法線ベクトルを表示する関数.

    Args:
        ptCloud(np.ndarray): 点群
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set(xlabel='x', ylabel='y', zlabel='z')

    # 点をプロット
    ax.scatter(ptCloud[:, 0], ptCloud[:, 1],
               ptCloud[:, 2], c='b', marker='o', label='Points')

    # 法線ベクトルをプロット
    scale = 0.1  # 矢印のスケール
    for i in range(len(ptCloud)):
        if ptCloud[i, 0] < -0.05:
            ax.quiver(ptCloud[i, 0], ptCloud[i, 1], ptCloud[i, 2],
                      normals[i, 0]*scale, normals[i, 1]*scale, normals[i, 2]*scale, color='r', length=1.0, normalize=True)

    ax.legend()

    plt.show()
