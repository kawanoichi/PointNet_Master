import numpy as np
import itertools
import matplotlib.pyplot as plt
import os


class KMeans:
    def __init__(self, n_clusters, max_iter=1000, random_seed=0):
        """
        KMeans clustering algorithm.

        Parameters:
        - n_clusters: int, number of clusters
        - max_iter: int, maximum number of iterations
        - random_seed: int, random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = np.random.RandomState(random_seed)

    def fit(self, X):
        """
        Fit the KMeans model to the input data.

        Parameters:
        - X: numpy array, input data
        """
        cycle = itertools.cycle(range(self.n_clusters))
        self.labels_ = np.fromiter(
            itertools.islice(cycle, X.shape[0]), dtype=int)
        self.random_state.shuffle(self.labels_)
        labels_prev = np.zeros(X.shape[0])
        count = 0
        self.cluster_centers_ = np.zeros((self.n_clusters, X.shape[1]))

        while (not (self.labels_ == labels_prev).all() and count < self.max_iter):
            for i in range(self.n_clusters):
                cluster_points = X[self.labels_ == i, :]
                self.cluster_centers_[i, :] = cluster_points.mean(axis=0)

            dist = np.sum(
                (X[:, :, np.newaxis] - self.cluster_centers_.T[np.newaxis, :, :]) ** 2, axis=1)
            labels_prev = self.labels_
            self.labels_ = dist.argmin(axis=1)
            count += 1

    def predict(self, X):
        """
        Predict the cluster labels for the input data.

        Parameters:
        - X: numpy array, input data

        Returns:
        - labels: numpy array, predicted cluster labels
        """
        dist = np.sum(
            (X[:, :, np.newaxis] - self.cluster_centers_.T[np.newaxis, :, :]) ** 2, axis=1)
        labels = dist.argmin(axis=1)
        return labels


if __name__ == "__main__":
    np.random.seed(0)
    points1 = np.random.randn(80, 2)
    points2 = np.random.randn(80, 2) + np.array([4, 0])
    points3 = np.random.randn(80, 2) + np.array([5, 8])

    points = np.r_[points1, points2, points3]
    np.random.shuffle(points)

    cluster_num =
    model = KMeans(cluster_num)
    model.fit(points)

    print(model.labels_)

    markers = ["+", "*", "o", "s"]
    colors = ['red', 'blue', 'green', "purple"]
    for i in range(3):
        cluster_points = points[model.labels_ == i, :]
        plt.scatter(cluster_points[:, 0], cluster_points[:,
                    1], marker=markers[i], color=colors[i])

    save_dir_path = "zikken_data"
    save_path = os.path.join(
        save_dir_path, "k-mean_" + str(cluster_num) + ".png")
    plt.savefig(save_path)
    print("完了")
