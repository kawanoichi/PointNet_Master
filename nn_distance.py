"""最近傍探索を行うファイル.

author kawanoichi
"""
import numpy as np
from tqdm import tqdm


class nn_distance():
    """最近傍距離を行うクラス"""

    def __init__(self, data1, data2, search_range=0.1):
        """コンストラクタ.
            data1からdata2の最近傍探索を行う.
        Args:
            data1:座標データ
            data2:座標データ
            search_range:最近傍探索を行う範囲        
        """
        nn_distance.check_data_shape(data1)
        nn_distance.check_data_shape(data2)
        self.__data1 = data1
        self.__data2 = data2
        self.__search_range = search_range  # 正の数
        self.__range_out_count = 0  # 領域内に点が見つからなかった回数をカウント
        self.__nn_dist_list = np.array([])  # 最近傍距離を格納する配列
        self.__nn_dist_average = None  # 最近傍距離の平均
        self.__nn_dist_max = None  # 最近傍距離の最大
        self.__nn_dist_min = None  # 最近傍距離の最小

    @staticmethod
    def check_data_shape(data) -> None:
        """入力データの形状のチェックをする.
        Args:
            data:形状のチェックを行う配列
        """
        if type(data).__module__ != "numpy":
            print("The type of the data is %s. mast be numpy.ndarray" %
                  str(data.shape))
            raise TypeError("Data type is wrong")

        if len(data.shape) != 2:
            print("The shape of the data is %s. mast be (data of numbert, dimnsion)" % str(
                data.shape))
            raise TypeError("Data shape is wrong")

        if data.shape[1] != 3:
            print("The dimension of the data is %d" % int(data[1]))
            raise ValueError("Dimension must be 3")

    def select_point_in_range(self, coordinate) -> np.ndarray:
        """注目点から遠い点を除外する.
        Args:
            coordinate: 注目している点座標
        Returns:
            data: 除外作業を行ったデータ
        """
        # 同じ座標
        data = self.__data2[np.where((self.__data2[:, 0] != coordinate[0])
                                     & (self.__data2[:, 1] != coordinate[1])
                                     & (self.__data2[:, 2] != coordinate[2]))]
        if data is None:
            return None

        # x座標
        data = data[np.where((data[:, 0] - self.__search_range < coordinate[0])
                             & (coordinate[0] < data[:, 0] + self.__search_range))]
        if data is None:
            return None
        # y座標
        data = data[np.where((data[:, 1] - self.__search_range < coordinate[1])
                             & (coordinate[1] < data[:, 1] + self.__search_range))]
        if data is None:
            return None
        # z座標
        data = data[np.where((data[:, 2] - self.__search_range < coordinate[2])
                             & (coordinate[2] < data[:, 2] + self.__search_range))]
        return data

    def cal_nn_distance(self):
        """L2ノルムを求める.

        Returns: 
            float: data1からdata2までのユークリッド距離
        """
        for d in tqdm(self.__data1):
            min_dis = 100
            # 最近傍点の候補を絞る
            data2 = self.select_point_in_range(d)
            if data2 is None:
                self.__range_out_count += 1
                pass

            X = data2 - d

            for x in X:
                dis = np.linalg.norm(x, ord=2)
                if dis < min_dis:
                    min_dis = dis

            # それぞれの最近傍距離を記録
            self.__nn_dist_list = np.append(self.__nn_dist_list, min_dis)

        # 最近傍距離の平均を求める
        self.__nn_dist_average = np.average(self.__nn_dist_list)

        # 最近傍距離の最大を求める
        self.__nn_dist_max = np.max(self.__nn_dist_list)

        # 最近傍距離の最小を求める
        self.__nn_dist_min = np.min(self.__nn_dist_list)

    @property
    def get_nn_dist_average(self) -> float:
        """Getter.

        Returns:
            float: 最近傍距離の平均
        """
        return self.__nn_dist_average

    @property
    def get_nn_dist_max(self) -> float:
        """Getter.

        Returns:
            float: 最近傍距離の最大距離
        """
        return self.__nn_dist_max

    @property
    def get_nn_dist_min(self) -> float:
        """Getter.

        Returns:
            float: 最近傍距離の最小距離
        """
        return self.__nn_dist_min


if __name__ == "__main__":
    # base_path = 'data/shapenet/'
    # file_path = '02691156/1a04e3eab45ca15dd86060f189eb133'
    # point_path = base_path + 'ShapeNet_pointclouds/' + file_path + '/pointcloud_2048.npy'
    # point_data = np.load(point_path)

    coord_file = "predict_points/e50_p_1024_airplane_08.pngpng.npy"
    point_data = np.load(coord_file)
    point_data2 = point_data.copy()

    nn = nn_distance(point_data, point_data2)
    nn.cal_nn_distance()
    print("nn distance average:", nn.get_nn_dist_average)
    print("nn distance max:", nn.get_nn_dist_max)
    print("nn distance min:", nn.get_nn_dist_min)

    print("終了")
