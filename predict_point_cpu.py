"""学習済みモデルを使用して画像から3Dオブジェクトを生成する.
https://github.com/yonghanzhang94/A-Single-View-3D-Object-Point-Cloud-Reconstruction/blob/master/show_shapenet.py
をもとに作成
@author kawanoichi
実行コマンド
$ python3 predict_point_shapenet.py
"""
import json
import os
import cv2
import numpy as np
import torch
from torch.autograd import Variable
from model import generator

SCRIPT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))


class Predict_Point:
    """画像から3Dオブジェクトを生成するクラス."""

    def __init__(self, predict_param_file):
        # パラメータpath
        self.__predict_param_file = predict_param_file
        # 学習パラメータの読み込み
        self.load_param()

    def check_param(self, key):
        """paramにkeyがあるかチェックする."""
        with open(self.__predict_param_file) as fp:
            self.predict_param = json.load(fp)
        if key not in self.predict_param:
            raise KeyError("key not found: '%s', file: %s" %
                           (key, self.__predict_param_file))

    def load_param(self) -> None:
        """インスタンス変数としてパラメータファイルを読み込む.
        Raises:
            FileNotFoundError: パラメータファイルが見つからない場合に発生
            KeyError: パラメータファイルの内容に期待するキーのデータが無い場合に発生
        """
        # パラメータファイルが存在するかの確認
        if not os.path.isfile(self.__predict_param_file):
            raise FileNotFoundError("No file '%s'" % self.__predict_param_file)

        with open(self.__predict_param_file) as fp:
            predict_param = json.load(fp)

        key = "base_path"
        self.check_param(key)
        self.__base_path = predict_param[key]

        key = "outfolder"
        self.check_param(key)
        self.__outfolder = predict_param[key]

        key = "learned_model"
        self.check_param(key)
        self.__learned_model = predict_param[key]

        key = "nepoch"
        self.check_param(key)
        self.__nepoch = predict_param[key]

        key = "num_points"
        self.check_param(key)
        self.__num_points = predict_param[key]

        key = "category_id"
        self.check_param(key)
        self.__category_id = predict_param[key]

        key = "object_id"
        self.check_param(key)
        self.__object_id = predict_param[key]

        key = "image_name"
        self.check_param(key)
        self.__image_name = predict_param[key]

        key = self.__category_id
        self.check_param(key)
        self.__category = predict_param[key]

        key = "pre_save_folder"
        self.check_param(key)
        self.__pre_save_folder = predict_param[key]

        key = "gt_save_folder"
        self.check_param(key)
        self.__gt_save_folder = predict_param[key]

        print("##PARAMETER", "-"*39)
        print("*base_path       :", self.__base_path)
        print("*outfolder       :", self.__outfolder)
        print("*learned_model   :", self.__learned_model)
        print("*nepoch          :", self.__nepoch)
        print("*num_points      :", self.__num_points)
        print("*category_id     :", self.__category_id)
        print("*object_id       :", self.__object_id)
        print("*image_name      :", self.__image_name)
        print("*category        :", self.__category)
        print("*pre_save_folder :", self.__pre_save_folder)
        print("*gt_save_folder  :", self.__gt_save_folder)
        print("-"*50)

    def predict(self):
        """学習済みモデルを使用して画像から点群を生成する."""
        read_img_path = os.path.join(
            SCRIPT_DIR_PATH, "test_image_airplane.png")
        if not os.path.exists(read_img_path):
            print("Error: image is not exist")
            print("Search path is ", read_img_path)
            exit()

        # 学習済みモデルpath
        pickle_path = os.path.join(".",
                                   self.__outfolder,
                                   self.__category + "-" +
                                   str(self.__num_points),
                                   self.__learned_model)

        # 予測点群の保存path
        try:
            os.makedirs(self.__pre_save_folder)
        except OSError:
            pass
        save_name = "point_data.npy"
        pre_save_path = os.path.join(self.__pre_save_folder, save_name)
        print("pre_save_path", pre_save_path)

        # 画像読み込み(128, 128, 3)
        image = cv2.imread(read_img_path)
        image = cv2.resize(image, (128, 128))
        image = image[4:-5, 4:-5, :3]

        # BGRからRGBへの変換
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 配列の形状を変更
        image = np.transpose(image, (2, 0, 1))

        # Tensorクラスのインスタンス化
        image = torch.Tensor(image)

        # 指定した位置にサイズ1の次元を挿入する
        # torch.Size([3, 128, 128]) >>> torch.Size([1, 3, 128, 128])
        image = image.unsqueeze(0)

        # generatorクラスのインスタンス化
        gen = generator(self.__num_points, use_gpu=False)

        gen.eval()

        # 学習済みモデルの読み込み
        with open(pickle_path, "rb") as f:
            gen.load_state_dict(torch.load(f, map_location=torch.device('cpu')))

        # torch.Tensorに計算グラフの情報を保持させる
        image = Variable(image.float())

        # 点群生成
        points, _, _, _ = gen(image)

        points = points.detach().numpy()

        # (1, 3, 1024) >>> (3, 1024)
        points = np.squeeze(points)

        # (3, 1024) >>> (3, 1024)
        predict_points = np.transpose(points, (1, 0))

        # 予測座標の保存
        np.save(pre_save_path, predict_points)


if __name__ == "__main__":
    # パラメータファイルの宣言
    predict_param_file = "predict_point_param.json"

    # 点群予測クラスのインスタンス化
    pp = Predict_Point(predict_param_file)

    # 点群予測関数の実行
    pp.predict()

    print("終了")
