"""学習済みモデルを使用して画像から3Dオブジェクトを生成する.
https://github.com/yonghanzhang94/A-Single-View-3D-Object-Point-Cloud-Reconstruction/blob/master/show_shapenet.py
をもとに作成
@author kawanoichi
実行コマンド
$ python3 predict_point_shapenet.py
"""
import json
import os
import torch.backends.cudnn as cudnn
import cv2
import matplotlib.pylab as plt
import numpy as np
import torch
from torch.autograd import Variable
from model import generator


class Predict_Point:
    """画像から3Dオブジェクトを生成するクラス."""
    def __init__(self, predict_param_file):
        # パラメータpath
        self.__predict_param_file = predict_param_file
        #学習パラメータの読み込み
        self.load_param()

    def load_param(self)->None:
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
        if key not in predict_param:
            raise KeyError("key not found: '%s', file: %s"%(key, self.__predict_param_file))
        self.__base_path = predict_param[key]

        key = "outfolder"
        if key not in predict_param:
            raise KeyError("key not found: '%s', file: %s"%(key, self.__predict_param_file))
        self.__outfolder = predict_param[key]

        key = "learned_model"
        if key not in predict_param:
            raise KeyError("key not found: '%s', file: %s"%(key, self.__predict_param_file))
        self.__learned_model = predict_param[key]

        key = "nepoch"
        if key not in predict_param:
            raise KeyError("key not found: '%s', file: %s"%(key, self.__predict_param_file))
        self.__nepoch = predict_param[key]

        key = "num_points"
        if key not in predict_param:
            raise KeyError("key not found: '%s', file: %s"%(key, self.__predict_param_file))
        self.__num_points = predict_param[key]

        key = "category_id"
        if key not in predict_param:
            raise KeyError("key not found: '%s', file: %s"%(key, self.__predict_param_file))
        self.__category_id = predict_param[key]        

        key = "object_id"
        if key not in predict_param:
            raise KeyError("key not found: '%s', file: %s"%(key, self.__predict_param_file))
        self.__object_id = predict_param[key]

        key = "image_name"
        if key not in predict_param:
            raise KeyError("key not found: '%s', file: %s"%(key, self.__predict_param_file))
        self.__image_name = predict_param[key]

        key = self.__category_id
        if key not in predict_param:
            raise KeyError("key not found: '%s', file: %s"%(key, self.__predict_param_file))
        self.__category = predict_param[key]

        key = "pre_save_folder"
        if key not in predict_param:
            raise KeyError("key not found: '%s', file: %s"%(key, self.__predict_param_file))
        self.__pre_save_folder = predict_param[key]

        key = "gt_save_folder"
        if key not in predict_param:
            raise KeyError("key not found: '%s', file: %s"%(key, self.__predict_param_file))
        self.__gt_save_folder = predict_param[key]

        print("##PARAMETER","-"*39)
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

    def predict(self, show:bool = False):
        """学習済みモデルを使用して画像から点群を生成する."""
        # 入力画像path
        file_path = os.path.join(self.__category_id, self.__object_id)
        read_img_path = os.path.join(self.__base_path, 
                                    "ShapeNetRendering", 
                                    file_path, 
                                    "rendering", 
                                    self.__image_name)
        
        # グランドトゥルースpath
        read_gt_path = os.path.join(self.__base_path, 
                                "ShapeNet_pointclouds", 
                                file_path, 
                                "pointcloud_"+str(self.__num_points)+".npy")
        # 学習済みモデルpath
        pickle_path = os.path.join(".",
                                    self.__outfolder, 
                                    self.__category + "-" + str(self.__num_points), 
                                    self.__learned_model)

        # 予測点群の保存path
        try:
            os.makedirs(self.__pre_save_folder)
        except OSError:
            pass
        save_name = "e%d_p%d_%s_%spng.npy"%(self.__nepoch, self.__num_points, self.__category, self.__image_name[:-4])
        pre_save_path = os.path.join(self.__pre_save_folder, save_name)

        # gtの保存path
        try:
            os.makedirs(self.__gt_save_folder)
        except OSError:
            pass
        save_name = "pointcloud_%d_%s.asc"%(self.__num_points, self.__category)
        gt_save_path = os.path.join(self.__gt_save_folder, save_name)
        print("gt_save_path", gt_save_path)

        # gtの保存
        gt = (np.load(read_gt_path)).astype('float32')
        np.savetxt(gt_save_path, gt)
        # """
        cudnn.benchmark = True

        # 画像読み込み(128, 128, 3)
        image = cv2.imread(read_img_path)[4:-5, 4:-5, :3]

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
        gen = generator(self.__num_points)

        # eval():評価モード
        gen.cuda().eval()

        # 学習済みモデルの読み込み
        with open(pickle_path, "rb") as f:
            gen.load_state_dict(torch.load(f))

        # torch.Tensorに計算グラフの情報を保持させる
        image = Variable(image.float())
        image = image.cuda()

        # 点群生成
        points, _, _, _ = gen(image)
        points = points.cpu().detach().numpy()

        # (1, 3, 1024) >>> (3, 1024)
        points = np.squeeze(points)

        # (3, 1024) >>> (3, 1024)
        predict_points = np.transpose(points, (1, 0))

        # 予測座標の保存
        np.save(pre_save_path, predict_points)
        np.savetxt(pre_save_path[:-3]+"asc", predict_points)

        # showがTrueの場合、予測点群を表示する
        if show:
            Predict_Point.show_points(predict_points)    
        # """

    # @staticmethod
    # def show_points(predict_points, read_gt_path=None):
    #     """生成した点群オブジェクトを表示する."""
    #     print("type(predict_points)", type(predict_points))
    #     fig = plt.figure()
    #     fig.suptitle("PREDICT")
    #     ax = fig.gca(projection="3d")
    #     ax.set_xlim(-0.5, 0.5)
    #     ax.set_ylim(-0.5, 0.5)
    #     ax.set_zlim(-0.5, 0.5)
    #     for i in range(len(predict_points)):
    #         ax.scatter(predict_points[i, 1], predict_points[i, 2], predict_points[i, 0], c = "blue", depthshade=True, s=1)
    #         # ax.scatter(predict_points[i, 1], predict_points[i, 2], predict_points[i, 0], c="#00008B", depthshade=True, s=10)
    #     ax.axis("off")
    #     ax.view_init(azim=90, elev=-160)
    #     plt.show()

    #     if not read_gt_path is None:
    #         fig = plt.figure()
    #         fig.suptitle("GT")
    #         ax2 = fig.gca(projection="3d")
    #         ax2.set_xlim(-0.5, 0.5)
    #         ax2.set_ylim(-0.5, 0.5)
    #         ax2.set_zlim(-0.5, 0.5)
    #         for i in range(len(read_gt_path)):
    #             ax2.scatter(read_gt_path[i, 1], read_gt_path[i, 2], read_gt_path[i, 0], c="#00008B", depthshade=True)
    #         ax2.axis("off")
    #         ax2.view_init(azim=90, elev=-160)
    #         plt.show()

    # @staticmethod
    # def load_points(coord_file:str)->None:
    #     """保存されて座標から点群を生成し、表示する."""
    #     # パラメータファイルが存在するかの確認
    #     if not os.path.isfile(coord_file):
    #         raise FileNotFoundError("No file '%s'" % coord_file)

    #     # 点群座標の読み込み
    #     points = np.load(coord_file)

    #     # 点群を表示
    #     Predict_Point.show_points(points)



if __name__ == "__main__":
    # パラメータファイルの宣言
    predict_param_file = "predict_point_param.json"
    
    # 点群予測クラスのインスタンス化
    pp = Predict_Point(predict_param_file)

    # 点群予測関数の実行
    pp.predict()

    print("終了")