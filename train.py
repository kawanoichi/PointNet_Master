"""機械学習モジュール.

修論「単一画像から3Dモデル生成」の機械学習を行うファイル
@author kawanoichi
実行コマンド
$ python3 train.py
"""
import json
import os
from datasets import GetShapenetDataset
import torch
import torch.backends.cudnn as cudnn
import model
import torch.optim as optim  #最適化アルゴリズムのパッケージ
from torch.autograd import Variable # 任意のスカラー値関数の自動微分を実装するクラスと関数
from tqdm import tqdm
from loss import batch_NN_loss, batch_EMD_loss



class train:
    """機械学習を行うクラス"""
    def __init__(self, train_param_file):
        self.__train_param_file = train_param_file
        self.load_param() #学習パラメータの読み込み
        self.load_dataset() #データセットの読み込み

    def load_param(self)->None:
        """インスタンス変数としてパラメータファイルを読み込む.
        Raises:
            FileNotFoundError: パラメータファイルが見つからない場合に発生
            KeyError: パラメータファイルの内容に期待するキーのデータが無い場合に発生
        """
        # パラメータファイルが存在するかの確認
        if not os.path.isfile(self.__train_param_file):
            raise FileNotFoundError("No file '%s'" % self.__train_param_file)

        with open(self.__train_param_file) as fp:
            train_param = json.load(fp)

        key = "batchsize"
        if key not in train_param:
            raise KeyError("key not found: '%s', file: %s"%(key, self.__train_param_file))
        self.__batchsize = train_param[key]

        key = "workers"
        if key not in train_param:
            raise KeyError("key not found: '%s', file: %s"%(key, self.__train_param_file))
        self.__workers = train_param[key]

        key = "nepoch"
        if key not in train_param:
            raise KeyError("key not found: '%s', file: %s"%(key, self.__train_param_file))
        self.__nepoch = train_param[key]

        key = "cats"
        if key not in train_param:
            raise KeyError("key not found: '%s', file: %s"%(key, self.__train_param_file))
        self.__cats = train_param[key]

        key = "num_points"
        if key not in train_param:
            raise KeyError("key not found: '%s', file: %s"%(key, self.__train_param_file))
        self.__num_points = train_param[key]

        key = "outfolder"
        if key not in train_param:
            raise KeyError("key not found: '%s', file: %s"%(key, self.__train_param_file))
        self.__outfolder = train_param[key]

        key = "modelG"
        if key not in train_param:
            raise KeyError("key not found: '%s', file: %s"%(key, self.__train_param_file))
        self.__modelG = train_param[key]

        key = "lr"
        if key not in train_param:
            raise KeyError("key not found: '%s', file: %s"%(key, self.__train_param_file))
        self.__lr = train_param[key]

        print("batchsize  :", self.__batchsize)
        print("workers    :", self.__workers)
        print("nepoch     :", self.__nepoch)
        print("cats       :", self.__cats)
        print("num_points :", self.__num_points)
        print("outfolder  :", self.__outfolder)
        print("modelG     :", self.__modelG)
        print("lr         :", self.__lr)


    def load_dataset(self)->None:
        """インスタンス変数としてデータセットを読み込む.
        Raises:
            FileNotFoundError: パラメータファイルが見つからない場合に発生
        """
        print("\nload_dataset...", end=" ")

        train_models = 'data/splits/'+'train_models.json'
        val_models = 'data/splits/'+'val_models.json'

        if not os.path.isfile(train_models):
            raise FileNotFoundError("No file '%s'" % train_models)
        
        with open(train_models, 'r') as f:
            train_models_dict = json.load(f)

        if not os.path.isfile(val_models):
            raise FileNotFoundError("No file '%s'" % val_models)

        with open(val_models, 'r') as f:
            val_models_dict = json.load(f)

        # データパス
        data_dir_imgs = 'data/shapenet/ShapeNetRendering'
        data_dir_pcl = 'data/shapenet/ShapeNet_pointclouds/'

        # パラメータファイルが存在するかの確認
        if not os.path.exists(data_dir_imgs):
            raise FileNotFoundError("No file '%s'" % data_dir_imgs)

        if not os.path.exists(data_dir_pcl):
            raise FileNotFoundError("No file '%s'" % data_dir_pcl)

        # GetShapenetDatasetクラスのインスタンス化
        self.__dataset = GetShapenetDataset(data_dir_imgs, data_dir_pcl, train_models_dict, self.__cats, self.__num_points)
        
        # 参考URL: https://pytorch.org/docs/stable/data.html
        self.__dataloader = torch.utils.data.DataLoader(self.__dataset, batch_size=self.__batchsize,
                                                shuffle=True, num_workers=int(self.__workers))

        # GetShapenetDatasetクラスのインスタンス化
        self.__test_dataset = GetShapenetDataset(data_dir_imgs, data_dir_pcl, val_models_dict, self.__cats, self.__num_points)
        
        # 参考URL: https://pytorch.org/docs/stable/data.html
        self.__testdataloader = torch.utils.data.DataLoader(self.__test_dataset, batch_size=self.__batchsize,
                                                shuffle=True, num_workers=int(self.__workers))
        print("完了\n")

    def train(self):
        """機械学習を行う関数．"""
        print("学習を開始")
        # 出力先フォルダの作成
        try:
            os.makedirs(self.__outfolder)
        except OSError:
            pass

        # Trueの場合、cuDNNが複数の畳み込みアルゴリズムをベンチマークし、最速のものを選択
        cudnn.benchmark = True

        # generatorクラスのインスタンス化
        gen = model.generator(num_points=self.__num_points)

        if not self.__modelG == '':
            print("例外")
            with open(self.__modelG, "rb") as f:
                gen.load_state_dict(torch.load(f))

        # GPU接続?
        print("gpuに接続...")
        gen.cuda()
        print("完了")

        # 最適化アルゴリズム
        # optimizerG = optim.RMSprop(gen.parameters(), lr = self.__lr)
        optimizerG = optim.Adam(gen.parameters(), lr = self.__lr)

        # バッチサイズで分けたグループ数
        num_batch = len(self.__dataset)/self.__batchsize

        # 学習
        print("学習開始")
        for epoch in range(self.__nepoch+1):
            data_iter = iter(self.__dataloader)
            i = 0
            while i < len(self.__dataloader):

                try:
                    data = next(data_iter)
                except StopIteration:
                    break
                
                i += 1
                images, points = data

                # 勾配を計算
                points = Variable(points.float())
                points = points.cuda()
                images = Variable(images.float())
                images = images.cuda()

                optimizerG.zero_grad()

                fake, _, _, _ = gen(images)
                fake = fake.transpose(2, 1)

                lossG1 = batch_NN_loss(points, fake)
                lossG2 = batch_EMD_loss(points, fake)

                lossG = lossG1 + lossG2

                lossG.backward()
                optimizerG.step()

                if i % 100 == 0:
                    print('[%d: %d/%d] train lossG: %f' %(epoch, i, num_batch, lossG.item()))

            if epoch % 20 == 0 and epoch != 0:
                self.__lr = self.__lr * 0.5
                for param_group in optimizerG.param_groups:
                    param_group['lr'] = self.__lr
                print('lr decay:', self.__lr)

            if epoch % 50 == 0:
                torch.save(gen.state_dict(), '%s/modelG_%d.pth' % (self.__outfolder, epoch))

        print("学習終了")
            
if __name__ == "__main__":
    train_param_file = "train_param.json"
    train = train(train_param_file)
    train.train()
    print("終了")