"""機械学習モジュール.

修論「単一画像から3Dモデル生成」の機械学習を行うファイル
@author kawanoichi
実行コマンド
$ bash scripts/train.sh
"""
import argparse
import json
import os

class Train:
    """学習を行うクラス."""
    def __init__(self, param_file="train_param.json"):
        """コンストラクタ.

        Args:
            param_file (str): 学習時に使うパラメータ
        """
        self.__param_file = param_file

    def load_param(self):
        """jsonファイルからパラメータを読み込む."""
        # ファイルがない場合のエラー
        if not os.path.isfile(self.__param_file):
            raise FileNotFoundError("'%s' not found" % self.__param_file)

        # パラメータを読み込む
        with open(self.__param_file) as fp:
            param_data = json.load(fp)
            self.ptcloud = param_data["PTCLOUD"]
            self.output_points = param_data["OUTPUTPOINTS"]
            self.batch_size = param_data["BATCH_SIZE"]
            self.epochs = param_data["EPOCHS"]
            self.data_set_len = param_data["DATASET_LEN"]
            
    def train(self):
        self.load_param()

        # for epoch in range(self.epochs):
        #     # dt_now = datetime.datetime.now()
        #     # print(dt_now)
        #     data_number = 0
        #     for i in range(DATASET_LEN):
        #         train_images, train_labels = get_data(data_number)
        #         if i == 0:
        #             print("train_images.shape", train_images.shape)
        #             print("train_labels.shape",train_labels.shape)
                
        #         train_step(train_images, train_labels, model, loss_object, optimizer, loss_average)
        #         data_number += 1
            
        #     time = str(datetime.datetime.now())
        #     template1 = "Epoch {}, time: {}, loss: {}"
        #     print(template1.format(epoch+1, time[11:16], loss_average.result()))

        #     train_loss_np = loss_average.result().numpy()
        #     train_loss_history = np.append(train_loss_history, train_loss_np)

        #     loss_average.reset_states()









if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="学習を実行するファイル")  
    parser.add_argument("data_set", help="学習を行うデータセット")
    args = parser.parse_args()

    # 学習
    train = Train()
    train.train()
    print("終了")