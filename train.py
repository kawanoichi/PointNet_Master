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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="学習を実行するファイル")  
    parser.add_argument("data_set", help="学習を行うデータセット")
    args = parser.parse_args()

    # 学習
    train = Train()
    train.train()
    print("終了")