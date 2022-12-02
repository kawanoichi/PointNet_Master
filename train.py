"""機械学習モジュール.

修論「単一画像から3Dモデル生成」の機械学習を行うファイル
@author kawanoichi
実行コマンド
$ bash scripts/train.sh
"""
import os
import torch
import argparse
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train for')
    # ALL: ['02691156','02828884','02933112','02958343','03001627','03636649','03211117','04090263','04256520','03691459','04379243','04401088','04530566']
    parser.add_argument('--cats', default=['02691156'], type=str,
                        help='Category to train on : ["airplane":02691156, "bench":02828884, "cabinet":02933112, '
                            '"car":02958343, "chair":03001627, "lamp":03636649, '
                            '"monitor":03211117, "rifle":04090263, "sofa":04256520, '
                            '"speaker":03691459, "table":04379243, "telephone":04401088, '
                            '"vessel"：04530566]')
    parser.add_argument('--num_points', type=int, default=1024, help='number of epochs to train for, [1024, 2048]')
    parser.add_argument('--outf', type=str, default='model',  help='output folder')
    parser.add_argument('--modelG', type=str, default = '', help='generator model path')
    parser.add_argument('--lr', type=float, default = '0.00005', help='learning rate')

    opt = parser.parse_args()
    print (opt)

    # 環境変数の宣言
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

    # pytorchのprintオプションの設定
    # 全て表示させるように設定
    torch.set_printoptions(profile='full')

    # 無名関数 ※ lamba 引数: 返り値
    blue = lambda x:'\033[94m' + x + '\033[0m'

    opt.manualSeed = random.randint(1, 10000) # fix seed
    print("Random Seed: ", opt.manualSeed)
    print(random.seed(opt.manualSeed))
    # print("Random Seed: ", opt.manualSeed)



    random.seed(random.randint(1, 10000))
# 
    








    print("終了")