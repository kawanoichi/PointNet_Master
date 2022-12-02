"""pytorchがgpuを認識できているか確認する.

@author kawanoichi
実行コマンド
$ python3 enviroment_check/check_torch_gpu.py
"""
import torch

print("torch version ", torch.__version__)

print("GPU確認", torch.cuda.is_available())