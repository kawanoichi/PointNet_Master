"""tensorflowがgpuを認識できているか確認する.

@author kawanoichi
実行コマンド
$ python3 enviroment_check/check_tensor_gpu.py
"""
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())