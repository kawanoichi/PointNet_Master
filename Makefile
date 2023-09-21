help:
	@echo 学習を行う
	@echo " $$ make train"
	@echo 学習済みモデルを使用して画像から点群予測を行う
	@echo " $$ make predict"
	@echo 拡張子の変更
	@echo " $$ make convert"
	@echo tensorflowのgpu認識のチェックを行う
	@echo " $$ make check tensor"
	@echo pytorchのgpu認識のチェックを行う
	@echo " $$ make check torch"

train:
	python3 train.py

predict:
	python3 predict_point_shapenet.py

convert:
	python3 convert_extension.py

check_tensor:
	python3 enviroment_check/check_tensor_gpu.py

check_torch:
	python3 enviroment_check/check_torch_gpu.py