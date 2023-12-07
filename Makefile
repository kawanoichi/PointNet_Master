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

format:
	poetry run python3 -m autopep8 -i *.py
	poetry run python3 -m autopep8 -i -r work/

train:
	poetry run python3 train.py

predict:
	poetry run python3 predict_point_cpu.py
# poetry run python3 predict_point_shapenet.py

check_gpu:
	@echo pytorch
	python3 enviroment_check/check_torch_gpu.py
	@echo tensorflow
	python3 enviroment_check/check_tensor_gpu.py

surface_run:
	 poetry run python3 work/make_surface.py

mesh:
	 poetry run python3 trush/mesh.py

mesh2:
	 poetry run python3 trush/mesh_02.py

zikken:
	 poetry run python3 work/zikken.py

