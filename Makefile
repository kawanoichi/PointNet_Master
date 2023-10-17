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
	python -m autopep8 -i *.py

train:
	python3 train.py

predict:
	python3 predict_point_shapenet.py

convert:
	python3 convert_extension.py

check_gpu:
	@echo pytorch
	python3 enviroment_check/check_torch_gpu.py
	@echo tensorflow
	python3 enviroment_check/check_tensor_gpu.py

# ソースコード実行
cube:
	 python3 create_mesh/create_point_of_cube.py 
info:
	 python3 create_mesh/infomation.py 
mesh:
	 python3 create_mesh/mesh.py 
marching:
	 python3 create_mesh/marching_cube.py
surface_run:
	 python3 work/make_surface.py
surface_run:
	 python3 work/zikken.py

