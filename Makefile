help:
	@echo 学習を行う
	@echo " $$ make train"
	@echo 学習済みモデルを使用して画像から点群予測を行う
	@echo " $$ make predict"

train:
	python3 train.py

predict:
	python3 predict_point_shapenet.py

convert:
	python3 convert_extension.py