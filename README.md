# 研究テーマ「単一画像から３Dモデルを再構築する」  
宮崎大学大学院　令和4,5年度  

## 先行研究
3D-ReConstnet<br>
https://github.com/yonghanzhang94/A-Single-View-3D-Object-Point-Cloud-Reconstruction<br>

## 環境
[環境構築の手順](https://github.com/kawanoichi/PointNet_Master/issues/5)<br>
Ubuntu     : 20.04.5<br>
GPU        : NVIDIA GeForce RTX 3060 Ti<br>
cuda       : 11.7<br>
python     : 3.7.15<br>
numpy      : 1.19.2<br>
pytorch    : 1.13.1<br>
tensorflow : 1.13.1<br>

生成した点群の表示はmatplotlibだと重いため、meshlabを使用する。<br>

### インストール<br>
```
sudo apt -y install meshlab
```
## 実行<br>

### オブジェクト(点群,メッシュ化オブジェクト)の視覚化のできるソフトウェア
meshlabの起動コマンド
```
meshlab
```

## 学習からメッシュ化までの流れ
### 学習
```train.py```ファイルを使用して画像から点群を予測し、学習済みモデルを作成する。
```
python3 train.py
```
学習済みモデルは```learned_model```のフォルダに作成する。

### 点群の予測
```predict_point_shapnet.py```ファイルと学習済みモデルを使用して、画像から点群を予測する。
```
python3 predict_point_shapenet.py
```
出力はnpyファイル。

### 表面を生成する
```convert_extension.py```ファイルを使用して予測したnpyファイルをplyファイルに変換する。
```
python3 convert_extension.py
```

```mesh.py```ファイルを使用して点群のメッシュ化(面の生成)を行う。
```
python3 mesh.py -m plyファイル名
```




## データセット
### ShapeNet
データセット(以下のURLからShapeNetRendering.tgzをダウンロード)<br>
https://cvgl.stanford.edu/data2/<br>
スプリットデータ<br>
https://drive.google.com/file/d/10FR-2Lbn55POB1y47MJ12euvobi6mgtc/view<br>
pointcloud<br>
https://drive.google.com/file/d/1cfoe521iTgcB_7-g_98GYAqO553W8Y0g/view<br>

### Pix3D
Pix3D dataset (~20 GB)<br>
https://github.com/xingyuansun/pix3d<br>
Pix3D pointclouds (~13 MB)<br>
https://drive.google.com/open?id=1RZakyBu9lPbG85SyconBn4sR8r2faInV<br>