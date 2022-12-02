nvcc = /usr/local/cuda-11.7/bin/nvcc
cudalib = /usr/local/cuda-11.7/lib64/
tensorflow = /home/issei/anaconda3/envs/PointNet/lib/python3.6/site-packages/tensorflow/include
tf = /home/issei/anaconda3/envs/PointNet/lib/python3.6/site-packages/tensorflow

all: tf_ops/cd/tf_nndistance_so.so tf_ops/emd/tf_auctionmatch_so.so
.PHONY : all

tf_ops/cd/tf_nndistance_so.so: tf_ops/cd/tf_nndistance_g.cu.o tf_ops/cd/tf_nndistance.cpp
	g++ -std=c++11 tf_ops/cd/tf_nndistance.cpp tf_ops/cd/tf_nndistance_g.cu.o -o tf_ops/cd/tf_nndistance_so.so -shared -fPIC -I $(tensorflow) -lcudart -L $(cudalib) -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -L $(tf) -ltensorflow_framework

tf_ops/cd/tf_nndistance_g.cu.o: tf_ops/cd/tf_nndistance_g.cu
	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o tf_ops/cd/tf_nndistance_g.cu.o tf_ops/cd/tf_nndistance_g.cu -I $(tensorflow) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2 -L $(tf) -ltensorflow_framework

tf_ops/emd/tf_auctionmatch_so.so: tf_ops/emd/tf_auctionmatch_g.cu.o tf_ops/emd/tf_auctionmatch.cpp
	g++ -std=c++11 tf_ops/emd/tf_auctionmatch.cpp tf_ops/emd/tf_auctionmatch_g.cu.o -o tf_ops/emd/tf_auctionmatch_so.so -shared -fPIC -I $(tensorflow) -lcudart -L $(cudalib) -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -L $(tf) -ltensorflow_framework

tf_ops/emd/tf_auctionmatch_g.cu.o: tf_ops/emd/tf_auctionmatch_g.cu
	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o tf_ops/emd/tf_auctionmatch_g.cu.o tf_ops/emd/tf_auctionmatch_g.cu -I $(tensorflow) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2 -arch=sm_50 -L $(tf) -ltensorflow_framework
