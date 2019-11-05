#!/bin/bash

rm gpu_util.o
rm ../examples/query
rm query.o

#compile cuda code
nvcc -arch=sm_61 -rdc=true -c gpu_util.cu -lcudadevrt -o gpu_util.o -I../include
#create static library from object code
ar rcs gpu_util.a gpu_util.o
#compile C++ code
g++ -fopenmp  -std=c++14 -O3 -c ../examples/query.cpp -o query.o -I/usr/local/cuda-10.0/include -I../include
#link
nvcc -arch=sm_61 query.o gpu_util.a -Xcompiler -fopenmp  -std=c++14 -O3  -o ../examples/query \
-I/usr/local/cuda-10.0/include -I../include /usr/local/cuda-10.0/lib64/libcudart_static.a \
-ldl -lrt /usr/local/cuda-10.0/lib64/libcudart_static.a