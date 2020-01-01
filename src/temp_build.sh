#!/bin/bash

rm gpu_util.o
rm ../examples/query
rm query.o
rm gpu_util_link.o

#compile cuda code
nvcc -x cu -g -G -arch=sm_61 -c gpu_util.cu -o gpu_util.o -I../include
#link cuda code
# nvcc -arch=sm_61 -dlink -o gpu_util_link.o gpu_util.o -lcudadevrt -lcudart
#compile C++ code
g++ -fopenmp  -std=c++14 -c ../examples/query.cpp -o query.o -I/usr/local/cuda-10.0/include -I../include
#link using nvcc
# nvcc --default-stream per-thread -arch=sm_61 query.o gpu_util.o -Xcompiler -fopenmp  -std=c++14 -O3  -o ../examples/query \
# -I/usr/local/cuda-10.0/include -I../include /usr/local/cuda-10.0/lib64/libcudart_static.a \
# -ldl -lrt /usr/local/cuda-10.0/lib64/libcudart_static.a

#link using g++
g++ query.o gpu_util.o -fopenmp -std=c++14 -o ../examples/query \
-I/usr/local/cuda-10.0/include -I../include -L/usr/local/cuda-10.0/lib64 -lcudart -lcudadevrt