CXX=nvcc
CXXFLAGS=-std=c++11 -O3 -I/util/common/cuda/cuda-8.0/include -L/util/common/cuda/cuda-8.0/lib64 -lcudart

all: count

count: query.cpp
	$(CXX) -c *.cu
	$(CXX) $(CXXFLAGS)  query.cpp gpu_util.o -o  $@
cuda: gpu_util.cu
	nvcc -c gpu_util.cu

clean:
	rm -f *.o
	rm -f test
