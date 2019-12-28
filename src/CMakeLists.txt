CUDA_ADD_LIBRARY(gpu_util STATIC gpu_util.cu)
target_link_libraries(gpu_util ${CUDA_cudadevrt_LIBRARY})