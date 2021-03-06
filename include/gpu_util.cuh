#pragma once

#ifndef GPU_UTIL
#define GPU_UTIL

#include <cinttypes>
#include <vector>
#include <assert.h>

#include <cuda_runtime.h>

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

#define cucheck_dev(call)                                   \
{                                                           \
  cudaError_t cucheck_err = (call);                         \
  if(cucheck_err != cudaSuccess) {                          \
    const char *err_str = cudaGetErrorString(cucheck_err);  \
    printf("%s (%d): %s\n", __FILE__, __LINE__, err_str);   \
    assert(0);                                              \
  }                                                         \
}

void copyAritiesToDevice(
                        int streamId,
                        const std::vector<uint64_t>& pArities,
                        const std::vector<uint64_t>& pAritiesPrefixProd,
                        const std::vector<uint64_t>& pAritiesPrefixSum);

void cudaCallBlockCount(const uint block_count,
                        const uint per_block_thread_count,
                        const uint words_per_vector,
                        const uint variablesCount,
                        const uint configs_per_query,
                        const uint64_t* bvectorsPtr,
                        uint64_t* results,
                        uint64_t *resultsPa,
                        uint64_t* intermediateData,
                        int streamId,
                        cudaStream_t stream);

#endif // GPU_UTIL
