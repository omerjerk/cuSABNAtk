#ifndef GPU_UTIL
#define GPU_UTIL

#include <cuda_runtime.h>
#include <cinttypes>
#include <vector>

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

__constant__ uint64_t aritiesPtr_[20];
__constant__ uint64_t aritiesPrefixProdPtr_[20];
__constant__ uint64_t aritiesPrefixSumPtr_[20];

void copyAritiesToDevice(const std::vector<uint64_t>& pArities,
  const std::vector<uint64_t>& pAritiesPrefixProd,
  const std::vector<uint64_t>& pAritiesPrefixSum);

void cudaCallBlockCount(
        const uint block_count,
        const uint per_block_thread_count,
        const uint words_per_vector,
        const uint vectors_per_config,
        const uint configs_per_query,
        const uint64_t* bvectorsPtr,
        uint64_t* results,
        uint64_t *resultsPa,
        uint64_t* intermediateStatesPtr,
        cudaStream_t streamId);

#endif
