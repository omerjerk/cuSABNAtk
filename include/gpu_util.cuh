#ifndef GPU_UTIL
#define GPU_UTIL

#include <cuda_runtime.h>

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

void cudaCallBlockCount(
        const uint block_count,
        const uint per_block_thread_count,
        const uint words_per_vector,
        const uint vectors_per_config,
        const uint configs_per_query,
        const uint64_t* arities,
        const uint64_t* aritiesPrefixProd,
        const uint64_t* aritiesPrefixSum,
        const uint64_t* bvectorsPtr,
        uint64_t* results,
        uint64_t* intermediateStatesPtr,
        cudaStream_t streamId);

#endif
