#ifndef GPU_UTIL
#define GPU_UTIL

#include <cuda_runtime.h>

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

void cudaCallIntersect(
	const uint block_count,
	const uint per_block_thread_count,
	const uint n,
	const unsigned long long*a,
	const unsigned long long*b,
  unsigned long long*reduce,
	unsigned long long*out);

  void cudaCallCount(
  	const uint block_count,
  	const uint per_block_thread_count,
  	const uint n,
    const uint s,
  	const unsigned long long**a,
    unsigned long long*reduce,
  	unsigned long long*out);

    void cudaCallBlockCount(
    	const uint block_count,
    	const uint per_block_thread_count,
    	const uint words_per_vector,
      const uint vectors_per_config,
      const uint configs_per_query,
    	const unsigned long long** bvectorsPtr,
      unsigned long long*results,
    	unsigned long long*states);

#endif
