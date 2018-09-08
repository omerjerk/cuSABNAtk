#include "gpu_util.cuh"
#include <cstdio>

//
// reduction based on cuda-8.0/samples/6_Advanced/reduction
//

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
    __device__ inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void
counts(const T **g_idata,
  T *g_odata,
  unsigned int words_per_vector,
  const int vectors_per_config,
  const int configs_per_query)
{
  T *sdata = SharedMemory<T>();

  int threads_per_config = (words_per_vector + 1) / 2;
  int total_threads = threads_per_config * configs_per_query;
  unsigned int tid = threadIdx.x;

  unsigned int global_index = blockIdx.x*blockSize*2 + threadIdx.x;
  unsigned int config_index = global_index / threads_per_config;
  unsigned int word_index = global_index % threads_per_config;
  unsigned int bv_start_index = config_index * vectors_per_config;
  unsigned int word_size_half = words_per_vector / 2;
  unsigned int gridSize = blockSize*2*gridDim.x;

  T mySum = 0;

  while (global_index < total_threads)
  {
    T localState = g_idata[bv_start_index][word_index]; // first word slice of config

    // running sum for all word slices
    for(int p = 1; p < vectors_per_config; p++)
    {
      localState = localState & g_idata[bv_start_index + p][word_index];
    }

    mySum += __popcll(localState);

    // first word slice reduce during load
    // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
    if (nIsPow2 || word_index + word_size_half < words_per_vector)
    {
      localState = g_idata[bv_start_index][word_index + word_size_half];
      for(int p = 1; p < vectors_per_config; p++)
      {
        localState = localState & g_idata[bv_start_index + p][word_index + word_size_half];
      }
      mySum += __popcll(localState);
    }
    global_index += gridSize;
  }

  // each thread puts its local sum into shared memory
  sdata[tid] = mySum;

  __syncthreads();

  if ((word_size_half >=   8) && (word_index <  4))
  {
    sdata[tid] = mySum = mySum + sdata[tid +  4];
  }

  __syncthreads();

  if ((word_size_half >=   4) && (word_index <  2))
  {
    sdata[tid] = mySum = mySum + sdata[tid +  2];
  }

  __syncthreads();

  if ((word_size_half >=   2) && (word_index <  1))
  {
    sdata[tid] = mySum = mySum + sdata[tid +  1];
  }

  __syncthreads();

  // write result for this block to global mem
  if (word_index == 0) g_odata[config_index] = mySum;
  __syncthreads();
} // counts

// from cuda samples reduction
unsigned int nextPow2(unsigned int x);

unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}
bool isPow2(unsigned int x);

bool isPow2(unsigned int x)
{
  return ((x&(x-1))==0);
}

void cudaCallBlockCount(
  const uint block_count,
  const uint per_block_thread_count,
  const uint words_per_vector,
  const uint vectors_per_config,
  const uint configs_per_query,
  const unsigned long long** bvectorsPtr,
  unsigned long long*results,
  unsigned long long*states) {
    int maxThreads = 1024;
    int blocks = 256;

    int threads_per_config = (words_per_vector + 1) / 2; // global data
    int threads = threads_per_config * configs_per_query;

    // int threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
    // blocks = (n + (threads * 2 - 1)) / (threads * 2);

    blocks = 1;

    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(unsigned long long) : threads * sizeof(unsigned long long);

    // debug setup for alarm data
    if (isPow2(words_per_vector) || true) // optimize out non pwr of 2 logic
    {
      counts<unsigned long long, 16, true><<< dimGrid, dimBlock, smemSize >>>(bvectorsPtr, results, words_per_vector, vectors_per_config, configs_per_query); //words, bvs/config, configs per query
    }
    else
    {
      counts<unsigned long long, 16, false><<< dimGrid, dimBlock, smemSize >>>(bvectorsPtr, results, words_per_vector, vectors_per_config, configs_per_query);
    }
  }
