#include "gpu_util.cuh"
#include <cstdio>

//
// from cuda-8.0/samples/6_Advanced/reduction
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
count(const T **g_idata, T *g_odata, unsigned int n,  const int s)
{
  T *sdata = SharedMemory<T>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
  unsigned int gridSize = blockSize*2*gridDim.x;

  T mySum = 0;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while (i < n)
  {
    T localState = g_idata[0][i];
    // running sum for all instances
    for(int d = 1; d < s; d++)
    {
      localState = localState & g_idata[d][i];
    }
    mySum += __popcll(localState);

    // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
    if (nIsPow2 || i + blockSize < n)
    {
      localState = g_idata[0][i+blockSize];
      for(int d = 1; d < s; d++)
      {
        localState = localState & g_idata[d][i+blockSize];
      }
      mySum += __popcll(localState);
    }

    i += gridSize;
  }

  // each thread puts its local sum into shared memory
  sdata[tid] = mySum;
  __syncthreads();

  // do reduction in shared mem
  if ((blockSize >= 512) && (tid < 256))
  {
    sdata[tid] = mySum = mySum + sdata[tid + 256];
  }

  __syncthreads();

  if ((blockSize >= 256) &&(tid < 128))
  {
    sdata[tid] = mySum = mySum + sdata[tid + 128];
  }

  __syncthreads();

  if ((blockSize >= 128) && (tid <  64))
  {
    sdata[tid] = mySum = mySum + sdata[tid +  64];
  }

  __syncthreads();

  #if (__CUDA_ARCH__ >= 300 )
  if ( tid < 32 )
  {
    // Fetch final intermediate sum from 2nd warp
    if (blockSize >=  64) mySum += sdata[tid + 32];
    // Reduce final warp using shuffle
    for (int offset = warpSize/2; offset > 0; offset /= 2)
    {
      mySum += __shfl_down(mySum, offset);
    }
  }
  #else
  // fully unroll reduction within a single warp
  if ((blockSize >=  64) && (tid < 32))
  {
    sdata[tid] = mySum = mySum + sdata[tid + 32];
  }

  __syncthreads();

  if ((blockSize >=  32) && (tid < 16))
  {
    sdata[tid] = mySum = mySum + sdata[tid + 16];
  }

  __syncthreads();

  if ((blockSize >=  16) && (tid <  8))
  {
    sdata[tid] = mySum = mySum + sdata[tid +  8];
  }

  __syncthreads();

  if ((blockSize >=   8) && (tid <  4))
  {
    sdata[tid] = mySum = mySum + sdata[tid +  4];
  }

  __syncthreads();

  if ((blockSize >=   4) && (tid <  2))
  {
    sdata[tid] = mySum = mySum + sdata[tid +  2];
  }

  __syncthreads();

  if ((blockSize >=   2) && ( tid <  1))
  {
    sdata[tid] = mySum = mySum + sdata[tid +  1];
  }

  __syncthreads();
  #endif

  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = mySum;
}

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

  // printf("gid=%d, tid=%d cidx=%d widx=%d wpv=%d vpc=%d cpq=%d\n",
  // global_index,
  // tid,
  // config_index,
  // word_index,
  // words_per_vector,
  // vectors_per_config,
  // configs_per_query);

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

  //__syncthreads();
  // if(tid == 0){
  //   int smem_element_count = configs_per_query * words_per_vector / 2;
  //   for(int i = 0; i < smem_element_count; i++){
  //     printf("%3llu ",sdata[i]);
  //     if(i%word_size_half == word_size_half - 1)
  //     {
  //       printf("\n");
  //     }
  //   }
  //   printf("\n");
  // }

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

  // if(tid == 0){
  //   int smem_element_count = configs_per_query * words_per_vector / 2;
  //   for(int i = 0; i < smem_element_count; i++){
  //     printf("%3llu ",sdata[i]);
  //     if(i%word_size_half == word_size_half - 1)
  //     {
  //       printf("\n");
  //     }
  //   }
  // }

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

void cudaCallCount(
  const uint block_count,
  const uint per_block_thread_count,
  const uint n,
  const uint s,
  const unsigned long long**bitvectors,
  unsigned long long*reduce,
  unsigned long long*out) {

    int maxThreads = 1024;
    int threads = 16;
    int blocks = 256;

    threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
    blocks = (n + (threads * 2 - 1)) / (threads * 2);

    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    int smemSize = (threads <= 32) ? 2 * threads * sizeof(unsigned long long) : threads * sizeof(unsigned long long);

    // todo add thread size switch statement
    // debug setup for n=16, t=8
    if (isPow2(n))
    {
      count<unsigned long long, 8, true><<< dimGrid, dimBlock, smemSize >>>(bitvectors, reduce, n, s);
    }
    else
    {
      count<unsigned long long, 8, false><<< dimGrid, dimBlock, smemSize >>>(bitvectors, reduce, n, s);
    }
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

    // printf("%d %d %d %d %d\n",
    // block_count,
    // per_block_thread_count,
    // words_per_vector,
    // vectors_per_config,
    // configs_per_query);

    int threads_per_config = (words_per_vector + 1) / 2; // global data
    int threads = threads_per_config * configs_per_query;

    // int threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
    // blocks = (n + (threads * 2 - 1)) / (threads * 2);

    blocks = 1;

    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(unsigned long long) : threads * sizeof(unsigned long long);

    //printf("launching counts with %d blocks, %d threads per block, %d smem per block\n", blocks, threads, smemSize);

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
