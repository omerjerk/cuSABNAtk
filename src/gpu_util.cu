/***
 *  $Id$
 **
 *  File: gpu_util.cu
 *  Created: Mar 22, 2019
 *
 * This code has been derived from NVIDIA samples: cuda-8.0/samples/6_Advanced/reduction
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *  This file is part of cuSABNAtk.
 */

#ifndef GPU_UTIL_CU
#define GPU_UTIL_CU

#include "gpu_util.cuh"

//
// Reduction based on cuda-8.0/samples/6_Advanced/reduction
//

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template <class T> struct SharedMemory {
    __device__ inline operator T*() {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }

    __device__ inline operator const T*() const {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }
}; // struct SharedMemory


template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void counts(const T** g_idata,
                       T* g_odata,
                       T* g_rdata,
                       unsigned int words_per_vector,
                       const int vectors_per_config,
                       const int configs_per_query) {
    T* sdata = SharedMemory<T>();

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockSize + threadIdx.x;
    unsigned int vector_index = vectors_per_config * blockIdx.x;
    unsigned int word_index = i % blockSize; // cant this be tid
    unsigned int result_index = blockIdx.x * words_per_vector + tid;

    T mySum = 0;
    T localState = g_idata[vector_index][word_index]; // first word slice of config

    // running sum for all word slices
    for(int p = 1; p < vectors_per_config; ++p) localState = localState & g_idata[vector_index + p][word_index];

    if (g_rdata != 0) { // todo can be compile time decision
        g_rdata[result_index] = localState;
    }

    mySum += __popcll(localState);

    // ensure we don't read out of bounds -- this is optimized away for power of 2 sized arrays
    if (nIsPow2 || (tid + blockSize < words_per_vector)) {
        unsigned int word_index_upper_half = word_index + blockSize;
        localState = g_idata[vector_index][word_index_upper_half];

        for(int p = 1; p < vectors_per_config; p++) localState = localState & g_idata[vector_index + p][word_index_upper_half];

        if(g_rdata != 0){ // todo can be compile time decision
            g_rdata[result_index + blockSize] = localState;
        }

        mySum += __popcll(localState);
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;

    __syncthreads();

    // do reduction in shared mem
    if ((blockSize >= 512) && (tid < 256)) sdata[tid] = mySum = mySum + sdata[tid + 256];

    __syncthreads();

    if ((blockSize >= 256) && (tid < 128))  sdata[tid] = mySum = mySum + sdata[tid + 128];

   __syncthreads();

   if ((blockSize >= 128) && (tid <  64)) sdata[tid] = mySum = mySum + sdata[tid +  64];

  __syncthreads();

  #if (__CUDA_ARCH__ >= 300 )
  if ( tid < 32 ) {
      // Fetch final intermediate sum from 2nd warp
      if (blockSize >=  64) mySum += sdata[tid + 32];
      // Reduce final warp using shuffle
      for (int offset = warpSize / 2; offset > 0; offset /= 2) mySum += __shfl_down_sync(0xFFFFFFFF, mySum, offset);
  }
  #else
  // fully unroll reduction within a single warp
  if ((blockSize >= 64) && (tid < 32)) sdata[tid] = mySum = mySum + sdata[tid + 32];

  __syncthreads();

  if ((blockSize >= 32) && (tid < 16)) sdata[tid] = mySum = mySum + sdata[tid + 16];

  __syncthreads();

  if ((blockSize >= 16) && (tid <  8)) sdata[tid] = mySum = mySum + sdata[tid +  8];

  __syncthreads();

  if ((blockSize >= 8) && (tid <  4)) sdata[tid] = mySum = mySum + sdata[tid +  4];

  __syncthreads();

  if ((blockSize >= 4) && (tid <  2)) sdata[tid] = mySum = mySum + sdata[tid +  2];

  __syncthreads();

  if ((blockSize >= 2) && ( tid <  1)) sdata[tid] = mySum = mySum + sdata[tid +  1];

  __syncthreads();
#endif

  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = mySum;

  __syncthreads();
} // counts



// from cuda samples reduction
inline unsigned int nextPow2(unsigned int x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
} // nextPow2

inline bool isPow2(unsigned int x) { return ((x & (x - 1)) == 0); }


void cudaCallBlockCount(const uint block_count,
                        const uint per_block_thread_count,
                        const uint words_per_vector,
                        const uint vectors_per_config,
                        const uint configs_per_query,
                        const unsigned long long** bvectorsPtr,
                        unsigned long long* results,
                        unsigned long long* states) {
    cudaDeviceSynchronize();

    int threads = nextPow2((words_per_vector + 1) >> 1);

    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(configs_per_query, 1, 1);
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(unsigned long long) : threads * sizeof(unsigned long long);

    if (isPow2(words_per_vector) && (words_per_vector > 1)) // optimize out non power of 2 logic
        {
            switch (threads)
                {
                  case 512:
                      counts<unsigned long long, 512, true><<< dimGrid, dimBlock, smemSize >>>(bvectorsPtr, results, states, words_per_vector, vectors_per_config, configs_per_query);
                      break;

                  case 256:
                      counts<unsigned long long, 256, true><<< dimGrid, dimBlock, smemSize >>>(bvectorsPtr, results, states, words_per_vector, vectors_per_config, configs_per_query);
                      break;

                  case 128:
                      counts<unsigned long long, 128, true><<< dimGrid, dimBlock, smemSize >>>(bvectorsPtr, results, states, words_per_vector, vectors_per_config, configs_per_query);
                      break;

                  case 64:
                      counts<unsigned long long, 64, true><<< dimGrid, dimBlock, smemSize >>>(bvectorsPtr, results, states, words_per_vector, vectors_per_config, configs_per_query);
                      break;

                  case 32:
                      counts<unsigned long long, 32, true><<< dimGrid, dimBlock, smemSize >>>(bvectorsPtr, results, states, words_per_vector, vectors_per_config, configs_per_query);
                      break;

                  case 16:
                      counts<unsigned long long, 16, true><<< dimGrid, dimBlock, smemSize >>>(bvectorsPtr, results, states, words_per_vector, vectors_per_config, configs_per_query);
                      break;

                  case 8:
                      counts<unsigned long long, 8, true><<< dimGrid, dimBlock, smemSize >>>(bvectorsPtr, results, states, words_per_vector, vectors_per_config, configs_per_query);
                      break;

                  case 4:
                      counts<unsigned long long, 4, true><<< dimGrid, dimBlock, smemSize >>>(bvectorsPtr, results, states, words_per_vector, vectors_per_config, configs_per_query);
                      break;

                  case 2:
                      counts<unsigned long long, 2, true><<< dimGrid, dimBlock, smemSize >>>(bvectorsPtr, results, states, words_per_vector, vectors_per_config, configs_per_query);
                      break;

                  case 1:
                      counts<unsigned long long, 1, true><<< dimGrid, dimBlock, smemSize >>>(bvectorsPtr, results, states, words_per_vector, vectors_per_config, configs_per_query);
                      break;
                }
        }
    else
        {
            switch (threads)
                {
                  case 512:
                      counts<unsigned long long, 512, false><<< dimGrid, dimBlock, smemSize >>>(bvectorsPtr, results, states, words_per_vector, vectors_per_config, configs_per_query);
                      break;

                  case 256:
                      counts<unsigned long long, 256, false><<< dimGrid, dimBlock, smemSize >>>(bvectorsPtr, results, states, words_per_vector, vectors_per_config, configs_per_query);
                      break;

                  case 128:
                      counts<unsigned long long, 128, false><<< dimGrid, dimBlock, smemSize >>>(bvectorsPtr, results, states, words_per_vector, vectors_per_config, configs_per_query);
                      break;

                  case 64:
                      counts<unsigned long long, 64, false><<< dimGrid, dimBlock, smemSize >>>(bvectorsPtr, results, states, words_per_vector, vectors_per_config, configs_per_query);
                      break;

                  case 32:
                      counts<unsigned long long, 32, false><<< dimGrid, dimBlock, smemSize >>>(bvectorsPtr, results, states, words_per_vector, vectors_per_config, configs_per_query);
                      break;

                  case 16:
                      counts<unsigned long long, 16, false><<< dimGrid, dimBlock, smemSize >>>(bvectorsPtr, results, states, words_per_vector, vectors_per_config, configs_per_query);
                      break;

                  case 8:
                      counts<unsigned long long, 8, false><<< dimGrid, dimBlock, smemSize >>>(bvectorsPtr, results, states, words_per_vector, vectors_per_config, configs_per_query);
                      break;

                  case 4:
                      counts<unsigned long long, 4, false><<< dimGrid, dimBlock, smemSize >>>(bvectorsPtr, results, states, words_per_vector, vectors_per_config, configs_per_query);
                      break;

                  case 2:
                      counts<unsigned long long, 2, false><<< dimGrid, dimBlock, smemSize >>>(bvectorsPtr, results, states, words_per_vector, vectors_per_config, configs_per_query);
                      break;

                  case 1:
                      counts<unsigned long long, 1, false><<< dimGrid, dimBlock, smemSize >>>(bvectorsPtr, results, states, words_per_vector, vectors_per_config, configs_per_query);
                      break;
                }
        }

    cudaDeviceSynchronize();

} // cudaCallBlockCount

#endif // GPU_UTIL_CU
