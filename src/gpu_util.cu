/***
 * $Id$
 **
 * File: gpu_util.cu
 * Created: Mar 22, 2019
 *
 * Parts of this code have been derived from NVIDIA samples: cuda-8.0/samples/6_Advanced/reduction
 * with the following copyright:
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
 * This file is part of cuSABNAtk.
 *
 */

#ifndef GPU_UTIL_CU
#define GPU_UTIL_CU

#include <cstdint>
#include <stdio.h>

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

__host__ void copyAritiesToDevice(const std::vector<uint64_t>& pArities,
                                  const std::vector<uint64_t>& pAritiesPrefixProd,
                                  const std::vector<uint64_t>& pAritiesPrefixSum) {
    // cudaMemcpyToSymbol(aritiesPtr_, pArities.data(), pArities.size() * sizeof(uint64_t));
    cudaMemcpyToSymbol(aritiesPrefixProdPtr_, pAritiesPrefixProd.data(), pAritiesPrefixProd.size() * sizeof(uint64_t));
    cudaMemcpyToSymbol(aritiesPrefixSumPtr_, pAritiesPrefixSum.data(), pAritiesPrefixSum.size() * sizeof(uint64_t));
} // m_copyAritiesToDevice__


template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void counts(const T* g_idata,
                       T* g_odata,
                       T* g_odataPa,
                       T* g_rdata,
                       unsigned int words_per_vector, // m / 64
                       int vectors_per_config, // number of variables in a query
                       int configs_per_query /* number of configs*/) {
    T* sDataPa = SharedMemory<T>();
    T* sDataTot = &sDataPa[blockSize];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockSize + threadIdx.x;
    unsigned int word_index = i % blockSize; // can't this be tid?

    T totSum = 0;
    T paSum = 0;
    int temp = ((blockIdx.x / aritiesPrefixProdPtr_[0]) % 2);
    T xiBitVect = *(((uint64_t*)g_idata) + ((aritiesPrefixSumPtr_[0] + temp) * words_per_vector) + word_index);

    temp = ((blockIdx.x / aritiesPrefixProdPtr_[1]) % 2);
    T paBitVect = *(((uint64_t*)g_idata) + ((aritiesPrefixSumPtr_[1] + temp) * words_per_vector) + word_index);

    // running sum for all word slices
    for (int p = 2; p < vectors_per_config; ++p) {
        temp = ((blockIdx.x / aritiesPrefixProdPtr_[p]) % 2);
        paBitVect = paBitVect & *(((uint64_t*)g_idata) + ((aritiesPrefixSumPtr_[p] + temp) * words_per_vector) + word_index);
    }

    // if (g_rdata != 0) { // todo can be compile time decision
        // g_rdata[result_index] = localState;
    // }
    xiBitVect &= paBitVect;
    totSum += __popcll(xiBitVect);
    paSum += __popcll(paBitVect);

    // ensure we don't read out of bounds -- this is optimized away for power of 2 sized arrays
    if (nIsPow2 || (tid + blockSize < words_per_vector)) {
        unsigned int word_index_upper_half = word_index + blockSize;
        temp = ((blockIdx.x / aritiesPrefixProdPtr_[0]) % 2);
        xiBitVect = *(((uint64_t*)g_idata) + ((aritiesPrefixSumPtr_[0] + temp) * words_per_vector) + word_index_upper_half);

        temp = ((blockIdx.x / aritiesPrefixProdPtr_[1]) % 2);
        paBitVect = *(((uint64_t*)g_idata) + ((aritiesPrefixSumPtr_[1] + temp) * words_per_vector) + word_index_upper_half);

        for (int p = 2; p < vectors_per_config; p++) {
            temp = ((blockIdx.x / aritiesPrefixProdPtr_[p]) % 2);
            paBitVect = paBitVect & *(((uint64_t*)g_idata) + ((aritiesPrefixSumPtr_[p] + temp) * words_per_vector) + word_index_upper_half);
        }

        // if (g_rdata != 0) { // todo can be compile time decision
            // g_rdata[result_index + blockSize] = localState;
        // }

        xiBitVect &= paBitVect;
        totSum += __popcll(xiBitVect);
        paSum += __popcll(paBitVect);
    }

    // each thread puts its local sum into shared memory
    sDataTot[tid] = totSum;
    sDataPa[tid] = paSum;

    __syncthreads();

    // do reduction in shared mem
    if ((blockSize >= 512) && (tid < 256)) {
      sDataTot[tid] = totSum = totSum + sDataTot[tid + 256];
      sDataPa[tid] = paSum = paSum + sDataPa[tid + 256];
    }

    __syncthreads();

    if ((blockSize >= 256) && (tid < 128)) {
      sDataTot[tid] = totSum = totSum + sDataTot[tid + 128];
      sDataPa[tid] = paSum = paSum + sDataPa[tid + 128];
    }

   __syncthreads();

    if ((blockSize >= 128) && (tid <  64)) {
      sDataTot[tid] = totSum = totSum + sDataTot[tid +  64];
      sDataPa[tid] = paSum = paSum + sDataPa[tid + 64];
    }

  __syncthreads();

  #if (__CUDA_ARCH__ >= 300 )
  if ( tid < 32 ) {
      // Fetch final intermediate sum from 2nd warp
      if (blockSize >=  64) {
        totSum += sDataTot[tid + 32];
        paSum += sDataPa[tid + 32];
      }
      // Reduce final warp using shuffle
      for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        totSum += __shfl_down_sync(0xFFFFFFFF, totSum, offset);
        paSum += __shfl_down_sync(0xFFFFFFFF, paSum, offset);
      }
  }
  #else
  // fully unroll reduction within a single warp
  if ((blockSize >= 64) && (tid < 32)) {
    sDataTot[tid] = totSum = totSum + sDataTot[tid + 32];
    sDataPa[tid] = paSum = paSum + sDataPa[tid + 32];
  }

  __syncthreads();

  if ((blockSize >= 32) && (tid < 16)) {
    sDataTot[tid] = totSum = totSum + sDataTot[tid + 16];
    sDataPa[tid] = paSum = paSum + sDataPa[tid + 16];
  }

  __syncthreads();

  if ((blockSize >= 16) && (tid <  8)) {
    sDataTot[tid] = totSum = totSum + sDataTot[tid +  8];
    sDataPa[tid] = paSum = paSum + sDataPa[tid + 8];
  }

  __syncthreads();

  if ((blockSize >= 8) && (tid <  4)) {
    sDataTot[tid] = totSum = totSum + sDataTot[tid +  4];
    sDataPa[tid] = paSum = paSum + sDataPa[tid + 4];
  }

  __syncthreads();

  if ((blockSize >= 4) && (tid <  2)) {
    sDataTot[tid] = totSum = totSum + sDataTot[tid +  2];
    sDataPa[tid] = paSum = paSum + sDataPa[tid + 2];
  }

  __syncthreads();

  if ((blockSize >= 2) && ( tid <  1)) {
    sDataTot[tid] = totSum = totSum + sDataTot[tid +  1];
    sDataPa[tid] = paSum = paSum + sDataPa[tid + 1];
  }

  __syncthreads();
#endif

  // write result for this block to global mem
  if (tid == 0) {
    g_odata[blockIdx.x] = totSum;
    g_odataPa[blockIdx.x] = paSum;
  }

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
                        const uint64_t *bvectorsPtr,
                        uint64_t *results,
                        uint64_t *resultsPa,
                        uint64_t *states,
                        cudaStream_t streamId) {
  // cudaDeviceSynchronize();

  int threads = nextPow2((words_per_vector + 1) >> 1);

  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(configs_per_query, 1, 1);
  int smemSize = (threads <= 32) ? 2 * threads * sizeof(uint64_t) : threads * sizeof(uint64_t);
  smemSize *= 2;

  if (isPow2(words_per_vector) &&
      (words_per_vector > 1)) // optimize out non power of 2 logic
  {
    switch (threads) {
    case 512:
      counts<uint64_t, 512, true><<<dimGrid, dimBlock, smemSize, streamId>>>(
          bvectorsPtr, results, resultsPa,
          states, words_per_vector, vectors_per_config, configs_per_query);
      break;

    case 256:
      counts<uint64_t, 256, true><<<dimGrid, dimBlock, smemSize, streamId>>>(
          bvectorsPtr, results, resultsPa,
          states, words_per_vector, vectors_per_config, configs_per_query);
      break;

    case 128:
      counts<uint64_t, 128, true><<<dimGrid, dimBlock, smemSize, streamId>>>(
          bvectorsPtr, results, resultsPa,
          states, words_per_vector, vectors_per_config, configs_per_query);
      break;

    case 64:
      counts<uint64_t, 64, true><<<dimGrid, dimBlock, smemSize, streamId>>>(
          bvectorsPtr, results, resultsPa,
          states, words_per_vector, vectors_per_config, configs_per_query);
      break;

    case 32:
      counts<uint64_t, 32, true><<<dimGrid, dimBlock, smemSize, streamId>>>(
          bvectorsPtr, results, resultsPa,
          states, words_per_vector, vectors_per_config, configs_per_query);
      break;

    case 16:
      counts<uint64_t, 16, true><<<dimGrid, dimBlock, smemSize, streamId>>>(
          bvectorsPtr, results, resultsPa,
          states, words_per_vector, vectors_per_config, configs_per_query);
      break;

    case 8:
      counts<uint64_t, 8, true><<<dimGrid, dimBlock, smemSize, streamId>>>(
          bvectorsPtr, results, resultsPa,
          states, words_per_vector, vectors_per_config, configs_per_query);
      break;

    case 4:
      counts<uint64_t, 4, true><<<dimGrid, dimBlock, smemSize, streamId>>>(
          bvectorsPtr, results, resultsPa,
          states, words_per_vector, vectors_per_config, configs_per_query);
      break;

    case 2:
      counts<uint64_t, 2, true><<<dimGrid, dimBlock, smemSize, streamId>>>(
          bvectorsPtr, results, resultsPa,
          states, words_per_vector, vectors_per_config, configs_per_query);
      break;

    case 1:
      counts<uint64_t, 1, true><<<dimGrid, dimBlock, smemSize, streamId>>>(
          bvectorsPtr, results, resultsPa,
          states, words_per_vector, vectors_per_config, configs_per_query);
      break;
    }
  } else {
      switch (threads) {
        case 512:
      counts<uint64_t, 512, false><<<dimGrid, dimBlock, smemSize, streamId>>>(
          bvectorsPtr, results, resultsPa,
          states, words_per_vector, vectors_per_config, configs_per_query);
      break;

    case 256:
      counts<uint64_t, 256, false><<<dimGrid, dimBlock, smemSize, streamId>>>(
          bvectorsPtr, results, resultsPa,
          states, words_per_vector, vectors_per_config, configs_per_query);
      break;

    case 128:
      counts<uint64_t, 128, false><<<dimGrid, dimBlock, smemSize, streamId>>>(
          bvectorsPtr, results, resultsPa,
          states, words_per_vector, vectors_per_config, configs_per_query);
      break;

    case 64:
      counts<uint64_t, 64, false><<<dimGrid, dimBlock, smemSize, streamId>>>(
          bvectorsPtr, results, resultsPa,
          states, words_per_vector, vectors_per_config, configs_per_query);
      break;

    case 32:
      counts<uint64_t, 32, false><<<dimGrid, dimBlock, smemSize, streamId>>>(
          bvectorsPtr, results, resultsPa,
          states, words_per_vector, vectors_per_config, configs_per_query);
      break;

    case 16:
      counts<uint64_t, 16, false><<<dimGrid, dimBlock, smemSize, streamId>>>(
          bvectorsPtr, results, resultsPa,
          states, words_per_vector, vectors_per_config, configs_per_query);
      break;

    case 8:
      counts<uint64_t, 8, false><<<dimGrid, dimBlock, smemSize, streamId>>>(
          bvectorsPtr, results, resultsPa,
          states, words_per_vector, vectors_per_config, configs_per_query);
      break;

    case 4:
      counts<uint64_t, 4, false><<<dimGrid, dimBlock, smemSize, streamId>>>(
          bvectorsPtr, results, resultsPa,
          states, words_per_vector, vectors_per_config, configs_per_query);
      break;

    case 2:
      counts<uint64_t, 2, false><<<dimGrid, dimBlock, smemSize, streamId>>>(
          bvectorsPtr, results, resultsPa,
          states, words_per_vector, vectors_per_config, configs_per_query);
      break;

    case 1:
      counts<uint64_t, 1, false><<<dimGrid, dimBlock, smemSize, streamId>>>(
          bvectorsPtr, results, resultsPa,
          states, words_per_vector, vectors_per_config, configs_per_query);
      break;
    }
  }

  cudaStreamSynchronize(streamId);

} // cudaCallBlockCount

#endif // GPU_UTIL_CU
