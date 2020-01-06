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

__constant__ uint64_t aritiesPtr_[4][10];
__constant__ uint64_t aritiesPrefixProdPtr_[4][11];
__constant__ uint64_t aritiesPrefixSumPtr_[4][10];

template <class T, unsigned int blockSize, bool nIsPow2, bool isSecondStage>
__global__ void counts(const T* inputData,
                       T* outputData,
                       T* outputDataPa,
                       T* intermediateData,
                       unsigned int words_per_vector, // m / 64
                       int variablesCount, // number of variables in a query
                       int configs_per_query, /* number of configs*/
                       int startVariableId,
                       int streamId,
                       cudaStream_t stream,
                       int parentBlockId);

// from cuda samples reduction
CUDA_CALLABLE unsigned int nextPow2(unsigned int x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
} // nextPow2

template <bool isSecondStage>
CUDA_CALLABLE void startKernel(const uint64_t* inputData,
    uint64_t* outputData,
    uint64_t* outputDataPa,
    uint64_t* intermediateData,
    unsigned int words_per_vector, // m / 64
    int variablesCount, // number of variables in a query
    int configs_per_query, /* number of configs*/
    int startVariableId,
    int streamId,
    cudaStream_t stream,
    int threadCount,
    int parentBlockId) {

    dim3 dimBlock(threadCount, 1, 1);
    dim3 dimGrid(configs_per_query, 1, 1);
    int smemSize = (threadCount <= 32) ? 2 * threadCount * sizeof(uint64_t) : threadCount * sizeof(uint64_t);
    if (isSecondStage) {
        smemSize *= 2;
    }
    /*
    if (isPow2(words_per_vector) &&
        (words_per_vector > 1)) // optimize out non power of 2 logic
    {
        switch (threadCount) {
        case 512:
        counts<uint64_t, 512, true, false><<<dimGrid, dimBlock, smemSize>>>(
            bvectorsPtr, results, resultsPa,
            states, words_per_vector, vectors_per_config, configs_per_query, 0, streamId);
        break;

        case 256:
        counts<uint64_t, 256, true, false><<<dimGrid, dimBlock, smemSize>>>(
            bvectorsPtr, results, resultsPa,
            states, words_per_vector, vectors_per_config, configs_per_query, 0, streamId);
        break;
    
        case 128:
        counts<uint64_t, 128, true, false><<<dimGrid, dimBlock, smemSize>>>(
            bvectorsPtr, results, resultsPa,
            states, words_per_vector, vectors_per_config, configs_per_query, 0, streamId);
        break;
    
        case 64:
        counts<uint64_t, 64, true, false><<<dimGrid, dimBlock, smemSize>>>(
            bvectorsPtr, results, resultsPa,
            states, words_per_vector, vectors_per_config, configs_per_query, 0, streamId);
        break;
    
        case 32:
        counts<uint64_t, 32, true, false><<<dimGrid, dimBlock, smemSize>>>(
            bvectorsPtr, results, resultsPa,
            states, words_per_vector, vectors_per_config, configs_per_query, 0, streamId);
        break;
    
        case 16:
        counts<uint64_t, 16, true, false><<<dimGrid, dimBlock, smemSize>>>(
            bvectorsPtr, results, resultsPa,
            states, words_per_vector, vectors_per_config, configs_per_query, 0, streamId);
        break;
    
        case 8:
        counts<uint64_t, 8, true, false><<<dimGrid, dimBlock, smemSize>>>(
            bvectorsPtr, results, resultsPa,
            states, words_per_vector, vectors_per_config, configs_per_query, 0, streamId);
        break;
    
        case 4:
        counts<uint64_t, 4, true, false><<<dimGrid, dimBlock, smemSize>>>(
            bvectorsPtr, results, resultsPa,
            states, words_per_vector, vectors_per_config, configs_per_query, 0, streamId);
        break;
    
        case 2:
        counts<uint64_t, 2, true, false><<<dimGrid, dimBlock, smemSize>>>(
            bvectorsPtr, results, resultsPa,
            states, words_per_vector, vectors_per_config, configs_per_query, 0, streamId);
        break;
    
        case 1:
        counts<uint64_t, 1, true, false><<<dimGrid, dimBlock, smemSize>>>(
            bvectorsPtr, results, resultsPa,
            states, words_per_vector, vectors_per_config, configs_per_query, 0, streamId);
        break;
        }
    } else { */
        // printf("streamid = %d\n", streamId);
    switch (threadCount) {
        case 512:
    counts<uint64_t, 512, false, isSecondStage><<<dimGrid, dimBlock, smemSize, stream>>>(
        inputData, outputData, outputDataPa, intermediateData, words_per_vector, variablesCount, 
        configs_per_query, startVariableId, streamId, stream, parentBlockId);
    break;

    case 256:
    counts<uint64_t, 256, false, isSecondStage><<<dimGrid, dimBlock, smemSize, stream>>>(
        inputData, outputData, outputDataPa, intermediateData, words_per_vector, variablesCount, 
        configs_per_query, startVariableId, streamId, stream, parentBlockId);
    break;

    case 128:
    counts<uint64_t, 128, false, isSecondStage><<<dimGrid, dimBlock, smemSize, stream>>>(
        inputData, outputData, outputDataPa, intermediateData, words_per_vector, variablesCount, 
        configs_per_query, startVariableId, streamId, stream, parentBlockId);
    break;

    case 64:
    counts<uint64_t, 64, false, isSecondStage><<<dimGrid, dimBlock, smemSize, stream>>>(
        inputData, outputData, outputDataPa, intermediateData, words_per_vector, variablesCount, 
        configs_per_query, startVariableId, streamId, stream, parentBlockId);
    break;

    case 32:
    counts<uint64_t, 32, false, isSecondStage><<<dimGrid, dimBlock, smemSize, stream>>>(
        inputData, outputData, outputDataPa, intermediateData, words_per_vector, variablesCount, 
        configs_per_query, startVariableId, streamId, stream, parentBlockId);
    break;

    case 16:
    counts<uint64_t, 16, false, isSecondStage><<<dimGrid, dimBlock, smemSize, stream>>>(
        inputData, outputData, outputDataPa, intermediateData, words_per_vector, variablesCount, 
        configs_per_query, startVariableId, streamId, stream, parentBlockId);
    break;

    case 8:
    counts<uint64_t, 8, false, isSecondStage><<<dimGrid, dimBlock, smemSize, stream>>>(
        inputData, outputData, outputDataPa, intermediateData, words_per_vector, variablesCount, 
        configs_per_query, startVariableId, streamId, stream, parentBlockId);
    break;

    case 4:
    counts<uint64_t, 4, false, isSecondStage><<<dimGrid, dimBlock, smemSize, stream>>>(
        inputData, outputData, outputDataPa, intermediateData, words_per_vector, variablesCount, 
        configs_per_query, startVariableId, streamId, stream, parentBlockId);
    break;

    case 2:
    counts<uint64_t, 2, false, isSecondStage><<<dimGrid, dimBlock, smemSize, stream>>>(
        inputData, outputData, outputDataPa, intermediateData, words_per_vector, variablesCount, 
        configs_per_query, startVariableId, streamId, stream, parentBlockId);
    break;

    case 1:
    counts<uint64_t, 1, false, isSecondStage><<<dimGrid, dimBlock, smemSize, stream>>>(
        inputData, outputData, outputDataPa, intermediateData, words_per_vector, variablesCount, 
        configs_per_query, startVariableId, streamId, stream, parentBlockId);
    break;
    default:
    printf("Unsupported thread count. Exiting.\n");
    }
    // }

    cucheck_dev(cudaGetLastError());
}

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

__host__ void copyAritiesToDevice(int streamId,
                                  const std::vector<uint64_t>& pArities,
                                  const std::vector<uint64_t>& pAritiesPrefixProd,
                                  const std::vector<uint64_t>& pAritiesPrefixSum) {
    cucheck_dev( cudaMemcpyToSymbol(aritiesPtr_, pArities.data(),
        pArities.size() * sizeof(uint64_t), streamId * sizeof(uint64_t) * 10) );
    cucheck_dev( cudaMemcpyToSymbol(aritiesPrefixProdPtr_,
        pAritiesPrefixProd.data(), pAritiesPrefixProd.size() * sizeof(uint64_t), streamId * sizeof(uint64_t) * 11) );
    cucheck_dev( cudaMemcpyToSymbol(aritiesPrefixSumPtr_,
        pAritiesPrefixSum.data(), pAritiesPrefixSum.size() * sizeof(uint64_t), streamId * sizeof(uint64_t) * 10) );
} // m_copyAritiesToDevice__


template <class T, unsigned int blockSize, bool nIsPow2, bool isSecondStage>
__global__ void counts(const T* inputData,
                       T* outputData,
                       T* outputDataPa,
                       T* intermediateData,
                       unsigned int words_per_vector, // m / 64
                       int variablesCount, // number of variables in a query
                       int configs_per_query, /* number of configs*/
                       int startVariableId,
                       int streamId,
                       cudaStream_t stream,
                       int parentBlockId) {
    //TODO: we don't really need two shared mems.
    T* sDataPa = SharedMemory<T>();
    T* sDataTot = &sDataPa[blockSize];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockSize + threadIdx.x;
    unsigned int word_index = i % blockSize; // can't this be tid?
    int intermediateResultIndex;

    //TODO: remove the below constant
    if (isSecondStage) {
        intermediateResultIndex = (streamId * words_per_vector * 32) + (parentBlockId * words_per_vector) + word_index;
    } else {
        intermediateResultIndex = (streamId * words_per_vector * 32) + (blockIdx.x * words_per_vector) + word_index;
    }

    T totSum = 0;
    T paSum = 0;
    T xiBitVect;

    int temp = ((blockIdx.x / aritiesPrefixProdPtr_[streamId][startVariableId]) % aritiesPtr_[streamId][startVariableId]);
    T paBitVect = *(((uint64_t*)inputData) + ((aritiesPrefixSumPtr_[streamId][startVariableId] + temp) * words_per_vector) + word_index);

    // running sum for all word slices
    for (int p = startVariableId + 1; p < min(5 + startVariableId, variablesCount-1); ++p) {
        temp = ((blockIdx.x / aritiesPrefixProdPtr_[streamId][p]) % aritiesPtr_[streamId][p]);
        paBitVect = paBitVect & *(((uint64_t*)inputData) + ((aritiesPrefixSumPtr_[streamId][p] + temp) * words_per_vector) + word_index);
    }

    if (isSecondStage) {
        paBitVect &= intermediateData[intermediateResultIndex];
        temp = ((blockIdx.x / aritiesPrefixProdPtr_[streamId][variablesCount-1]) % aritiesPtr_[streamId][variablesCount-1]);
        xiBitVect = *(((uint64_t*)inputData) + ((aritiesPrefixSumPtr_[streamId][variablesCount-1] + temp) * words_per_vector) + word_index);
        xiBitVect &= paBitVect;
        totSum += __popcll(xiBitVect);
    } else {
        intermediateData[intermediateResultIndex] = paBitVect;
    }

    paSum += __popcll(paBitVect);

    // ensure we don't read out of bounds -- this is optimized away for power of 2 sized arrays
    if (nIsPow2 || (tid + blockSize < words_per_vector)) {
        unsigned int word_index_upper_half = word_index + blockSize;

        temp = ((blockIdx.x / aritiesPrefixProdPtr_[streamId][startVariableId]) % aritiesPtr_[streamId][startVariableId]);
        paBitVect = *(((uint64_t*)inputData) + ((aritiesPrefixSumPtr_[streamId][startVariableId] + temp) * words_per_vector) + word_index_upper_half);

        for (int p = startVariableId + 1; p < min(5 + startVariableId, variablesCount-1); p++) {
            temp = ((blockIdx.x / aritiesPrefixProdPtr_[streamId][p]) % aritiesPtr_[streamId][p]);
            paBitVect = paBitVect & *(((uint64_t*)inputData) + ((aritiesPrefixSumPtr_[streamId][p] + temp) * words_per_vector) + word_index_upper_half);
        }

        if (isSecondStage) {
            paBitVect &= intermediateData[intermediateResultIndex + blockSize];
            temp = ((blockIdx.x / aritiesPrefixProdPtr_[streamId][variablesCount-1]) % aritiesPtr_[streamId][variablesCount-1]);
            xiBitVect = *(((uint64_t*)inputData) + ((aritiesPrefixSumPtr_[streamId][variablesCount-1] + temp) * words_per_vector) + word_index_upper_half);
            xiBitVect &= paBitVect;
            totSum += __popcll(xiBitVect);
        } else {
            intermediateData[intermediateResultIndex + blockSize] = paBitVect;
        }
        paSum += __popcll(paBitVect);
    }

    // each thread puts its local sum into shared memory
    if (isSecondStage) {
        sDataTot[tid] = totSum;
    }
    sDataPa[tid] = paSum;

    __syncthreads();

    // do reduction in shared mem
    if ((blockSize >= 512) && (tid < 256)) {
        if (isSecondStage) {
            sDataTot[tid] = totSum = totSum + sDataTot[tid + 256];
        }
        sDataPa[tid] = paSum = paSum + sDataPa[tid + 256];
    }

    __syncthreads();

    if ((blockSize >= 256) && (tid < 128)) {
        if (isSecondStage) {
            sDataTot[tid] = totSum = totSum + sDataTot[tid + 128];
        }
        sDataPa[tid] = paSum = paSum + sDataPa[tid + 128];
    }

    __syncthreads();

    if ((blockSize >= 128) && (tid <  64)) {
        if (isSecondStage) {
            sDataTot[tid] = totSum = totSum + sDataTot[tid +  64];
        }
        sDataPa[tid] = paSum = paSum + sDataPa[tid + 64];
    }

    __syncthreads();

    #if (__CUDA_ARCH__ >= 300 )
    if ( tid < 32 ) {
        // Fetch final intermediate sum from 2nd warp
        if (blockSize >=  64) {
            if (isSecondStage) {
                totSum += sDataTot[tid + 32];
            }
            paSum += sDataPa[tid + 32];
        }
        // Reduce final warp using shuffle
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            if (isSecondStage) {
                totSum += __shfl_down_sync(0xFFFFFFFF, totSum, offset);
            }
            paSum += __shfl_down_sync(0xFFFFFFFF, paSum, offset);
        }
    }
    #else
    // fully unroll reduction within a single warp
    if ((blockSize >= 64) && (tid < 32)) {
        if (isSecondStage) {
            sDataTot[tid] = totSum = totSum + sDataTot[tid + 32];
        }
        sDataPa[tid] = paSum = paSum + sDataPa[tid + 32];
    }

    __syncthreads();

    if ((blockSize >= 32) && (tid < 16)) {
        if (isSecondStage) {
            sDataTot[tid] = totSum = totSum + sDataTot[tid + 16];
        }
        sDataPa[tid] = paSum = paSum + sDataPa[tid + 16];
    }

    __syncthreads();

    if ((blockSize >= 16) && (tid <  8)) {
        if (isSecondStage) {
            sDataTot[tid] = totSum = totSum + sDataTot[tid +  8];
        }
        sDataPa[tid] = paSum = paSum + sDataPa[tid + 8];
    }

    __syncthreads();

    if ((blockSize >= 8) && (tid <  4)) {
        if (isSecondStage) {
            sDataTot[tid] = totSum = totSum + sDataTot[tid +  4];
        }
        sDataPa[tid] = paSum = paSum + sDataPa[tid + 4];
    }

    __syncthreads();

    if ((blockSize >= 4) && (tid <  2)) {
        if (isSecondStage) {
            sDataTot[tid] = totSum = totSum + sDataTot[tid +  2];
        }
        sDataPa[tid] = paSum = paSum + sDataPa[tid + 2];
    }

    __syncthreads();

    if ((blockSize >= 2) && ( tid <  1)) {
        if (isSecondStage) {
            sDataTot[tid] = totSum = totSum + sDataTot[tid +  1];
        }
        sDataPa[tid] = paSum = paSum + sDataPa[tid + 1];
    }

    __syncthreads();
#endif

    // write result for this block to global mem
    if (tid == 0) {
        //TODO: bypass this logic if number of variables is already small
        if (isSecondStage) {
            //TODO: use global constant here or something else?
            outputData[(streamId*1024) + (parentBlockId * 32) + blockIdx.x] = totSum;
            outputDataPa[(streamId*1024) + (parentBlockId * 32) + blockIdx.x] = paSum;
        } else if (paSum > 0) {
            int threadCount = nextPow2((words_per_vector + 1) >> 1);
            startKernel<true>(inputData,
                outputData,
                outputDataPa,
                intermediateData,
                words_per_vector,
                variablesCount, // number of variables in a query
                32, /* number of configs*/
                5, //TODO: make it safer
                0,
                stream,
                threadCount,
                blockIdx.x);
            //TODO: memset 0 results here
        }
    }

    __syncthreads();
} // counts

inline bool isPow2(unsigned int x) { return ((x & (x - 1)) == 0); }

void cudaCallBlockCount(const uint block_count,
                        const uint per_block_thread_count,
                        const uint words_per_vector,
                        const uint variablesCount,
                        const uint configs_per_query,
                        const uint64_t *bvectorsPtr,
                        uint64_t *results,
                        uint64_t *resultsPa,
                        uint64_t *intermediateData,
                        int streamId,
                        cudaStream_t stream) {

  int threadCount = nextPow2((words_per_vector + 1) >> 1);

  startKernel<false>(bvectorsPtr, results, resultsPa, intermediateData, words_per_vector,
                        variablesCount, configs_per_query, 0, streamId, stream, threadCount, -1);

  cucheck_dev( cudaStreamSynchronize(stream) );

} // cudaCallBlockCount

#endif // GPU_UTIL_CU
