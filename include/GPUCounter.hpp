/***
 *    $Id$
 **
 *    File: GPUCounter.hpp
 *    Created: Oct 22, 2016
 *
 *    Authors: Marc Greenbaum <marcgree@buffalo.edu>
 *             Mohammad Umair <m39@buffalo.edu>
 *    Copyright (c) 2015-2019 SCoRe Group http://www.score-group.org/
 *    Distributed under the MIT License.
 *    See accompanying file LICENSE.
 */

#ifndef GPU_COUNTER_HPP
#define GPU_COUNTER_HPP

#include <cinttypes>
#include <vector>
#include <atomic>

#include <cuda_runtime.h>
#include <omp.h>

#include "bit_util.hpp"
#include "gpu_util.cuh"
#include "RadCounter.hpp"

static const int MAX_COUNTS_PER_QUERY = 1024;
static const int MAX_VARS_FIRST_STAGE = 5;
static const int MAX_COUNTS_FIRST_STAGE = 1 << 5; //considering all variables have arity of 2

#define STREAM_COUNT 2
static std::atomic_flag isStreamFree[STREAM_COUNT] = {ATOMIC_FLAG_INIT};

template <int N> class GPUCounter {
public:
    using set_type = uint_type<N>;

    ~GPUCounter() {
        cudaFree(&base_->bvPtr_);
        delete[] nodeList_;
        delete base_;
    } // GPUCounter

    void initRadCounter(RadCounter<N>* rad) {
        this->radCounter = rad;
    }

    int n() const { return base_->n_; }

    int m() const { return base_->m_; }

    int r(int i) const { return base_->nodeList_[i].r_; }

    bool is_reorderable() { return false; }

    // FIXME: consider cases when |xa_vect| is greater than 1
    //        SABNAtk assumes that apply() is const, to allow CUDA concurrency, we are changing that
    template <typename score_functor>
    void apply(const std::vector<int>& xa_vect, const std::vector<int>& pa_vect, std::vector<score_functor>& F) {

        int streamId = -1;
        for (int i = 0; i < STREAM_COUNT; ++i) {
            if (!isStreamFree[i].test_and_set()) {
                streamId = i;
                break;
            }
        }
        if (streamId == -1) {
            // printf("sending to CPU\n");
            return this->radCounter->apply(xa_vect, pa_vect, F);
        }
        // printf("sending to GPU %d\n", streamId);
        int paSize = pa_vect.size();
        std::vector<int> xi(1 + paSize);

        std::vector<uint64_t> arities;
        std::vector<uint64_t> aritiesPrefixProd;
        std::vector<uint64_t> aritiesPrefixSum;

        // build arities list
        for (int i = 0; i < paSize; i++) {
            xi[i] = pa_vect[i];
            arities.push_back(r(xi[i]));
            if (i%MAX_VARS_FIRST_STAGE == 0) {
                aritiesPrefixProd.push_back(1);
            } else {
                aritiesPrefixProd.push_back(aritiesPrefixProd[i-1] * arities[i-1]);
            }
            aritiesPrefixSum.push_back(aritiesPrefixSumGlobal_[xi[i]]);
        } // for i
        xi[paSize] = xa_vect[0];
        arities.push_back(r(xi[paSize]));
        aritiesPrefixProd.push_back(aritiesPrefixProd[paSize-1] * arities[paSize-1]);
        aritiesPrefixSum.push_back(aritiesPrefixSumGlobal_[xi[paSize]]);
        // aritiesPrefixProd.push_back(aritiesPrefixProd[paSize] * arities[paSize]);

        // int maxConfigCount = (paSize + 1) > MAX_VARS_FIRST_STAGE ? aritiesPrefixProd[MAX_VARS_FIRST_STAGE-1] * arities[MAX_VARS_FIRST_STAGE-1]
                                    //  : aritiesPrefixProd[paSize+1];
        // std::cout<<"max config count = "<<maxConfigCount<<std::endl;

        copyAritiesToDevice(streamId, arities, aritiesPrefixProd, aritiesPrefixSum);

        // call gpu kernel on each subgroup
        cudaCallBlockCount(65535,                          // device limit, not used
                           1024,                           // device limit, not used
                           base_->bitvectorSize_,          // number of words in each bitvector
                           arities.size(),                 // number of variables in one config
                           32,                 // number of configurations
                           base_->nodeList_[0].bitvectors, // starting address of our data
                           resultList_,                    // results array for Nijk
                           resultListPa_,                  // results array for Nij
                           intermediaResult_,              // memory for intermediate results
                           streamId,
                           streams[streamId]);

        //TODO: fix this condition
        for (int i = 0; i < 1024; ++i) {
            if (resultList_[(streamId * MAX_COUNTS_PER_QUERY) + i] > 0) {
                F[0](resultList_[(streamId * MAX_COUNTS_PER_QUERY) + i], resultListPa_[(streamId * MAX_COUNTS_PER_QUERY) + i]);
            }
        }

        isStreamFree[streamId].clear();

    } // apply

private:
    int m_bvSize__() const { return this->base_.bitvectorSize_; }

    // given the variable number and its state, this function returns the GPU address
    // where that particular bitvector starts
    uint64_t* m_getBvPtr__(int pNode, int pState) const {
        uint64_t* resultPtr = nullptr;
        if (pNode < n() && pState < r(pNode)) {
            resultPtr = base_->nodeList_[pNode].bitvectors + (pState * base_->bitvectorSize_);
        }
        return resultPtr;
    } // m_getBvPtr__

    struct node {
        int id;
        int r_;
        uint64_t* bitvectors;
    }; // struct node

    struct base {
        int n_;
        int m_;
        int bitvectorSize_;
        uint64_t* bvPtr_;
        node* nodeList_;
    }; // struct base

    // minimal description of the input data
    base* base_ = nullptr;

    // contains GPU memory addresses where each particular value of xi starts
    node* nodeList_ = nullptr;

    std::vector<int> aritiesPrefixSumGlobal_;

    // results of each configuration of the given query
    uint64_t* resultList_;
    uint64_t* resultListPa_;
    uint64_t* intermediaResult_;

    std::vector<cudaStream_t> streams;

    RadCounter<N>* radCounter;

    template <int M, typename Iter>
    friend GPUCounter<M> create_GPUCounter(int, int, Iter);
}; // class GPUCounter


// TODO: add option to select which CUDA device to use
//       we should be also configuring number of streams automatically
template <int N, typename Iter> GPUCounter<N> create_GPUCounter(int n, int m, Iter it) {
    int devicesCount = 0;
    cudaGetDeviceCount(&devicesCount);

    if (devicesCount == 0) throw std::runtime_error("no CUDA capable devices found");

    GPUCounter<N> p;

    int indices[256];
    int temp = 0;
    int size = 0;

    Iter temp_it = it;
    int bitvectorCount = 0;

    // we use 64bit words
    // you will see constants: shifting by 6, and 63 and 64
    int bitvectorSize = (m + 63) / 64;

    p.base_ = new typename GPUCounter<N>::base;
    p.base_->n_ = n;
    p.base_->m_ = m;
    p.base_->bitvectorSize_ = bitvectorSize;
    p.base_->nodeList_ = new typename GPUCounter<N>::node[n];

    // determine |ri| of each Xi
    // FIXME: report and terminate if arities exceed our GPU capability
    for (int xi = 0; xi < n; ++xi) {
        size = 0;
        std::fill_n(indices, 256, -1);

        for (int j = 0; j < m; ++j) {
            temp = *temp_it++;
            if (indices[temp] == -1) indices[temp] = size++;
        }

        p.base_->nodeList_[xi].r_ = size;
        bitvectorCount += size;
    } // for xi

    int bitvectorWordCount = bitvectorCount * bitvectorSize;

    uint64_t* bvPtr = nullptr;
    cudaMalloc(&bvPtr, sizeof(uint64_t) * bitvectorWordCount);

    p.base_->bvPtr_ = bvPtr;

    // set bitvector addresses (device addrs) in node list (host mem)
    int offset = 0;

    for (int xi = 0; xi < n; ++xi) {
        p.base_->nodeList_[xi].bitvectors = bvPtr + offset;
        offset += p.base_->nodeList_[xi].r_ * bitvectorSize;
        if (!xi) {
            p.aritiesPrefixSumGlobal_.push_back(0);
        } else {
            p.aritiesPrefixSumGlobal_.push_back(p.aritiesPrefixSumGlobal_[p.aritiesPrefixSumGlobal_.size() - 1] + p.base_->nodeList_[xi - 1].r_);
        }
    } // for xi

    // build bitvectors for each Xi and copy into device
    uint64_t* tempBvPtr = new uint64_t[bitvectorWordCount];
    memset(tempBvPtr, 0, sizeof(uint64_t) * bitvectorWordCount);

    temp_it = it;
    offset = 0;

    for (int xi = 0; xi < n; ++xi) {
        for (int j = 0; j < m; ++j) {
            temp = *temp_it++;
            int bv_index = temp;
            uint64_t& bitvector = tempBvPtr[offset + (bv_index * bitvectorSize + (j >> 6))]; // find the block
            bitvector |= (1L << (j & 63)); // set the bit
        }
        offset += p.base_->nodeList_[xi].r_ * bitvectorSize;
    }

    cucheck_dev( cudaMemcpy(bvPtr, tempBvPtr, sizeof(uint64_t) * bitvectorWordCount, cudaMemcpyHostToDevice) );
    delete[] tempBvPtr;

    // expected size = (number of configurations in the query) * sizeof(uint64_t)
    cucheck_dev( cudaMallocManaged(&p.resultList_, sizeof(uint64_t) * MAX_COUNTS_PER_QUERY * STREAM_COUNT) );
    cucheck_dev( cudaMallocManaged(&p.resultListPa_, sizeof(uint64_t) * MAX_COUNTS_PER_QUERY * STREAM_COUNT) );

    cucheck_dev( cudaMalloc(&p.intermediaResult_, sizeof(uint64_t) * bitvectorSize * 32 * STREAM_COUNT) );

    p.streams.resize(STREAM_COUNT);

    for (int i = 0; i < STREAM_COUNT; ++i) {
        cucheck_dev( cudaStreamCreate(&p.streams[i]) );
    }

    cucheck_dev( cudaDeviceSynchronize() );

    return p;
} // create_GPUCounter

#endif // GPU_COUNTER_HPP
