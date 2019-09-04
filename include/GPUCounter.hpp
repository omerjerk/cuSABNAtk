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

#include <cuda_runtime.h>

#include "bit_util.hpp"
#include "gpu_util.cuh"


// TODO: check if these should be hardware dependent
static const int MAX_COUNTS_PER_QUERY = 1024;
static const int MAX_INTERMEDIATE_RESULTS = 128;
static const int MAX_NUM_STREAMS = 50;


struct ResultRecord {
    unsigned long long mCount;
    uint64_t* mResultPtr;
}; // struct ResultRecord


template <int N> class GPUCounter {
public:
    typedef uint_type<N> set_type;

    int n() const { return base_->n_; }

    int m() const { return base_->m_; }

    int r(int i) const { return base_->nodeList_[i].r_; }

    bool is_reorderable() { return false; }

    // FIXME: consider cases when |xa_vect| is greater than 1
    template <typename score_functor>
    void apply(const std::vector<int>& xa_vect, const std::vector<int>& pa_vect, std::vector<score_functor>& F) const {
        std::vector<int> xi;

        std::vector<uint64_t> arities;
        std::vector<uint64_t> aritiesPrefixProd;
        std::vector<uint64_t> aritiesPrefixSum;

        // build arities list
        xi.push_back(xa_vect[0]);
        arities.push_back(r(xi[0]));
        aritiesPrefixProd.push_back(1);
        aritiesPrefixSum.push_back(aritiesPrefixSumGlobal_[xi[0]]);

        int arity = 0;

        for (int i = 0; i < pa_vect.size(); i++) {
            xi.push_back(pa_vect[i]);

            arity = r(xi[i + 1]);

            arities.push_back(arity);
            aritiesPrefixProd.push_back(aritiesPrefixProd[i] * arities[i]);
            aritiesPrefixSum.push_back(aritiesPrefixSumGlobal_[xi[i + 1]]);
        } // for i

        m_copyAritiesToDevice__(arities, aritiesPrefixProd, aritiesPrefixSum);

        long int maxConfigCount = aritiesPrefixProd[aritiesPrefixProd.size() - 1] * arities[arities.size() - 1];
        int streamId = *queryCountPtr % MAX_NUM_STREAMS;

        // call gpu kernel on each subgroup
        cudaCallBlockCount(65535,                          // device limit, not used
                           1024,                           // device limit, not used
                           base_->bitvectorSize_,          // number of words in each bitvector
                           arities.size(),                 // number of variables in one config
                           maxConfigCount,                 // number of configurations
                           aritiesPtr_,
                           aritiesPrefixProdPtr_,
                           aritiesPrefixSumPtr_,
                           base_->nodeList_[0].bitvectors, // starting address of our data
                           resultList_,                    // results array for Nijk
                           resultListPa_,                  // results array for Nij
                           0 /*intermediateResultsPtr_*/,
                           streams[streamId]);             // start of intermediate results

        // TODO: can we overlap this with GPGPU execution
        // execute callback for all non zero results
        for (int i = 0; i < maxConfigCount; ++i) {
            if (resultList_[i] > 0) {
                F[0](resultList_[i], resultListPa_[i]);
            }
        }

        ++*queryCountPtr;
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

    void m_copyAritiesToDevice__(const std::vector<uint64_t>& pArities,
                                 const std::vector<uint64_t>& pAritiesPrefixProd,
                                 const std::vector<uint64_t>& pAritiesPrefixSum) const {
        cudaMemcpy(aritiesPtr_, pArities.data(), pArities.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(aritiesPrefixProdPtr_, pAritiesPrefixProd.data(), pAritiesPrefixProd.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(aritiesPrefixSumPtr_, pAritiesPrefixSum.data(), pAritiesPrefixSum.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
    } // m_copyAritiesToDevice__


    struct node {
        int id;
        int r_;
        uint64_t* bitvectors;
    }; // struct node

    struct base {
        int n_;
        int m_;
        int bitvectorSize_;
        node* nodeList_;
    }; // struct base


    base* base_;
    //contains GPU memory addresses where each particular value of xi starts
    node* nodeList_;

    std::vector<int> aritiesPrefixSumGlobal_;

    //results of each configuration of the given query
    uint64_t* resultList_;
    uint64_t* resultListPa_;
    //intermediate results of a part of query
    //intermediate results of multiple rounds are ANDed to generate the final result
    //this isn't used in case there is only one round
    uint64_t* intermediateResultsPtr_;
    uint64_t* aritiesPtr_;
    uint64_t* aritiesPrefixProdPtr_;
    uint64_t* aritiesPrefixSumPtr_;
    int* xiPtr_;
    int* queryCountPtr;
    cudaStream_t streams[MAX_NUM_STREAMS];

    template <int M, typename Iter>
    friend GPUCounter<M> create_GPUCounter(int, int, Iter);
}; // class GPUCounter


// TODO: add option to select which CUDA device to use
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

    // FIXME: this is not really portable (64 should be replaced with sizeof)
    int bitvectorSize_InWords = static_cast<int>((m + 63) / 64);

    p.base_ = new typename GPUCounter<N>::base;
    p.base_->n_ = n;
    p.base_->m_ = m;
    p.base_->bitvectorSize_ = bitvectorSize_InWords;
    p.base_->nodeList_ = new typename GPUCounter<N>::node[n];

    // determine |ri| of each Xi
    for (int xi = 0; xi < n; ++xi) {
        size = 0;
        std::fill_n(indices, 256, -1);

        for (int j = 0; j < m; ++j) {
            temp = *temp_it++;
            if (indices[temp] == -1) {
                indices[temp] = size++;
            }
        }

        p.base_->nodeList_[xi].r_ = size;
        bitvectorCount += size;
    } // for xi

    int bitvectorWordCount = bitvectorCount * bitvectorSize_InWords;

    uint64_t* bvPtr = nullptr;
    cudaMalloc(&bvPtr, sizeof(uint64_t) * bitvectorWordCount);

    // set bitvector addresses (device addrs) in node list (host mem)
    int offset = 0;

    for (int xi = 0; xi < n; ++xi) {
        p.base_->nodeList_[xi].bitvectors = bvPtr + offset;
        offset += p.base_->nodeList_[xi].r_ * bitvectorSize_InWords;
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
            uint64_t& bitvector = tempBvPtr[offset + (bv_index * bitvectorSize_InWords + (j >> 6))]; // find the block
            bitvector |= (1L << (j & 63)); // set the bit
        }
        offset += p.base_->nodeList_[xi].r_ * bitvectorSize_InWords;
    }

    cudaMemcpy(bvPtr, tempBvPtr, sizeof(uint64_t) * bitvectorWordCount, cudaMemcpyHostToDevice);
    delete[] tempBvPtr;

    // expected size = (number of configurations in the query) * sizeof(uint64_t)
    cudaMallocManaged(&p.resultList_, sizeof(uint64_t) * MAX_COUNTS_PER_QUERY);
    cudaMallocManaged(&p.resultListPa_, sizeof(uint64_t) * MAX_COUNTS_PER_QUERY);
    // cudaMallocManaged(&p.intermediateResultsPtr_, sizeof(uint64_t) * 1024 * bitvectorWordCount);
    // memset(p.intermediateResultsPtr_, 0, sizeof(uint64_t) * 1024 * bitvectorWordCount);

    // TODO: define a more realistic size later
    cudaMalloc(&p.aritiesPtr_, sizeof(uint64_t) * 20);
    cudaMalloc(&p.aritiesPrefixProdPtr_, sizeof(uint64_t) * 20);
    cudaMalloc(&p.aritiesPrefixSumPtr_, sizeof(uint64_t) * 20);
    cudaMalloc(&p.xiPtr_, sizeof(int) * 20);

    p.queryCountPtr = new int;
    *p.queryCountPtr = 0;

    for (int i = 0; i < MAX_NUM_STREAMS; ++i) {
        cudaStreamCreate(&p.streams[i]);
    }

    cudaDeviceSynchronize();

    return p;
} // create_GPUCounter

#endif // GPU_COUNTER_HPP
