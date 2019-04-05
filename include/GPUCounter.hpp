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

#include "TaskEnumerator.hpp"
#include "bit_util.hpp"
#include "gpu_util.cuh"

//TODO: check if these should be hardware dependent
static const int MAX_COUNTS_PER_QUERY = 64;
static const int MAX_INTERMEDIATE_RESULTS = 128;


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
    void apply(const set_type &xi, const set_type &pa, std::vector<score_functor> &F) const {
        int activeRoundTwoCount = 0;

        std::vector<uint64_t*> bitvectorsForIntersect;
        std::vector<ResultRecord> resultList;

        auto xa_vect = as_vector(xi);
        auto pa_vect = as_vector(pa);

        // build node list
        std::vector<int> nodeList;
        nodeList.push_back(r(xa_vect[0]));
        for (int i = 0; i < pa_vect.size(); i++) {
            nodeList.push_back(r(pa_vect[i]));
        }

        // initialize task enumerator
        int configCount = 1;
        // FIXME: this should be either dynamic or should be documented
        TaskEnumerator te(nodeList, 6);
        int roundOneGroupCount = te.getRoundOneGroupCount();
        int previousGroupCount = roundOneGroupCount;
        TaskEnumerator::TaskList tl;
        int subGroup = 0;

        // execute round one
        while (roundOneGroupCount != 1) {
            bitvectorsForIntersect.clear();
            std::vector<TaskEnumerator::NodeState> nextTask;

            // build round one sub groups
            for (int i = 0; i < roundOneGroupCount; ++i) {
                nextTask = te.next1();
                tl.push_back(nextTask);

                // build each sub group query
                for (int j = 0; j < nextTask.size(); ++j) {
                    if (nextTask[j].mActive){
                        uint64_t* bv_ptr = this->m_getBvPtr__(nextTask[j].mNode,nextTask[j].mState);
                        bitvectorsForIntersect.push_back(bv_ptr);
                    }
                }
            }

            uint64_t* intermediateStatesRoundPtr =
                (intermediateResultsPtr_ + subGroup * previousGroupCount * base_->bitvectorSize_);

            m_copyBvListToDevice__(bitvectorsForIntersect);

            // call gpu kernel on each subgroup
            cudaCallBlockCount(
                               65535, // device limit, not used
                               1024, // device limit, not used
                               base_->bitvectorSize_, // word count
                               bitvectorsForIntersect.size() / roundOneGroupCount, // state count
                               roundOneGroupCount, // config count
                               countListPtr_, // pointer to bitvector pointers
                               resultList_, // results array
                               intermediateStatesRoundPtr); // start of intermediate results

            // add subgroup results to task list
            for (int i = 0; i < roundOneGroupCount; ++i) {
                ResultRecord tempResult{resultList_[i], intermediateStatesRoundPtr + i * base_->bitvectorSize_}; // intermediate result
                resultList.push_back(tempResult);
            }

            previousGroupCount = roundOneGroupCount;
            roundOneGroupCount = te.getRoundOneGroupCount();
            subGroup++;
        }

        if (subGroup > 1) {
            // Round two: execute all non-zero round 1 combinations
            activeRoundTwoCount = 0;
            bitvectorsForIntersect.clear();
            for(int i = 0; i < te.getTaskCount(); i++) {
                std::vector<int> combo = te.next2();

                bool nonZero = true;
                for(int state : combo) nonZero &= (resultList[state].mCount > 0);

                if (nonZero) {
                    activeRoundTwoCount++;
                    for (int nonZeroState : combo) {
                        bitvectorsForIntersect.push_back((uint64_t*)resultList[nonZeroState].mResultPtr);
                    }
                }
            }

            m_copyBvListToDevice__(bitvectorsForIntersect);

            cudaCallBlockCount(65535, // device limit, not used
                               1024, // device limit, not used
                               base_->bitvectorSize_, // word count
                               bitvectorsForIntersect.size() / activeRoundTwoCount, // state count
                               activeRoundTwoCount, // config count
                               countListPtr_, // pointer to bitvector pointers
                               resultList_, // results array
                               0); // no intermediate results
        }

        // execute callback for all non zero results
        int resultCount = subGroup > 1 ? activeRoundTwoCount : te.getTaskCount();
        for (int i = 0; i < resultCount; ++i) {
            if (resultList_[i] > 0) F[0](resultList_[i], i);
        }

    } // end apply - non zero

private:
    uint64_t* m_getBvPtr__(int pNode, int pState) const {
        uint64_t* resultPtr = 0;
        if(pNode < n() && pState < r(pNode)) {
            resultPtr = base_->nodeList_[pNode].bitvectors + (pState * base_->bitvectorSize_);
        }
        return resultPtr;
    } // m_getBvPtr__

    void m_copyBvListToDevice__(std::vector<uint64_t*>& pBitvectorsForIntersect) const {
        memcpy(countListPtr_,
               pBitvectorsForIntersect.data(),
               pBitvectorsForIntersect.size() * sizeof(uint64_t*));
    } // m_copyBvListToDevice__

    int m_bvSize__() const { return this->base_.bitvectorSize_; }


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
    node* nodeList_;
    int maxNodesPerTask_ = 6;

    uint64_t* resultList_;
    const uint64_t** countListPtr_; // copy address list into gpu memory
    uint64_t* intermediateResultsPtr_; // intermediate results list in gpu memory

    template <int M, typename Iter>
    friend GPUCounter<M> create_GPUCounter(int, int, Iter);
}; // class GPUCounter


template <int N, typename Iter> GPUCounter<N> create_GPUCounter(int n, int m, Iter it) {
    GPUCounter<N> p;

    int indices[256];
    int temp;
    int size;

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
    }

    int bitvectorWordCount = bitvectorCount * bitvectorSize_InWords;

    uint64_t* bvPtr;
    cudaMalloc(&bvPtr, sizeof(uint64_t) * bitvectorWordCount);

    // set bitvector addresses (device addrs) in node list (host mem)
    int offset = 0;
    for (int xi = 0; xi < n; ++xi) {
        p.base_->nodeList_[xi].bitvectors = bvPtr + offset;
        offset += p.base_->nodeList_[xi].r_ * bitvectorSize_InWords;
    }

    // build bitvectors for each Xi and copy into device
    uint64_t* tempBvPtr = new uint64_t[bitvectorWordCount];
    std::fill_n(tempBvPtr, bitvectorWordCount, 0);
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
    cudaMemcpy(p.base_->nodeList_[0].bitvectors, tempBvPtr, bitvectorWordCount, cudaMemcpyHostToDevice);
    delete[] tempBvPtr;

    //expected size = (number of configurations in the query) * sizeof(uint64_t)
    cudaMallocManaged(&p.resultList_, sizeof(uint64_t) * MAX_COUNTS_PER_QUERY);
    //expected size = (number of configs in a query) * (number of variables in a query) * sizeof(uint64_t*)
    cudaMallocManaged(&p.countListPtr_, sizeof(uint64_t*) * 256);
    //expected size = (bitVectorSize) * (max number of configurations in a round) * sizeof(uint64_t)
    cudaMalloc(&p.intermediateResultsPtr_, p.base_->bitvectorSize_ * sizeof(uint64_t) * MAX_INTERMEDIATE_RESULTS);

    cudaMemset(p.intermediateResultsPtr_, 0, p.base_->bitvectorSize_ * sizeof(uint64_t) * MAX_INTERMEDIATE_RESULTS);

    cudaDeviceSynchronize();

    return p;
} // create_GPUCounter

#endif // GPU_COUNTER_HPP
