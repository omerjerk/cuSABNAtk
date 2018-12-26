/***
*    $Id$
**
*    File: GPUCounter.hpp
*    Created: Oct 22, 2016
*
*    Authors: Marc Greenbaum <marcgree@buffalo.edu>
*    Copyright (c) 2015-2018 SCoRe Group http://www.score-group.org/
*    Distributed under the MIT License.
*    See accompanying file LICENSE.
*/

#ifndef GPU_COUNTER_HPP
#define GPU_COUNTER_HPP

#include <cuda_runtime.h>
#include "gpu_util.cuh"
#include "bit_util.hpp"
#include "TaskEnumerator.hpp"

static const int MAX_COUNTS_PER_QUERY = 64;
static const int MAX_INTERMEDIATE_RESULTS = 128;

void printTask(std::vector<TaskEnumerator::NodeState> pTask, char pTerm = '\n') {
  for(auto itr = pTask.begin(); itr != pTask.end(); itr++)
  {
    if(itr->mActive) {
      printf("X%d=%d ", itr->mNode, itr->mState);
    } else{
      printf("X%d=%s ", itr->mNode, "x");
    }
  }
  printf("%c", pTerm);
}

struct ResultRecord
{
  unsigned long long mCount;
  unsigned long long* mResultPtr;
};

template<int N>
class GPUCounter {
public:
  typedef uint_type<N> set_type;

  int n() const { return base_->n_; }

  int m() const { return base_->m_; }

  int r(int i) const { return base_->nodeList_[i].r_; }

  bool is_reorderable() { return false; }

  template<typename score_functor, typename data_type>
  void apply(const set_type &xi, const set_type &pa, const std::vector<data_type> &state_xi,
    const std::vector<data_type> &state_pa, std::vector<score_functor> &F) const {
      std::vector<uint64_t*> bitvectorsForIntersect;

      // consolidate this to one loop for xi/pa
      auto xi_vect = as_vector(xi);
      for(int i = 0; i < xi_vect.size(); i++) {
        int id = xi_vect[i];
        int state = state_xi[i];
        uint64_t* bv_ptr = getBvPtr(id, state);
        bitvectorsForIntersect.push_back(bv_ptr);
      }

      auto pa_vect = as_vector(pa);
      for(int i = 0; i < pa_vect.size(); i++) {
        int id = pa_vect[i];
        int state = state_pa[i];
        uint64_t* bv_ptr = getBvPtr(id, state);
        bitvectorsForIntersect.push_back(bv_ptr);
      }

      // todo setup block sizes...
      const int block_size = 1024;//deviceProp.maxThreadsPerBlock;
      const uint max_block_count = 65535;

      copyBvListToDevice(bitvectorsForIntersect);

      cudaCallBlockCount(
        max_block_count, //todo
        block_size, //threads per block
        base_->bitvectorSize_, // word count
        bitvectorsForIntersect.size(), // state count
        1, // always one configuration
        countListPtr, // bit vectors for counting
        resultList_, // result
        0); // where to put intermediate result

      F[0](resultList_[0],0);
  } // end apply - state specific

  template<typename score_functor>
  void apply(const set_type &xi, const set_type &pa, std::vector<score_functor> &F) const {
    std::vector<int> nodeList;
    int activeRoundTwoCount = 0;
    std::vector<uint64_t*> bitvectorsForIntersect;
    std::vector<ResultRecord> resultList;
    auto xa_vect = as_vector(xi);
    auto pa_vect = as_vector(pa);

    // build node list
    nodeList.push_back(r(xa_vect[0]));
    for(int i = 0; i < pa_vect.size(); i++) {
      xa_vect.push_back(pa_vect[i]);
      nodeList.push_back(r(pa_vect[i]));
    }

    // initialize task enumerator
    int configCount = 1;
    TaskEnumerator te(nodeList, 6);
    int roundOneGroupCount = te.getRoundOneGroupCount();
    int previousGroupCount = roundOneGroupCount;
    TaskEnumerator::TaskList tl;
    int subGroup = 0;

    // execute round one
    while(roundOneGroupCount != 1) {
      bitvectorsForIntersect.clear();
      std::vector<TaskEnumerator::NodeState> nextTask;

      // build round one sub groups
      for(int i = 0; i < roundOneGroupCount; i++) {
        nextTask = te.next1();
        tl.push_back(nextTask);

        // build each sub group query
        for(int j = 0; j < nextTask.size(); j++){
          if(nextTask[j].mActive){
            uint64_t* bv_ptr = this->getBvPtr(nextTask[j].mNode,nextTask[j].mState);
            bitvectorsForIntersect.push_back(bv_ptr);
          }
        }
      }

      unsigned long long* intermediateStatesRoundPtr =
      (intermediateResultsPtr + subGroup * previousGroupCount * base_->bitvectorSize_);

      copyBvListToDevice(bitvectorsForIntersect);

      // call gpu kernel on each subgroup
      cudaCallBlockCount(
        65535, //device limit, not used
        1024, //device limit, not used
        base_->bitvectorSize_, // word count
        bitvectorsForIntersect.size() / roundOneGroupCount, // state count
        roundOneGroupCount, // config count
        countListPtr, // pointer to bitvector pointers
        resultList_, // results array
        intermediateStatesRoundPtr); // start of intermediate results

      // add subgroup results to task list
      for(int i = 0; i < roundOneGroupCount; i++) {
        ResultRecord tempResult{resultList_[i], intermediateStatesRoundPtr + i * base_->bitvectorSize_}; // intermediate result
        resultList.push_back(tempResult);
      }
      previousGroupCount = roundOneGroupCount;
      roundOneGroupCount = te.getRoundOneGroupCount();
      subGroup++;
    }

    if(subGroup > 1) {
      // round two. Execute all non-zero round 1 combinations
      activeRoundTwoCount = 0;
      bitvectorsForIntersect.clear();
      for(int i = 0; i < te.getTaskCount(); i++) {
          std::vector<int> combo = te.next2();

          bool nonZero = true;
          for(int state : combo) {
            nonZero &= (resultList[state].mCount > 0);
          }

          if(nonZero) {
            activeRoundTwoCount++;
            for(int nonZeroState : combo) {
              bitvectorsForIntersect.push_back((uint64_t*)resultList[nonZeroState].mResultPtr);
            }
          }
      }

      copyBvListToDevice(bitvectorsForIntersect);

      cudaCallBlockCount(
              65535, //device limit, not used
              1024, //device limit, not used
              base_->bitvectorSize_, // word count
              bitvectorsForIntersect.size() / activeRoundTwoCount, // state count
              activeRoundTwoCount, // config count
              countListPtr, // pointer to bitvector pointers
              resultList_, // results array
              0); // no intermediate results
    }

    // execute callback for all non zero results
    int resultCount = subGroup > 1 ? activeRoundTwoCount : te.getTaskCount();
    for(int i = 0; i < resultCount; i++) {
      if(resultList_[i] > 0){
        F[0](resultList_[i],i);
      }
    }
    return;
  } // end apply - non zero

private:

  uint64_t* getBvPtr(int pNode, int pState) const {
    uint64_t* resultPtr = 0;
    if(pNode < n() && pState < r(pNode)) {
      resultPtr = base_->nodeList_[pNode].bitvectors + (pState * base_->bitvectorSize_);
    }
    return resultPtr;
  }

  void copyBvListToDevice(std::vector<uint64_t*>& pBitvectorsForIntersect) const {
    memcpy(countListPtr,
      pBitvectorsForIntersect.data(),
      pBitvectorsForIntersect.size() * sizeof(uint64_t*));
  }

  int bvSize() {
    return this->base_.bitvectorSize_;
  }

  struct node {
    int id;
    int r_;
    uint64_t *bitvectors;
    node() : id(0),
    r_(0),
    bitvectors(0){};
  };

  struct base {
    int n_;
    int m_;
    int bitvectorSize_;
    node *nodeList_;
  };

  base *base_;
  node *nodeList_;
  int maxNodesPerTask_ = 6;

  unsigned long long* resultList_;
  const unsigned long long** countListPtr; // copy address list into gpu memory
  unsigned long long* intermediateResultsPtr; // intermediate results list in gpu memory
  template<int M, typename Iter>
  friend GPUCounter<M> create_GPUCounter(int, int, Iter);
};

template<int N, typename Iter>
GPUCounter<N> create_GPUCounter(int n, int m, Iter it) {
  GPUCounter<N> p;

  Iter temp_it = it;
  int indices[256];
  int temp;
  int size;
  int bitvectorCount = 0;
  int bitvectorSize_InWords = (int) ((m + 63) / 64);
  typename GPUCounter<N>::base temp_base = {n, m, bitvectorSize_InWords, 0};
  temp_base.nodeList_ = new typename GPUCounter<N>::node[n];

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

    temp_base.nodeList_[xi].r_ = size;
    bitvectorCount += size;
  }

  int adminBlockSize = sizeof(typename GPUCounter<N>::base) + n * sizeof(typename GPUCounter<N>::node);
  int bitvectorWordCount = bitvectorCount * bitvectorSize_InWords;

  p.base_ = (typename GPUCounter<N>::base*) new char[adminBlockSize];
  memset(p.base_, 0, adminBlockSize);

  uint64_t* bvPtr;
  cudaMalloc(&bvPtr, sizeof(uint64_t) * bitvectorWordCount);

  // copy base and node list to GPUCounter
  memcpy(p.base_, &temp_base, sizeof(typename GPUCounter<N>::base));
  memcpy((char*) p.base_ + sizeof(typename GPUCounter<N>::base), temp_base.nodeList_, n * sizeof(typename GPUCounter<N>::node));
  p.base_->nodeList_ = (typename GPUCounter<N>::node *) ((char *) p.base_ + sizeof(typename GPUCounter<N>::base));
  delete[] temp_base.nodeList_;

  // set bitvector addresses (device addrs) in node list (host mem)
  int offset = 0;
  for (int xi = 0; xi < n; ++xi) {
    p.base_->nodeList_[xi].bitvectors = bvPtr + offset;
    offset += p.base_->nodeList_[xi].r_ * bitvectorSize_InWords;
  }

  printf("base and node footprint is %d bytes\n", adminBlockSize);
  printf("database footprint is %lu bytes\n", sizeof(uint64_t) * bitvectorWordCount);

  // build bitvectors for each Xi and copy into device
  temp_it = it;
  for (int xi = 0; xi < n; ++xi) {
    uint64_t* tempBvsPtr = new uint64_t[p.base_->nodeList_[xi].r_ * bitvectorSize_InWords];
    std::fill_n(tempBvsPtr, p.base_->nodeList_[xi].r_ * bitvectorSize_InWords, 0);

    for (int j = 0; j < m; ++j) {
      temp = *temp_it++;
      int bv_index = temp;
      uint64_t& bitvector = tempBvsPtr[bv_index * bitvectorSize_InWords + (j>>6)]; // find the block
      bitvector |= (1L<< (j & 63)); // set the bit
    }

    int bvsSize = p.base_->nodeList_[xi].r_ * bitvectorSize_InWords * sizeof(uint64_t);
    cudaMemcpy(p.base_->nodeList_[xi].bitvectors, tempBvsPtr, bvsSize, cudaMemcpyHostToDevice);
    delete[] tempBvsPtr;
  }

  cudaMallocManaged(&p.resultList_, p.base_->bitvectorSize_ * sizeof(unsigned long long) * MAX_COUNTS_PER_QUERY);
  cudaMallocManaged(&p.countListPtr, sizeof(unsigned long long*) * 256);
  cudaMalloc(&p.intermediateResultsPtr, p.base_->bitvectorSize_ * sizeof(unsigned long long) * MAX_INTERMEDIATE_RESULTS);
  cudaMemset(p.intermediateResultsPtr, 0, p.base_->bitvectorSize_ * sizeof(unsigned long long) * MAX_INTERMEDIATE_RESULTS);
  cudaDeviceSynchronize();
  return p;
} // create_GPUCounter

#endif // GPU_COUNTER_HPP
