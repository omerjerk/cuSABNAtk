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
static const int MAX_INTERMEDIATE_RESULTS = 1024;

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
  unsigned long long* mResultPtr; // todo make ptr to intermediate result vector
};

template<int N>
class GPUCounter {
public:
  typedef uint_type<N> set_type;

  int n() const { return base_ptr->n; }

  int m() const { return base_ptr->m; }

  int r(int i) const { return base_ptr->node_ptr[i].r; }

  bool is_reorderable() { return false; }

  template<typename score_functor, typename data_type>
  void apply(const set_type &xi, const set_type &pa, const std::vector<data_type> &state_xi,
    const std::vector<data_type> &state_pa, std::vector<score_functor> &F) const {
      std::vector<uint64_t*> bitvectors_for_intersect;

      auto xi_vect = as_vector(xi);
      for(int i = 0; i < xi_vect.size(); i++) {
        int id = xi_vect[i];
        int state = state_xi[i];
        uint64_t* bv_ptr = &this->base_ptr->node_ptr[id].bitvectors[state * base_ptr->bitvector_size];
        bitvectors_for_intersect.push_back(bv_ptr);
      }

      auto pa_vect = as_vector(pa);
      for(int i = 0; i < pa_vect.size(); i++) {
        int id = pa_vect[i];
        int state = state_pa[i];
        uint64_t* bv_ptr = &this->base_ptr->node_ptr[id].bitvectors[state * base_ptr->bitvector_size];
        bitvectors_for_intersect.push_back(bv_ptr);
      }

      // todo setup block sizes...
      const int block_size = 1024;//deviceProp.maxThreadsPerBlock;
      const uint max_block_count = 65535;
      uint num_blocks = (base_ptr->bitvector_size + block_size - 1) / block_size;

      memcpy(countListPtr, bitvectors_for_intersect.data(), bitvectors_for_intersect.size() * sizeof(unsigned long long*));

      cudaDeviceSynchronize();

      cudaCallBlockCount(
        max_block_count, //todo
        block_size, //threads per block
        base_ptr->bitvector_size, // word count
        bitvectors_for_intersect.size(), // state count
        1, // always one configuration
        countListPtr, // bit vectors for counting
        reduce, // result
        0); // where to put intermediate result

      cudaDeviceSynchronize();

      F[0](reduce[0],0);
  } // end apply - state specific

  template<typename score_functor>
  void apply(const set_type &xi, const set_type &pa, std::vector<score_functor> &F) const {
    std::vector<int> nodeList;
    int activeRoundTwoCount = 0;
    std::vector<uint64_t*> bitvectors_for_intersect;
    std::vector<ResultRecord> resultList;
    auto xa_vect = as_vector(xi);
    auto pa_vect = as_vector(pa);

    // build node list
    nodeList.push_back(r(xa_vect[0]));
    for(int i = 0; i < pa_vect.size(); i++){
      xa_vect.push_back(pa_vect[i]);
      nodeList.push_back(r(pa_vect[i]));
    }

    // initialize task enumerator
    int configCount = 1;
    TaskEnumerator te(nodeList, maxNodesPerTask);
    int roundOneGroupCount = te.getRoundOneGroupCount();
    TaskEnumerator::TaskList tl;
    int subGroup = 0;

    // execute round one
    while(roundOneGroupCount != 1){
      bitvectors_for_intersect.clear();
      std::vector<TaskEnumerator::NodeState> nextTask;

      // build round one sub groups
      for(int i = 0; i < roundOneGroupCount; i++) {
        nextTask = te.next1();
        tl.push_back(nextTask);

        // build each sub group query
        for(int j = 0; j < nextTask.size(); j++){
          if(nextTask[j].mActive){
            uint64_t* bv_ptr =
            &this->base_ptr->node_ptr[xa_vect[nextTask[j].mNode]].bitvectors[nextTask[j].mState * base_ptr->bitvector_size];
            bitvectors_for_intersect.push_back(bv_ptr);
          }
        }
      }

      unsigned long long* intermediateStatesRoundPtr =
      (intermediateResultsPtr + subGroup * roundOneGroupCount * base_ptr->bitvector_size);

      memcpy(countListPtr, bitvectors_for_intersect.data(), bitvectors_for_intersect.size() * sizeof(unsigned long long*));
      cudaDeviceSynchronize();

      // call gpu kernel on each subgroup
      cudaCallBlockCount(
        65535, //device limit, not used
        1024, //device limit, not used
        base_ptr->bitvector_size, // word count
        bitvectors_for_intersect.size() / roundOneGroupCount, // state count
        roundOneGroupCount, // config count
        countListPtr, // pointer to bitvector pointers
        reduce, // results array
        intermediateStatesRoundPtr); // start of intermediate results

      cudaDeviceSynchronize();

      // add subgroup results to task list
      for(int i = 0; i < roundOneGroupCount; i++)
      {
        ResultRecord tempResult{reduce[i], intermediateStatesRoundPtr + i * base_ptr->bitvector_size}; // intermediate result
        resultList.push_back(tempResult);
      }
      roundOneGroupCount = te.getRoundOneGroupCount();
      subGroup++;
    }

    if(subGroup > 1)
    {
      // round two. Execute all non-zero round 1 combinations
      activeRoundTwoCount = 0;
      bitvectors_for_intersect.clear();
      for(int i = 0; i < te.getTaskCount(); i++) {
          std::vector<int> combo = te.next2();
          if(resultList[combo[0]].mCount > 0 && resultList[combo[1]].mCount > 0) {
            activeRoundTwoCount++;
            bitvectors_for_intersect.push_back((uint64_t*)resultList[combo[0]].mResultPtr);
            bitvectors_for_intersect.push_back((uint64_t*)resultList[combo[1]].mResultPtr);
          }
      }

      memcpy(countListPtr,
             bitvectors_for_intersect.data(),
             bitvectors_for_intersect.size() * sizeof(unsigned long long*));

      cudaDeviceSynchronize();
      cudaCallBlockCount(
              65535, //device limit, not used
              1024, //device limit, not used
              base_ptr->bitvector_size, // word count
              bitvectors_for_intersect.size() / activeRoundTwoCount, // state count
              activeRoundTwoCount, // config count
              countListPtr, // pointer to bitvector pointers
              reduce, // results array
              0); // no intermediate results

      cudaDeviceSynchronize();
    }

    // execute callback for all non zero results
    int resultCount = subGroup > 1 ? activeRoundTwoCount : te.getTaskCount();
    for(int i = 0; i < resultCount; i++) {
      if(reduce[i] > 0){
        F[0](reduce[i],i);
      }
    }
    return;
  } // end apply - non zero

private:

  struct node {
    int id;
    int r;
    uint64_t *bitvectors;
    node() : id(0),
    r(0),
    bitvectors(0){};
  };

  struct base {
    int n;
    int m;
    int bitvector_size;
    node *node_ptr;
  };

  base *base_ptr;
  node *node_ptr;
  int maxNodesPerTask = 5;

  unsigned long long* reduce;
  const unsigned long long** countListPtr; // copy address list into gpu memory
  unsigned long long** intermediateResultsListPtrPtr; // intermediate results list ptrs
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
  int bitvector_count = 0;
  int bitvector_size = (int) ((m + 63) / 64);
  printf("bitvector takes up %d uint64_t(s)\n", bitvector_size);

  typename GPUCounter<N>::base temp_base = {n, m, bitvector_size, 0};
  temp_base.node_ptr = new typename GPUCounter<N>::node[n];

  for (int xi = 0; xi < n; ++xi) {
    size = 0;
    std::fill_n(indices, 256, -1);

    for (int j = 0; j < m; ++j) {
      temp = *temp_it++;
      if (indices[temp] == -1) {
        indices[temp] = size++;
      }
    }

    temp_base.node_ptr[xi].r = size;
    bitvector_count += size;
  }

  int database_size = sizeof(typename GPUCounter<N>::base) + n * sizeof(typename GPUCounter<N>::node);
  int bcvs_count = bitvector_count * bitvector_size;

  p.base_ptr = (typename GPUCounter<N>::base*) new char[database_size];
  uint64_t* bvPtr;
  cudaMalloc(&bvPtr, sizeof(uint64_t) * bcvs_count); // cudaMalloc

  char* zero_ptr = (char*) p.base_ptr;
  for(int i = 0; i < database_size; i++)
  {
    zero_ptr[i] = 0;
  }

  memcpy(p.base_ptr, &temp_base, sizeof(typename GPUCounter<N>::base)); // copy in base
  memcpy((char*) p.base_ptr + sizeof(typename GPUCounter<N>::base), temp_base.node_ptr, n * sizeof(typename GPUCounter<N>::node)); // copy in node array
  p.base_ptr->node_ptr = (typename GPUCounter<N>::node *) ((char *) p.base_ptr + sizeof(typename GPUCounter<N>::base)); // set node ptr

  delete[] temp_base.node_ptr;

  printf("base and node footprint is %d bytes\n", database_size);
  printf("database footprint is %lu bytes\n", sizeof(uint64_t) * bcvs_count);

  int offset = 0;

  for (int xi = 0; xi < n; ++xi) {
    p.base_ptr->node_ptr[xi].bitvectors = bvPtr + offset;
    offset += p.base_ptr->node_ptr[xi].r * bitvector_size;
//    printf("x%d r=%d bvs=%p\n", xi, p.base_ptr->node_ptr[xi].r, (void*) p.base_ptr->node_ptr[xi].bitvectors);
  }

  temp_it = it;
  for (int xi = 0; xi < n; ++xi) {

    // allocate a tempory buffer in host memory
    uint64_t* tempBvsPtr = new uint64_t[p.base_ptr->node_ptr[xi].r * bitvector_size];

    // zero it out
    memset(tempBvsPtr, 0, p.base_ptr->node_ptr[xi].r * bitvector_size * sizeof(uint64_t));

    // calculate the xi bit vectors for each ri
    for (int j = 0; j < m; ++j) {
      temp = *temp_it++;
      int bv_index = temp;
      uint64_t& bitvector = tempBvsPtr[bv_index * bitvector_size + (j>>6)]; // find the block
      bitvector |= (1L<< (j & 63)); // set the bit
    }

    // copy the xi bitvectors into GPU memory
    int bvsSize = p.base_ptr->node_ptr[xi].r * bitvector_size * 8;
    cudaMemcpy(p.base_ptr->node_ptr[xi].bitvectors, tempBvsPtr, bvsSize, cudaMemcpyHostToDevice); // cudaCopyHostToDevice
//    cudaMemcpy(p.base_ptr->node_ptr[xi].bitvectors, tempBvsPtr, bvsSize, cudaMemcpyHostToDevice); // cudaCopyHostToDevice

  //  cudaDeviceSynchronize();
    // delete the tempory host buffer
    delete[] tempBvsPtr;
  }

  uint64_t* bvCheck = new uint64_t[bcvs_count];
  std::fill_n(bvCheck, bcvs_count, 0);
//  printf("verifying %p %d bytes in device memory\n", p.base_ptr->node_ptr[0].bitvectors, bcvs_count * 8);
  cudaMemcpy(bvCheck, p.base_ptr->node_ptr[0].bitvectors, bcvs_count * 8, cudaMemcpyDeviceToHost);

  // for(int i = 0; i < bcvs_count; i++)
  // {
  //    printf("%8lx\n", bvCheck[i]);
  // }

  // unsigned char * bvCheckBytes = (unsigned char*) bvCheck;
  // for(int i = 0; i < bcvs_count * 8; i++)
  // {
  //   if(i%8==0){
  //     printf("\n%d\n", i);
  //   }
  //    printf("%x ", bvCheckBytes[i] & 0xff);
  // }

//   uint64_t sanity_check = 0;
//   for(int ri = 0; ri < p.base_ptr->node_ptr[xi].r; ri++) {
// //      printf("%d %d %llx\n", xi, xi, tempBvsPtr[ri * bitvector_size]);
//     uint64_t tempBv = tempBvsPtr[ri * bitvector_size];
//     sanity_check |= tempBv;
//   }
//
//   if(m < 64 && sanity_check != (1LL << m) -1) {
//     printf("something's wrong! %lu != %llu\n", sanity_check, (1LL << m) - 1);
//   }

  cudaMallocManaged(&p.reduce, p.base_ptr->bitvector_size * sizeof(unsigned long long) * MAX_COUNTS_PER_QUERY);
  cudaMallocManaged(&p.countListPtr, sizeof(unsigned long long*) * 1024);
  cudaMalloc(&p.intermediateResultsPtr, p.base_ptr->bitvector_size * sizeof(unsigned long long) * MAX_INTERMEDIATE_RESULTS);

  return p;
} // create_GPUCounter

#endif // GPU_COUNTER_HPP
