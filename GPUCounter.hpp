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
        uint64_t* bv_ptr =  &this->base_ptr->node_ptr[id].bitvectors[state * base_ptr->bitvector_size];
        bitvectors_for_intersect.push_back(bv_ptr);
        //printf("%d %d %p %lu\n", id, state, bv_ptr, *bv_ptr);
      }

      auto pa_vect = as_vector(pa);
      for(int i = 0; i < pa_vect.size(); i++) {
        int id = pa_vect[i];
        int state = state_pa[i];
        uint64_t* bv_ptr =  &this->base_ptr->node_ptr[id].bitvectors[state * base_ptr->bitvector_size];
        bitvectors_for_intersect.push_back(bv_ptr);
        //printf("%d %d %p %lu\n", id, state, bv_ptr, *bv_ptr);
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
      	countListPtr, // gridPtr
        reduce, // result
      	0);

       cudaDeviceSynchronize();

        F[0](reduce[0],0);
    }

    template<typename score_functor>
    void apply(const set_type &xi, const set_type &pa, std::vector<score_functor> &F) const {

      std::vector<uint64_t*> bitvectors_for_intersect;
      auto xa_vect = as_vector(xi);
      auto pa_vect = as_vector(pa);

      for(int i = 0; i < pa_vect.size(); i++)
      {
        xa_vect.push_back(pa_vect[i]);
      }

      int configCount = 1;

      for(int i = 0; i < xa_vect.size(); i++)
      {
         int id = xa_vect[i];
         int states = this->base_ptr->node_ptr[id].r;
         configCount *= states;
  //       printf("%d %d\n", id, states);
      }

      std::vector<int> bases;
      bases.push_back(1);
      for(int b = 1; b < xa_vect.size(); b++)
      {
         bases[b] = bases[b-1] * this->base_ptr->node_ptr[b-1].r;
  //       printf("base %d\n", bases[b]);
      }

  //    printf("total configs %d\n",configCount);

      for(int configIndex = 0; configIndex < configCount; configIndex++)
      {
         for(int stateIndex = 0; stateIndex < xa_vect.size(); stateIndex++)
         {
              int state = configIndex / bases[stateIndex] % this->base_ptr->node_ptr[stateIndex].r;
              uint64_t* bv_ptr =  &this->base_ptr->node_ptr[stateIndex].bitvectors[state * base_ptr->bitvector_size];
              bitvectors_for_intersect.push_back(bv_ptr);
  //            printf("%d ", state);
         }
  //       printf("\n");
      }

      if(bitvectors_for_intersect.size() * base_ptr->bitvector_size > 2048)
      {
        printf("Query block is too big for this implementation! %d %d %d\n",
        bitvectors_for_intersect.size() * base_ptr->bitvector_size,
        bitvectors_for_intersect.size(),
        base_ptr->bitvector_size);
        return;
      }
      // todo generate block of queires
      memcpy(countListPtr, bitvectors_for_intersect.data(), bitvectors_for_intersect.size() * sizeof(unsigned long long*));

     cudaDeviceSynchronize();

     cudaCallBlockCount(
      	65535, //todo
      	1024, //threads per block
      	base_ptr->bitvector_size, // word count
        bitvectors_for_intersect.size() / configCount, // state count
        configCount, // config count
      	countListPtr, // gridPtr
        reduce, // result
      	0);

       cudaDeviceSynchronize();

       for(int c = 0; c < configCount; c++){
         F[0](reduce[c],c);
       }
    }

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

    unsigned long long* reduce;
    const unsigned long long** countListPtr; // copy address list into gpu memory

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
  int bitvector_size = m / (sizeof(uint64_t) * 8) + 1;
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

  int database_size = sizeof(typename GPUCounter<N>::base) + n * sizeof(typename GPUCounter<N>::node) + bitvector_count * bitvector_size * sizeof(uint64_t);

  cudaMallocManaged(&p.base_ptr, database_size);

  //p.base_ptr = (typename GPUCounter<N>::base *) new char[database_size]; // gpu malloc

  char* zero_ptr = (char*) p.base_ptr;
  for(int i = 0; i < database_size; i++) {
    zero_ptr[i] = 0;
  }

  memcpy(p.base_ptr, &temp_base, sizeof(typename GPUCounter<N>::base)); // copy in base
  memcpy((char*) p.base_ptr + sizeof(typename GPUCounter<N>::base), temp_base.node_ptr, n * sizeof(typename GPUCounter<N>::node)); // copy in node array
  p.base_ptr->node_ptr = (typename GPUCounter<N>::node *) ((char *) p.base_ptr + sizeof(typename GPUCounter<N>::base)); // set node ptr

  delete temp_base.node_ptr;
  //printf("sizes:\n\tbase=%lu\n\tnode=%lu\n\tuint64=%lu\n", sizeof(typename GPUCounter<N>::base), sizeof(typename GPUCounter<N>::node), sizeof(uint64_t));
  //printf("database for %d by %d has %d bv and %d byte footprint\n", p.base_ptr->n, p.base_ptr->m, bitvector_count, database_size);
  printf("database footprint is %d bytes\n", database_size);
  int offset = 0;
  uint64_t * starting_bitvector = (uint64_t*)((char*) p.base_ptr->node_ptr + n * sizeof(typename GPUCounter<N>::node));
  for (int xi = 0; xi < n; ++xi) {
    p.base_ptr->node_ptr[xi].bitvectors = starting_bitvector + offset;
    offset += p.base_ptr->node_ptr[xi].r * bitvector_size;
    //printf("x%d r=%d bvs=%p\n", xi, p.base_ptr->node_ptr[xi].r, (void*) p.base_ptr->node_ptr[xi].bitvectors);
  }

  temp_it = it;
  for (int xi = 0; xi < n; ++xi) {
    for (int j = 0; j < m; ++j) {
      temp = *temp_it++;
      int bv_index = temp;
      uint64_t& bitvector = p.base_ptr->node_ptr[xi].bitvectors[bv_index * bitvector_size + (j>>6)]; // find the block
      bitvector |= (1L<< (j & 63)); // set the bit
    }

    uint64_t sanity_check = 0;
    for(int ri = 0; ri < p.base_ptr->node_ptr[xi].r; ri++) {
//      printf("%d %d %lu\n", xi, xi,p.base_ptr->node_ptr[xi].bitvectors[ri * bitvector_size]);
      sanity_check |= p.base_ptr->node_ptr[xi].bitvectors[ri * bitvector_size];
    }
    if(m < 64 && sanity_check != (1LL << m) -1) {
      printf("something's wrong! %lu != %llu\n", sanity_check, (1LL << m) - 1);
    }
  }

  cudaMallocManaged(&p.reduce, p.base_ptr->bitvector_size * sizeof(unsigned long long));
  cudaMallocManaged(&p.countListPtr, sizeof(unsigned long long*) * 2048); // 2048 bit vectors per query block

  return p;
} // create_GPUCounter

#endif // GPU_COUNTER_HPP
