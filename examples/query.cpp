/***
 *  $Id$
 **
 *  File: query.cpp
 *  Created: Jun 29, 2019
 *
 *  Authors: Mohammad Umair <m39@buffalo.edu>
 *           Jaroslaw Zola <jaroslaw.zola@hush.com>
 *  Copyright (c) 2019 SCoRe Group http://www.score-group.org/
 *  Distributed under the MIT License.
 *  See accompanying file LICENSE.
 */

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include <stdlib.h>
#include <sched.h>

#include <jaz/logger.hpp>

#include <GPUCounter.hpp>
#include <RadCounter.hpp>

#include "csv.hpp"

#ifdef _OPENMP
#include <omp.h>
#else
inline int omp_get_num_threads() { return 1; }
inline int omp_get_thread_num() { return 0; }
#endif



struct Call {
    void init(int, int) { score_ = 0; }

    void finalize(int) { }

    void operator()(int) { }

    void operator()(int Nijk, int Nij) {
        double p = static_cast<double>(Nijk) / Nij;
        score_ += (Nijk * std::log2(p));
    } // operator()

    double score() const { return score_; }

    double score_ = 0.0;
}; // struct Call


std::tuple<bool, std::vector<int>, std::vector<int>> read_query(const std::string& query_file) {
    std::ifstream f(query_file);
    std::vector<int> pa, xi;

    if (!f) return std::make_tuple(false, pa, xi);

    int paCount, xiCount, temp;

    f >> paCount;

    for (int i = 0; i < paCount; ++i) {
        f >> temp;
        pa.push_back(temp);
    }

    f >> xiCount;

    for (int i = 0; i < xiCount; ++i) {
        f >> temp;
        xi.push_back(temp);
    }

    return std::make_tuple(true, pa, xi);
} // read_query

std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<int>>> get_benchmark_queries(int n, int nt, int nq, int seed = 0) {
    std::vector<int> var(n);
    std::iota(std::begin(var), std::end(var), 0);

    if (seed == 0) {
        std::random_device rd;
        seed = rd();
    }

    std::mt19937 rng(seed);

    std::vector<std::vector<int>> pas(nt, std::vector<int>(nq - 1));
    std::vector<std::vector<int>> xis(nt, std::vector<int>(1));

    for (int i = 0; i < nt; ++i) {
        std::shuffle(std::begin(var), std::end(var), rng);
        xis[i][0] = var[0];
        for (int j = 1; j < nq; ++j) pas[i][j - 1] = var[j];
    }

    return std::make_tuple(xis, pas);
} // get_benchmark_queries


template <typename Engine>
double test_queries(Engine& qe, int nt,
                    const std::vector<std::vector<int>>& xis,
                    const std::vector<std::vector<int>>& pas,
                    jaz::Logger& Log,
                    int numThreads) {

    std::vector<Call> F(1);

    auto t0 = std::chrono::system_clock::now();

    // #pragma omp parallel
    // {
        // #pragma omp single
        // Log.info() << "running with " << omp_get_num_threads() << " threads" << std::endl;

        if (numThreads == -1) {
            #pragma omp parallel for firstprivate(F) shared(qe, xis, pas)
            for (int i = 0; i < nt; ++i) qe.apply(xis[i], pas[i], F);
        } else {
            #pragma omp parallel for firstprivate(F) shared(qe, xis, pas)
            for (int i = 0; i < nt; ++i) qe.apply(xis[i], pas[i], F);
        }

    // }

    auto t1 = std::chrono::system_clock::now();

    return std::chrono::duration<double>(t1 - t0).count();
} // test_queries


int main(int argc, char* argv[]) {
    using set_type = typename GPUCounter<3>::set_type;
    jaz::Logger Log;

    if ((argc != 3) && (argc != 5)) {
        std::cout << "standard mode: " << argv[0] << " <data_file> <query_file>" << std::endl;
        std::cout << "benchmark mode: " << argv[0] << " <data_file> benchmark <# iterations> <# variables>" << std::endl;
        return 0;
    }

    std::string csv_name = argv[1];
    std::string query_file = argv[2];

    using data_type = uint8_t;
    std::vector<data_type> D;

    bool b = false;
    int n = -1;
    int m = -1;

    Log.info() << "reading input data" << std::endl;

    std::tie(b, n, m) = read_csv(csv_name, D);

    if (!b) {
        Log.error() << "could not read input data" << std::endl;
        return -1;
    }

    Log.info() << "creating counters" << std::endl;

    GPUCounter<3> gcount = create_GPUCounter<3>(n, m, std::begin(D));
    RadCounter<3> rad = create_RadCounter<3>(n, m, std::begin(D));
    gcount.initRadCounter(&rad);

    if (query_file == "benchmark") {
        // perform the benchmark
        int nt = std::stoi(argv[3]); // number of iterations
        int nq = std::stoi(argv[4]); // number of variables in the query

        if (n < nq) {
            Log.error() << "Number of query variables cannot be greater than the total number of variables" << std::endl;
            return -1;
        }

        std::vector<std::vector<int>> xis;
        std::vector<std::vector<int>> pas;

        std::tie(xis, pas) = get_benchmark_queries(n, nt, nq);

        Log.info() << "testing GPU..." << std::endl;
        auto gput = test_queries(gcount, nt, xis, pas, Log, 2);
        Log.info() << "time for " << nt << " queries with " << nq << " variables: " <<  jaz::log::second_to_time(gput) << std::endl;

        Log.info() << "testing Rad..." << std::endl;
        auto radt = test_queries(rad, nt, xis, pas, Log, -1);
        Log.info() << "time for " << nt << " queries with " << nq << " variables: " <<  jaz::log::second_to_time(radt) << std::endl;

        Log.info() << "GPU speedup: " << (radt / gput) << std::endl;
    } else {
        // execute the query
        std::vector<int> paVec;
        std::vector<int> xiVec;

        std::tie(b, paVec, xiVec) = read_query(query_file);

        std::vector<Call> F(1);

        gcount.apply(xiVec, paVec, F);
        Log.info() << "GPU result: " << F[0].score() << std::endl;

        rad.apply(xiVec, paVec, F);
        Log.info() << "Rad result: " << F[0].score() << std::endl;
    }

    return 0;
} // main
