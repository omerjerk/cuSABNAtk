#include <iostream>
#include <tuple>
#include <vector>
#include <fstream>
#include <string>
#include <random>
#include <algorithm>

#include <jaz/logger.hpp>

#include "GPUCounter.hpp"
#include "csv.hpp"

struct Call {
    void init(int, int) { nijk = 0; }

    void finalize(int) {}

    void operator()(int) {}

    void operator()(int Nijk, int i) {
      nijk = Nijk;
    //   printf("%d %d\n", nijk, i);
    }

    int score() const { return nijk; }

    int nijk = 0;
};

std::tuple<bool, std::vector<int>, std::vector<int>> read_query(std::string query_file) {
    std::ifstream infile(query_file);
    std::vector<int> pa, xi;
    if (!infile) {
        return std::make_tuple(false, pa, xi);
    }
    int paCount, xiCount, temp;
    infile>>paCount;
    for (int i = 1; i <= paCount; ++i) {
        infile>>temp;
        pa.push_back(temp);
    }
    infile>>xiCount;
    for (int i = 1; i <= xiCount; ++i) {
        xi.push_back(xiCount);
    }
    return std::make_tuple(true, pa, xi);
}

std::vector<std::vector<int>> get_benchmark_queries(int n, int nt, int nq) {
    std::vector<int> variables(n);
    for (int i = 0; i < n; ++i) {
        variables[i] = i;
    }
    int seed = 100;
    auto start = variables.begin();
    auto end = variables.end();
    std::vector<std::vector<int>> queries(nt, std::vector<int>(nq));
    for (int i = 0; i < nt; ++i) {
        shuffle(start, end, std::mt19937(seed));
        for (int j = 0; j < nq; ++j) {
            queries[i][j] = variables[j];
            // printf("%d ", variables[j]);
        } //printf("\n");
    }
    return queries;
}

int main(int argc, char* argv[]) {
    using set_type = typename GPUCounter<3>::set_type;
    jaz::Logger Log;

    if (argc < 3) {
        std::cout << "Normal usage: " << argv[0] << " <data_file> <query_file>" << std::endl;
        std::cout << "Benchmark usage: " << argv[0] << " <data_file> benchmark <number_of_variables_in_query>" << std::endl;
        return 0;
    }

    std::string csv_name = argv[1];
    std::string query_file = argv[2];

    using data_type = uint8_t;
    std::vector<data_type> D;

    bool b = false;
    int n = -1;
    int m = -1;

    std::tie(b, n, m) = read_csv(csv_name, D);

    // printf("n = %d m = %d\n", n, m);

    if (!b) {
        Log.error() << "could not read input data" << std::endl;
        return -1;
    }

    // printf("n = %d m = %d\n", n, m);

    GPUCounter<3> gcount = create_GPUCounter<3>(n, m, std::begin(D));

    if (query_file.compare("benchmark") == 0) {
        //perform the benchmark
        int nt = std::stoi(argv[3]);//number of iterations
        int nq = std::stoi(argv[4]);//number of variables in the query
        if (n < nq) {
            Log.error() << "Number of variables in query cannot be more than number of variables in the dataset" << std::endl;
            return -1;
        }
        double time = 0;
        std::vector<Call> F(1);
        auto queries = get_benchmark_queries(n, nt, nq);
        std::vector<int> xi(1, -1);
        std::vector<int> pa(nq-1, -1);
        for (int i = 0; i < queries.size(); ++i) {
            xi[0] = queries[i][0];
            for (int j = 1; j < nq; ++j) {
                pa[j-1] = queries[i][j];
            }
            auto t0 = std::chrono::system_clock::now();
            gcount.apply(xi, pa, F);
            auto t1 = std::chrono::system_clock::now();
            time += std::chrono::duration<double>(t1 - t0).count();
        }
        printf("Time for %d queries with %d variables = %f\n", nt, nq, time);
    } else {
        //execute the query
        std::vector<int> paVec;
        std::vector<int> xiVec;
        std::tie(b, paVec, xiVec) = read_query(query_file);

        std::vector<Call> F(1);

        gcount.apply(xiVec, paVec, F);
    }

    return 0;
} // main
