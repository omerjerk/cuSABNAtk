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
        infile>>temp;
        xi.push_back(temp);
    }
    return std::make_tuple(true, pa, xi);
}

std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<int>>> get_benchmark_queries(int n, int nt, int nq) {
    std::vector<int> variables(n);
    for (int i = 0; i < n; ++i) {
        variables[i] = i;
    }
    int seed = 100;
    auto start = variables.begin();
    auto end = variables.end();
    std::vector<std::vector<int>> pas(nt, std::vector<int>(nq-1));
    std::vector<std::vector<int>> xis(nt, std::vector<int>(1));
    for (int i = 0; i < nt; ++i) {
        shuffle(start, end, std::mt19937(seed));
        xis[i][0] = variables[0];
        for (int j = 1; j < nq; ++j) {
            pas[i][j-1] = variables[j];
            // printf("%d ", variables[j]);
        } //printf("\n");
    }
    return std::make_tuple(xis, pas);
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
        std::vector<Call> F(1);
        std::vector<std::vector<int>> xis;
        std::vector<std::vector<int>> pas;
        std::tie(xis, pas) = get_benchmark_queries(n, nt, nq);
        auto t0 = std::chrono::system_clock::now();
        for (int i = 0; i < nt; ++i) {
            gcount.apply(xis[i], pas[i], F);
        }
        auto t1 = std::chrono::system_clock::now();
        printf("Time for %d queries with %d variables = %fs\n", nt, nq, std::chrono::duration<double>(t1-t0).count());
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
