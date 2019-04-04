#include <iostream>
#include <tuple>
#include <vector>
#include <fstream>
#include <string>

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
    std::string line;
    std::getline(infile, line);
    std::vector<std::string> tokens = split(line, " ");
    int paCount = stoi(tokens[0]);
    for (int i = 1; i <= paCount; ++i) {
        pa.push_back(stoi(tokens[i]));
    }
    
    std::getline(infile, line);
    tokens = split(line, " ");
    int xiCount = stoi(tokens[0]);
    for (int i = 1; i <= xiCount; ++i) {
        xi.push_back(stoi(tokens[i]));
    }
    return std::make_tuple(true, pa, xi);
}

int main(int argc, char* argv[]) {
    jaz::Logger Log;

    if (argc != 3) {
        std::cout << "usage: " << argv[0] << " data_file query_file" << std::endl;
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

    if (b == false) {
        Log.error() << "could not read input data" << std::endl;
        return -1;
    }

    std::vector<int> paVec;
    std::vector<int> xiVec;
    std::tie(b, paVec, xiVec) = read_query(query_file);

    GPUCounter<2> gcount = create_GPUCounter<2>(n, m, std::begin(D));

    using set_type = typename GPUCounter<2>::set_type;
    auto xi = set_empty<set_type>();
    auto pa = set_empty<set_type>();

    for (int i = 0; i < paVec.size(); ++i) {
        pa = set_add(pa, paVec[i]);
    }

    for (int i = 0; i < xiVec.size(); ++i) {
        xi = set_add(xi, xiVec[i]);
    }

    std::vector<Call> F(1);

    gcount.apply(xi, pa, F);

    return 0;
} // main
