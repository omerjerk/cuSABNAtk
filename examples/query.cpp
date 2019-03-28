#include <iostream>
#include <tuple>
#include <vector>

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

int main(int argc, char* argv[]) {
    jaz::Logger Log;

    int N = 2;

    if (argc != 3) {
        std::cout << "usage: " << argv[0] << " data_file query_file" << std::endl;
        return 0;
    }

    std::string csv_name = argv[1];

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

    GPUCounter<2> gcount = create_GPUCounter<2>(n, m, std::begin(D));

    using set_type = typename GPUCounter<2>::set_type;
    auto xi = set_empty<set_type>();
    auto pa = set_empty<set_type>();
    xi = set_add(xi, 98);

    std::vector<int> paVec = {9, 71, 40, 39, 43, 82, 85, 20, 66, 52};
    for (int i = 0; i < paVec.size(); ++i) {
        pa = set_add(pa, paVec[i]);
    }

    std::vector<Call> F(1);

    gcount.apply(xi, pa, F);

    return 0;
} // main
