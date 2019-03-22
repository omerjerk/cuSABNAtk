#include <iostream>
#include <tuple>
#include <vector>

#include <jaz/logger.hpp>

#include "GPUCounter.hpp"
#include "csv.hpp"


int main(int argc, char* argv[]) {
    jaz::Logger Log;

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

    return 0;
} // main
