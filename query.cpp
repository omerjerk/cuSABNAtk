#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>

#include "BVCounter.hpp"
#include "GPUCounter.hpp"

struct call {
    void init(int, int) { nijk = 0; }

    void finalize(int) { }

    void operator()(int) { }

    void operator()(int Nijk, int i) {
        std::cout << "call from bvc: " << i << " " << Nijk << std::endl;
        nijk = Nijk;
    } // operator()

    int score() const { return nijk; }

    int nijk = 0;
}; // struct call


int main(int argc, char* argv[]) {
    // 3 variables (say X0, X1, X2), 8 observations
    std::vector<char> D{0, 1, 1, 0, 1, 1, 0, 0, \
                        0, 0, 2, 0, 1, 2, 0, 1, \
                        1, 1, 1, 0, 1, 1, 1, 0 };
    int n = 3;
    int m = 8;
    int c = 3;
    if(argc > 1) {
      printf("data file: %s\n", argv[1]);
      D.clear();
      n = 0;
      std::ifstream inFile;
      inFile.open(argv[1]);
      bool open = inFile.is_open();
      printf("file open: %s\n", open ? "sucess" : "failed");
      char *inputArray = new char[2048 * 10];

      bool done = !open;
      while (!done) {

        inFile.getline(inputArray, 2048 * 10);
        int len = strlen(inputArray);

        for (int charIndex = 0; charIndex < len; charIndex += 2) {
          D.push_back(atoi(&inputArray[charIndex]));
        }
        if (n == 0) {
          m = D.size();
        }
        n++;
        done = inFile.eof();
      }
    }

    if(argc > 2)
    {
      c = atoi(argv[2]);
    }

    printf("n=%d m=%d c=%d\n", n, m, c);

    // use one word (64bit) because n < 64
    BVCounter<1> bvc = create_BVCounter<1>(n, m, std::begin(D));

    GPUCounter<1> gpuc = create_GPUCounter<1>(n, m, std::begin(D));

    using set_type = BVCounter<1>::set_type;

    auto xi = set_empty<set_type>();
    auto pa = set_empty<set_type>();

    // let's count X0=0 and [X1=0,X2=0...Xn-1=0]:

    // first node
    xi = set_add(xi, 0);
    std::vector<char> sxi{0};

    // then parents
    std::vector<char> spa;
    spa.reserve(n);
    for(int i = 1; i < c; i++)
    {
      pa = set_add(pa, i);
      spa.push_back(1);
    }

    // callback for each xi
    std::vector<call> CCpu(1);
    std::vector<call> CGpu1(1);
    std::vector<call> CGpu2(1);

    // and here we go
    auto t1 = std::chrono::system_clock::now();
    bvc.apply(xi, pa, CCpu);
    auto t2 = std::chrono::system_clock::now();
    auto elapsed_cpu = std::chrono::duration<double>(t2 - t1);

    auto t3 = std::chrono::system_clock::now();
    gpuc.apply(xi, pa, CGpu1); // first time in penalty?
    auto t4 = std::chrono::system_clock::now();
    gpuc.apply(xi, pa, CGpu2);
    auto t5 = std::chrono::system_clock::now();

    auto elapsed_gpu1 = std::chrono::duration<double>(t4 - t3);
    auto elapsed_gpu2 = std::chrono::duration<double>(t5 - t4);

    printf("cpu=%lf\ngpu1=%lf\ngpu2=%lf\n", elapsed_cpu.count(), elapsed_gpu1.count(),elapsed_gpu2.count());
    printf("cpu=%d\ngpu1=%d\ngpu2=%d\n", CCpu[0].score(), CGpu1[0].score(), CGpu2[0].score());

    return 0;
} // main
