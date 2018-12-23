/*

COMMAND LINE ARGUMENTS

  "-data=<path>":       the path to the database file
  "-query=<path>":      the path to the query parameters file
  "-i=<iterations>":    specify the number of iterations

*/


#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <utility>
#include "BVCounter.hpp"
#include "helper_string.h"
#include "GPUCounter.hpp"

static std::vector<char> def{0, 1, 1, 0, 1, 1, 0, 0, \
                             0, 0, 2, 0, 1, 2, 0, 1, \
                             1, 1, 1, 0, 1, 1, 1, 0};

typedef std::pair<int,int> state;

struct call {
    void init(int, int) { nijk = 0; }

    void finalize(int) {}

    void operator()(int) {}

    void operator()(int Nijk, int i) {
      nijk = Nijk;
    } // operator()

    int score() const { return nijk; }

    int nijk = 0;
}; // struct call

struct call_v {
    void init(int, int) { nijk = 0; }

    void finalize(int) {}

    void operator()(int) {}

    void operator()(int Nijk, int i) {
      std::cout << "call from bvc: " << i << " " << Nijk << std::endl;
      nijk = Nijk;
    } // operator()

    int score() const { return nijk; }

    int nijk = 0;
}; // struct call

int loadDatabase(const char* pFile, std::vector<char> &pDatabase);
bool loadQuery(const char* pFile, std::vector<state>& pStates);
std::vector<std::string> split(char *phrase, std::string delimiter);

template <int N, typename score_functor>
bool runTest(std::vector<char>& D, std::vector<state> states, int n, bool stateQuery, int iterations, std::vector<score_functor>& F);

int main(int argc, char *argv[]) {
  std::vector<char> D;
  std::vector<state> states;
  int n = 0;
  int iterations = 1;
  bool stateQuery = false;
  char *filePath = 0;
  char *queryPath = 0;
  bool verbose = false;
  getCmdLineArgumentString(argc, (const char **) argv, "data", &filePath);

  printf("loading data: %s...\n", filePath);

  if (0 != filePath) {
    n = loadDatabase(filePath, D);
  } else {
    D = def;
    n = 3;
  }

  getCmdLineArgumentString(argc, (const char **) argv, "query", &queryPath);

  printf("loading query: %s...\n", queryPath);
  if (0 != queryPath) {
    stateQuery = loadQuery(queryPath, states);
  } else {
    // default query
    states.push_back(state{0,0});
    states.push_back(state{1,0});
    states.push_back(state{2,0});
    stateQuery = false;
  }

  if (checkCmdLineFlag(argc, (const char **) argv, "i")) {
    iterations = getCmdLineArgumentInt(argc, (const char **) argv, "i");
  }

  if (checkCmdLineFlag(argc, (const char **) argv, "v")) {
    verbose = true;
  }

  int m = D.size() / n;

  int words = ceil(n / 64.0);
  printf("database: n=%d, m=%d, words=%d\n", n, m, words);
  printf("running %s query on %lu nodes %d times\n",
  stateQuery ? "state specific" : "non-zero search",
  states.size(),
  iterations);
  std::vector<call> C(1);
  std::vector<call_v> CV(1);

  if(!verbose)
  {
    // running for timing. Run verbose once to verify correctness
    switch (words) {
      case 1:
        runTest<1>(D, states, n, stateQuery, 1, CV);
        runTest<1>(D, states, n, stateQuery, iterations, C);
        break;
      case 2:
        runTest<2>(D, states, n, stateQuery, 1, CV);
        runTest<2>(D, states, n, stateQuery, iterations, C);
        break;
      case 3:
        runTest<3>(D, states, n, stateQuery, 1, CV);
        runTest<3>(D, states, n, stateQuery, iterations, C);
        break;
      case 4:
        runTest<4>(D, states, n, stateQuery, 1, CV);
        runTest<4>(D, states, n, stateQuery, iterations, C);
        break;
      // ... todo add more to accomadate larger data sets ...
      default:
        printf(" %d variables not suppoted!\n", n);
    }
  }
  else
  {
    printf("running verbose, prints will invalidate timing data...\n");
    switch (words) {
      case 1:
        runTest<1, call_v>(D, states, n, stateQuery, iterations, CV);
        break;
      case 2:
        runTest<2, call_v>(D, states, n, stateQuery, iterations, CV);
        break;
      case 3:
        runTest<3, call_v>(D, states, n, stateQuery, iterations, CV);
        break;
      case 4:
        runTest<4, call_v>(D, states, n, stateQuery, iterations, CV);
        break;
      // ... todo add more to accomadate larger data sets ...
      default:
        printf(" %d variables not suppoted!\n", n);
    }
  }
  return 0;
}

template <int N, typename score_functor>
bool runTest(std::vector<char>& D,
  std::vector<state> states,
  int n,
  bool stateQuery,
  int iterations,
  std::vector<score_functor> &F)
{
  int m = D.size() / n;
  BVCounter<N> bvc = create_BVCounter<N>(n, m, std::begin(D));
  GPUCounter<N> gpuc = create_GPUCounter<N>(n, m, std::begin(D));

  using set_type = typename BVCounter<N>::set_type;

  auto xi = set_empty<set_type>();
  auto pa = set_empty<set_type>();

  // first node
  xi = set_add(xi, states[0].first);
  std::vector<char> sxi{ (char)states[0].second};

  // then parents
  std::vector<char> spa;
  spa.reserve(n);
  for (int i = 1; i < states.size(); i++) {
    pa = set_add(pa, states[i].first);
    spa.push_back((char)states[i].second);
  }

  // callback for each xi
  std::vector<std::chrono::duration<double>> times;
  std::vector<std::chrono::duration<double>> gpuTimes;

  printf("cpu...\n");
  // cpu runs...
  for(int i = 0; i < iterations; i++) {
    auto t1 = std::chrono::system_clock::now();
    if(stateQuery) {
      bvc.apply(xi, pa, sxi, spa, F);
    }
    else{
      bvc.apply(xi, pa, F);
    }
    auto t2 = std::chrono::system_clock::now();
    auto elapsed_cpu = std::chrono::duration<double>(t2 - t1);
    times.push_back(elapsed_cpu);
  }

  for(int index = 0; index < F.size(); index++){
    printf("cpu score %d\n", F[index].score());
  }

  printf("gpu...\n");
  // gpu runs...
  for(int i = 0; i < iterations; i++) {
    auto t1 = std::chrono::system_clock::now();
    if(stateQuery) {
      gpuc.apply(xi, pa, sxi, spa, F);
    }
    else{
      gpuc.apply(xi, pa, F);
    }
    auto t2 = std::chrono::system_clock::now();
    auto elapsed_cpu = std::chrono::duration<double>(t2 - t1);
    gpuTimes.push_back(elapsed_cpu);
  }

  double total = 0.0;
  double max = 0.0;

  for(int i = 0; i < iterations; i++)
  {
    if(times[i].count() > max){
      max = times[i].count();
    }

    total += times[i].count();
  }

  double average = total / iterations;

  printf("cpu avg=%lf max=%lf\n", average, max);

  total = 0.0;
  max = 0.0;
  for(int i = 0; i < iterations; i++)
  {
    if(gpuTimes[i].count() > max){
      max = gpuTimes[i].count();
    }

    total += gpuTimes[i].count();
  }

  average = total / iterations;

  for(int index = 0; index < F.size(); index++){
    printf("gpu score %d\n", F[index].score());
  }

  printf("gpu avg=%lf max=%lf\n", average, max);
  return true;
}

static const int MAX_DATA_PER_LINE = 2048 * 64;

int loadDatabase(const char* pFile, std::vector<char> &pDatabase) {
  int variableCount = 0;
  bool done = false;
  char *inputArray = 0;

  std::ifstream inFile(pFile);

  if (!inFile.is_open()) {
    printf("could not open %s\n", pFile);
    return false;
  }

  inputArray = new char[MAX_DATA_PER_LINE];
  pDatabase.clear();

  while (!done) {
    printf(".");
    inFile.getline(inputArray, MAX_DATA_PER_LINE);
    int len = strlen(inputArray);

    std::vector<std::string> tokens;
    tokens = split(inputArray, " ");

    for (int token = 0; token < tokens.size(); token++) {
      std::string temp = tokens[token];
      pDatabase.push_back(atoi(temp.c_str()));
    }

    if (len > 0) {
      variableCount++;
    }
    done = inFile.eof();
  }

  if (pDatabase.size() % variableCount != 0) {
    printf("WARNING dimension mismatch total: %lu\t variable %d\n",
           pDatabase.size(),
           variableCount);
  }

  delete[] inputArray;
  return variableCount;
}

bool loadQuery(const char* pFile, std::vector<state>& pStates){
  bool done = false;
  bool stateQuery = false;
  char *inputArray = 0;

  std::ifstream inFile(pFile);

  if (!inFile.is_open()) {
    printf("could not open %s\n", pFile);
    return false;
  }

  inputArray = new char[MAX_DATA_PER_LINE];
  pStates.clear();

  while (!done) {
    inFile.getline(inputArray, MAX_DATA_PER_LINE);
    int len = strlen(inputArray);

    if(len > 0) {
      std::vector<std::string> tokens;
      tokens = split(inputArray, " ");
      state temp;
      temp.first = atoi(tokens[0].c_str());

      if (tokens.size() > 1) {
        temp.second = atoi(tokens[1].c_str());
        stateQuery = true;
      }

      pStates.push_back(temp);
    }

    done = inFile.eof();
  }

  delete[] inputArray;
  return stateQuery;
}

std::vector<std::string> split(char *phrase, std::string delimiter){
  std::vector<std::string> list;
  std::string s = std::string(phrase);
  size_t pos = 0;
  std::string token;
  while ((pos = s.find(delimiter)) != std::string::npos) {
    token = s.substr(0, pos);
    list.push_back(token);
    s.erase(0, pos + delimiter.length());
  }

  if(s.size() > 0) {
    list.push_back(s);
  }

  return list;
}
