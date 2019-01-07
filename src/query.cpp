/*

COMMAND LINE ARGUMENTS

  "-data=<path>":       the path to the database file
  "-query=<path>":      the path to the query parameters file
  "-i=<iterations>":    specify the number of iterations

*/

#include <stdio.h>
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
bool loadQueries(const char* pFile, std::vector<std::vector<state>>& pQueries);
std::vector<state> loadQueryLine(char* pInputArray);
void storeResults(const char* pFile, std::vector<std::vector<state>>& pQueries, std::vector<double> pTimes);
std::string toString(std::vector<state>& pQuery);
std::vector<std::string> split(char *phrase, std::string delimiter);

template <int N>
std::vector<double> runTests(std::vector<char>& D, std::vector< std::vector<state>> queries, int n, int iterations);

int main(int argc, char *argv[]) {
  std::vector<char> D;
  std::vector <std::vector<state> > queries;
  std::vector<state> states;
  std::vector<double> times;
  int n = 0;

  int iterations = 1;
  bool stateQuery = false;
  char *filePath = 0;
  char *queryPath = 0;
  bool verbose = false;
  getCmdLineArgumentString(argc, (const char **) argv, "data", &filePath);


  if (0 != filePath) {
    n = loadDatabase(filePath, D);
  } else {
    D = def;
    n = 3;
  }

  getCmdLineArgumentString(argc, (const char **) argv, "query", &queryPath);

  printf("loading query: %s...\n", queryPath);
  if (0 != queryPath) {
    loadQueries(queryPath, queries);
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

  std::vector<call> C(1);
  std::vector<call_v> CV(1);

  switch (words) {
    case 1:
      times = runTests<1>(D, queries, n, iterations);
      break;
    case 2:
      times = runTests<2>(D, queries, n, iterations);
      break;
    case 3:
      times = runTests<3>(D, queries, n, iterations);
      break;
    default:
      printf(" %d variables not supported!\n", n*64);
  }

  // store out timing results to <query>.dat
  std::vector<std::string> tokens = split(queryPath, ".");
  char buffer[2048];
  bzero(buffer, 2048);
  sprintf(buffer, "%s.dat", tokens[0].c_str());
  std::string resultsPath(buffer);
  storeResults(resultsPath.c_str(), queries, times);

  return 0;
}

template <int N>
std::vector<double> runTests(std::vector<char>& D, std::vector< std::vector<state> > queries, int n, int iterations)
{
  int m = D.size() / n;

  // BVCounter<N> bvc = create_BVCounter<N>(n, m, std::begin(D));
  printf("gpu counter test...\n");
  GPUCounter<N> bvc = create_GPUCounter<N>(n, m, std::begin(D));

  using set_type = typename BVCounter<N>::set_type;

  // setup timekeeping
  std::vector<std::chrono::duration<double>> times;
  int queryIndex = 0;
  int totalTimes = iterations * queries.size();
  times.reserve(totalTimes);
  for(int timesIndex = 0; timesIndex < totalTimes; timesIndex++) {
    times.push_back(std::chrono::duration<double>(0));
  }

  int queryCount = queries.size();
  for(int testIndex = 0; testIndex < iterations; testIndex++) {
    for (std::vector<state> &states : queries) {
      auto xi = set_empty<set_type>();
      auto pa = set_empty<set_type>();

      // first node ... end of list
      xi = set_add(xi, states[states.size() - 1].first);

      // then parents
      for (int i = 1; i < states.size(); i++) {
        pa = set_add(pa, states[i].first);
      }

      // callback for each xi
      std::vector<call> CCpu(1);
      // runs...
      auto t1 = std::chrono::system_clock::now();
      bvc.apply(xi, pa, CCpu);
      auto t2 = std::chrono::system_clock::now();
      auto elapsed_cpu = std::chrono::duration<double>(t2 - t1);

      int timeIndex = (queryIndex % queryCount) * iterations + floor(queryIndex / queryCount);
      times[timeIndex] = elapsed_cpu;
      queryIndex++;
    }
  }

  // compute average times
  std::vector<double> averagedTimes;
  double runningSum = 0.0;

  for(int index = 0; index < totalTimes; index++) {
    runningSum += times[index].count();

    if(index % iterations == (iterations-1)) {
      averagedTimes.push_back(runningSum / iterations);
      runningSum = 0.0;
    }
  }

  return averagedTimes;
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

bool loadQueries(const char* pFile, std::vector<std::vector<state> >& pQueries){
  bool done = false;
  bool stateQuery = false;
  char *inputArray = 0;

  std::ifstream inFile(pFile);

  if (!inFile.is_open()) {
    printf("could not open %s\n", pFile);
    return false;
  }

  inputArray = new char[MAX_DATA_PER_LINE];
  pQueries.clear();

  int count = 0;
  while (!done) {

    // two lines per query... 1st line Pa, Second Line Xi
    inFile.getline(inputArray, MAX_DATA_PER_LINE);
    std::vector<state> pa = loadQueryLine(inputArray);
    inFile.getline(inputArray, MAX_DATA_PER_LINE);
    std::vector<state> xi = loadQueryLine(inputArray);
    count++;

    // merge and add
    if(pa.size() > 0 || xi.size() > 0) {
      pa.insert(pa.end(), xi.begin(), xi.end());
      pQueries.push_back(pa);
    }
    done = inFile.eof();
  }

  delete[] inputArray;
  return stateQuery;
}

std::vector<state> loadQueryLine(char*  pInputArray)
{
  std::vector<state> stateLine;
  int len = strlen(pInputArray);

  if(len > 0) {
    std::vector<std::string> tokens;
    tokens = split(pInputArray, " ");

    int nodeCount = atoi(tokens[0].c_str());

    for (int nodeIndex = 1; nodeIndex <= nodeCount; nodeIndex++) {
      state temp;
      temp.first = atoi(tokens[nodeIndex].c_str());
      stateLine.push_back(temp);
    }
  }
  return stateLine;
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

void storeResults(const char* pFile, std::vector<std::vector<state>>& pQueries, std::vector<double> pTimes)
{
  std::ofstream outFile(pFile);

  if (!outFile.is_open()) {
    printf("could not open %s\n", pFile);
  }

  for(int queryIndex = 0; queryIndex < pTimes.size(); queryIndex++) {
    char buffer[2048];
    bzero(&buffer, 2048);
    double timeInUs = pTimes[queryIndex] * 1000000.0;
    std::string queryAsString = toString(pQueries[queryIndex]);
    sprintf(buffer, "%.6lfus| %s\n", timeInUs, queryAsString.c_str());
    outFile.write(buffer, strlen(buffer));
  }
  outFile.close();
}

std::string toString(std::vector<state>& pQuery)
{
  std::string queryAsString;

  for(int stateIndex = 0; stateIndex < pQuery.size() - 1; stateIndex++){
    queryAsString += std::to_string(pQuery[stateIndex].first) + " ";
  }

  queryAsString += "|" + std::to_string(pQuery[pQuery.size() - 1].first) + " ";

  return queryAsString;
}
