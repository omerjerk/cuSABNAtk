#!/bin/bash

if [ -n "$1" ]; then
  DIR="$1"
else
  DIR=`pwd`
fi

mkdir -p build/
rm -rf build/*
cd build/

# Add -DCMAKE_BUILD_TYPE=Debug to debug
# Change NVCC flags as needed
cmake ../ -DCMAKE_INSTALL_PREFIX=$DIR -DCUDA_NVCC_FLAGS="-arch=sm_61"

# make -j8 install VERBOSE=1
make -j8 install
