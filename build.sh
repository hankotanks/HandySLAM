#!/bin/bash

echo "Initializing submodules"

git submodule update --init --recursive

echo "Building ORB_SLAM3"

cd ORB_SLAM3
chmod +x ./build.sh
./build.sh

cd ..

echo "Building HandySLAM"

mkdir -p build 
cd build
cmake ..
