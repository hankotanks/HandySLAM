#!/bin/bash

echo "Initializing submodules"
git submodule update --init --recursive

echo "Building ORB_SLAM3"
cd ORB_SLAM3
chmod +x ./build.sh
./build.sh
cd ..

echo "Installing PromptDA dependencies"
python3 -m venv env
source ./env/bin/activate
cd PromptDA
python3 -m pip install -r requirements.txt
deactivate
cd ..

echo "Building HandySLAM"
mkdir -p build 
cd build
cmake ..
make -j$(nproc)
