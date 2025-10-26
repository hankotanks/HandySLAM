#!/bin/bash
git submodule update --init --recursive
cd Pangolin
./scripts/install_prerequisites.sh --dry-run required
cmake -B build && cmake --build build
cd ..
chmod +x ./scripts/build_orb_slam3.sh
cd ORB_SLAM3
sed -i 's/++11/++14/g' CMakeLists.txt
../scripts/build_orb_slam3.sh
cd ..
cmake -B build
