cmake_minimum_required(VERSION 3.14)

project(HandySLAM VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wno-deprecated-declarations)

find_package(Eigen3 3.3 REQUIRED)
set(Pangolin_DIR "${PROJECT_SOURCE_DIR}/Pangolin/build")
find_package(Pangolin REQUIRED)
find_package(OpenCV 4.4 REQUIRED)

add_executable(HandySLAM 
    src/main.cpp
    src/DataloaderStray.cpp
    src/Dataloader.cpp)

target_compile_definitions(HandySLAM PRIVATE PROJECT_ROOT="${PROJECT_SOURCE_DIR}")

set(VOCAB_PATH "${PROJECT_SOURCE_DIR}/ORB_SLAM3/Vocabulary/ORBvoc.txt")
set(PROJECT_ROOT "${PROJECT_SOURCE_DIR}")
configure_file(Config.h.in ${PROJECT_BINARY_DIR}/Config.h @ONLY)

target_include_directories(HandySLAM 
    PRIVATE 
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_BINARY_DIR}
    SYSTEM PRIVATE 
        ${PROJECT_SOURCE_DIR}/ORB_SLAM3
        ${PROJECT_SOURCE_DIR}/ORB_SLAM3/include 
        ${PROJECT_SOURCE_DIR}/ORB_SLAM3/include/CameraModels
        ${PROJECT_SOURCE_DIR}/ORB_SLAM3/Thirdparty/Sophus
        ${OpenCV_INCLUDE_DIRS}
        ${Pangolin_INCLUDE_DIRS})
target_link_directories(HandySLAM PRIVATE ${PROJECT_SOURCE_DIR}/ORB_SLAM3/lib)
target_link_libraries(HandySLAM PRIVATE Eigen3::Eigen ${Pangolin_LIBRARIES} ORB_SLAM3 ${OpenCV_LIBS})