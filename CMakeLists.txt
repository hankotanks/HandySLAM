cmake_minimum_required(VERSION 3.14)

project(HandySLAM VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT DEFINED ORB_SLAM3_DIR)
    message(FATAL_ERROR "Provide ORB_SLAM3_DIR.")
endif()

find_package(Eigen3 3.3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(OpenCV 4.4 REQUIRED)

add_executable(HandySLAM src/main.cpp)

target_compile_definitions(HandySLAM PRIVATE ORB_SLAM3_DIR_VOC="${ORB_SLAM3_DIR}/Vocabulary/ORBvoc.txt")
target_compile_definitions(HandySLAM PRIVATE PROJECT_ROOT="${PROJECT_SOURCE_DIR}")

target_include_directories(HandySLAM 
    PRIVATE 
        ${PROJECT_SOURCE_DIR}/include
    SYSTEM PRIVATE 
        ${ORB_SLAM3_DIR}
        ${ORB_SLAM3_DIR}/include 
        ${ORB_SLAM3_DIR}/include/CameraModels
        ${ORB_SLAM3_DIR}/Thirdparty/Sophus
        ${OpenCV_INCLUDE_DIRS}
        ${Pangolin_INCLUDE_DIRS})
target_link_directories(HandySLAM PRIVATE ${ORB_SLAM3_DIR}/lib)
target_link_libraries(HandySLAM PRIVATE Eigen3::Eigen ${Pangolin_LIBRARIES} ORB_SLAM3 ${OpenCV_LIBS})