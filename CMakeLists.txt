cmake_minimum_required(VERSION 3.14)

project(HandySLAM VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(USE_SYSTEM_ZLIB ON CACHE BOOL "Use system zlib for Pangolin.")

# set(GLIBCXX_USE_CXX11_ABI ON CACHE BOOL "Fix for Open3D build process.")
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
set(BUILD_SHARED_LIBS ON CACHE BOOL "Enable shared library building.")
set(BUILD_EXAMPLES OFF CACHE BOOL "Disable Open3D examples.")
set(BUILD_PYTHON_MODULE OFF CACHE BOOL "Don't build Open3D python module.")
set(BUILD_CUDA_MODULE OFF CACHE BOOL "")
set(OPEN3D_ENABLE_HEADLESS_RENDERING ON CACHE BOOL "" FORCE)
set(DEVELOPER_BUILD OFF CACHE BOOL "Build Open3D release.")
set(BUILD_GUI OFF CACHE BOOL "Disable GUI in Open3D.")
set(USE_SYSTEM_OPENSSL ON CACHE BOOL "Use system pre-installed OpenSSL.")
set(USE_SYSTEM_CURL ON CACHE BOOL "Use system pre-installed curl.")
# set(BUILD_FILAMENT_FROM_SOURCE ON CACHE BOOL "Build Filament from source for Open3D.")
add_subdirectory(${PROJECT_SOURCE_DIR}/Open3D)

find_package(Eigen3 3.3 REQUIRED)
set(Pangolin_DIR "${PROJECT_SOURCE_DIR}/ORB_SLAM3/Pangolin/build")
find_package(Pangolin REQUIRED)
find_package(OpenCV 4.4 REQUIRED)
find_package(ZLIB REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(lz4 REQUIRED IMPORTED_TARGET liblz4)
find_package(OpenSSL REQUIRED)

add_executable(HandySLAM 
    src/main.cpp
    src/Initializer.cpp
    src/DataloaderStray.cpp
    src/DataloaderScanNet.cpp
    src/Dataloader.cpp
    src/VolumeBuilder.cpp)

target_compile_definitions(HandySLAM PRIVATE 
    PROJECT_ROOT="${PROJECT_SOURCE_DIR}")

target_compile_options(HandySLAM PUBLIC
    -Wno-unused-variable
    -Wno-deprecated-declarations)

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
target_link_directories(HandySLAM PRIVATE 
    ${PROJECT_SOURCE_DIR}/ORB_SLAM3/lib)
target_link_libraries(HandySLAM PRIVATE 
    ZLIB::ZLIB 
    PkgConfig::lz4 
    Eigen3::Eigen 
    ${Pangolin_LIBRARIES} 
    ORB_SLAM3 
    ${OpenCV_LIBS}
    Open3D::Open3D
    OpenSSL::SSL
    OpenSSL::Crypto)