cmake_minimum_required(VERSION 2.6)
project(trt7-yolo)

file(GLOB_RECURSE source plugin/include/*.h plugin/src/*.cpp plugin/src/*.cu convert/include/*h convert/src/*.cpp)

set(CMAKE_CXX_COMPILIER "/usr/bin/g++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wno-write-strings")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,$ORIGIN")

include_directories(plugin/include convert/include)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
find_package(CUDA REQUIRED)

set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11;-g;-G;-gencode;arch=compute_30;code=sm_30)


if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message("embed_platform on")
    include_directories(/usr/local/cuda/targets/aarch64-linux/include)
    link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else ()
    message("embed_platform off")
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)
endif ()
message("CUDA Configure done")

# Find OpenCV
set(OpenCV_DIR "/home/songtf/opencv-3.4.0/build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


include_directories(/home/songtf/package/TensorRT-7.1.3.4.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn8.0/TensorRT-7.1.3.4/include)
link_directories(/home/songtf/package/TensorRT-7.1.3.4.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn8.0/TensorRT-7.1.3.4/lib)

cuda_add_library(detector SHARED ${source})


add_executable(trt7-yolo ${PROJECT_SOURCE_DIR}/main.cpp)
target_include_directories(detector PRIVATE extra/ modules/ ${OpenCV_INCLUDE_DIRS} ${CUDA_TOOLKIT_ROOT_DIR}/include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_link_libraries(detector nvinfer nvinfer_plugin nvcaffe_parser "stdc++fs")


target_link_libraries(trt7-yolo detector ${OpenCV_LIBS} nvinfer cudart)
add_definitions(-O3 -pthread)
