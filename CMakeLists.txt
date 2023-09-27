cmake_minimum_required(VERSION 3.26)
project(cuda_programming CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(cuda_programming
        main.cu)
set_target_properties(cuda_programming PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
