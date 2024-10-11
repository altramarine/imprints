project(imprints)
cmake_minimum_required(VERSION 2.8)

add_compile_options(--std=c++11 -O3 -Wall -mfma -mavx2 -mavx)
include_directories(include)

add_executable(example src/main.cpp)
# add_executable(example_without_fmcd src/examples/example_without_fmcd.cpp)
# add_executable(example_bulk_load src/examples/example_bulk_load.cpp)