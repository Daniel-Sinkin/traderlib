cmake_minimum_required(VERSION 3.14)

# Suppress CMP0148 warning thrown by the cmake script of pybind
if(POLICY CMP0148)
    cmake_policy(SET CMP0148 NEW)
endif()

# Project name
project(traderlib VERSION 1.0 LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add the path to the vendorized pybind11
set(pybind11_DIR "${CMAKE_SOURCE_DIR}/external/pybind11/share/cmake/pybind11")
list(APPEND CMAKE_PREFIX_PATH "${pybind11_DIR}")

# Find pybind11
find_package(pybind11 REQUIRED)

# Add the src/main.cpp file
add_library(${PROJECT_NAME} MODULE src/main.cpp)

# Link pybind11
target_link_libraries(${PROJECT_NAME} PRIVATE pybind11::module)

# Set the output name for the Python module
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "")
set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".so")
set_target_properties(${PROJECT_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/traderlib")