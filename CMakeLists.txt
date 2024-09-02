cmake_minimum_required(VERSION 3.14)

# Suppress some warnings about the pybind11 internal cmake script
if(POLICY CMP0148)
    cmake_policy(SET CMP0148 NEW)
endif()


# Project name
project(traderlib VERSION 1.0 LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Manually set pybind11_DIR relative to the traderlib directory
set(pybind11_DIR "${CMAKE_SOURCE_DIR}/.venv/lib/python3.12/site-packages/pybind11/share/cmake/pybind11")

# Find pybind11
find_package(pybind11 REQUIRED)

# Add the src/main.cpp file
add_library(${PROJECT_NAME} MODULE ${CMAKE_SOURCE_DIR}/src/main.cpp)

# Link pybind11
target_link_libraries(${PROJECT_NAME} PRIVATE pybind11::module)

# Set the output name for the Python module
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "")
set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".so")