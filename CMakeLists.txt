cmake_minimum_required(VERSION 3.14)

# Project name
project(traderlib VERSION 1.0 LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Manually set pybind11_DIR using the installed package in the virtual environment
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -m pybind11 --cmake-dir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

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