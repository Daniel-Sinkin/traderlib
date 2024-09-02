#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Create and navigate to the build directory
mkdir -p build
cd build

# Run CMake to configure and generate the build files
cmake ..

# Build the project using CMake
cmake --build .

# Navigate back to the root directory
cd ..

# Run the Python setup script to create source and wheel distributions
python3 setup.py sdist bdist_wheel

echo "Build and packaging completed successfully."