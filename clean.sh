#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Uninstall the traderlib package using pip
pip uninstall -y traderlib

# Remove the build directory if it exists
if [ -d "build" ]; then
    rm -r build
    echo "Removed build directory."
else
    echo "Build directory does not exist."
fi

# Remove the traderlib.egg-info directory if it exists
if [ -d "traderlib.egg-info" ]; then
    rm -r traderlib.egg-info
    echo "Removed traderlib.egg-info directory."
else
    echo "traderlib.egg-info directory does not exist."
fi

# Remove the dist directory if it exists
if [ -d "dist" ]; then
    rm -r dist
    echo "Removed dist directory."
else
    echo "Dist directory does not exist."
fi

# Remove the traderlib.so file if it exists
if [ -f "traderlib/traderlib.so" ]; then
    rm traderlib/traderlib.so
    echo "Removed traderlib.so file."
else
    echo "traderlib.so file does not exist."
fi

echo "Clean-up completed successfully."