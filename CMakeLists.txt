cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(SparseConv LANGUAGES CXX CUDA VERSION 1.0)

option(SPCONV_BuildTests "Build the unit tests when BUILD_TESTING is enabled." OFF)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES "52;60;61;70;75" CACHE STRING "Supported CUDA architectures" FORCE)

# Include the FetchContent module
include(FetchContent)

# Fetch Catch2
FetchContent_Declare(
  catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.0.1 # Specify the tag, branch, or commit you want to use
)

# Fetch pybind11
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.11.0 # Specify the tag, branch, or commit you want to use
)

# Set C++ version to 17 -> you need a compiler that is C++17 compatible
set(CMAKE_CXX_STANDARD 17)

# Instruct Caffe to use CUDNN
set(CAFFE2_USE_CUDNN True)

# If Torch_DIR has not been defined, automatically look for TorchConfig.cmake in the
# activated Conda environment
if(NOT DEFINED Torch_DIR)
    # Find Torch based on the activated Conda environment
    if(DEFINED ENV{CONDA_PREFIX})
        # Look for TorchConfig.cmake files in the activated Conda environment
        file(GLOB_RECURSE CONDA_TORCH_CONFIG_FILE_LIST $ENV{CONDA_PREFIX}/*/TorchConfig.cmake)

        if(CONDA_TORCH_CONFIG_FILE_LIST)
            # Extract the first element from the list
            list(GET CONDA_TORCH_CONFIG_FILE_LIST 0 CONDA_TORCH_CONFIG_FILE_PATH)
            message(STATUS "Found TorchConfig.cmake at ${CONDA_TORCH_CONFIG_FILE_PATH}")
            # Get the directory from the path to the TorchConfig.cmake file
            get_filename_component(CONDA_TORCH_CONFIG_DIR ${CONDA_TORCH_CONFIG_FILE_PATH} DIRECTORY)
            # Set Torch_DIR so that Torch will be found by find_package
            set(Torch_DIR ${CONDA_TORCH_CONFIG_DIR} CACHE PATH "The directory containing a CMAKE configuration file for Torch." FORCE)
        else()
            message(WARNING "Did not find TorchConfig.cmake at ${CONDA_TORCH_PATH}")
        endif()
    endif()
endif()

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(Torch REQUIRED)
find_package(fmt REQUIRED)
FetchContent_MakeAvailable(pybind11)

# Add subdirectories
add_subdirectory(src/spconv)
add_subdirectory(src/utils)

# Add tests
if(SPCONV_BuildTests)
    # Make the content available
    FetchContent_MakeAvailable(catch2)
    enable_testing()
    include(CTest)
    include(Catch)
    add_subdirectory(test)
endif()
