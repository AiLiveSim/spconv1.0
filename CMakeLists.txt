cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(SparseConv LANGUAGES CXX CUDA VERSION 1.0)

option(SPCONV_BuildTests "Build the unit tests when BUILD_TESTING is enabled." OFF)

set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES "52;60;61;70;75" CACHE STRING "Supported CUDA architectures" FORCE)

# The reason this needs to be set, take a look at https://stackoverflow.com/questions/62693218/how-to-solve-gtest-and-libtorch-linkage-conflict
# If this is not used, GTest linking fails if linking against PyTorch
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

# Set C++ standard to C++17 or later
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Include the FetchContent module
include(FetchContent)

# Boost support
include(cmake/Boost.cmake)

# Fetch GTest
 FetchContent_Declare(
   googletest
   GIT_REPOSITORY https://github.com/google/googletest.git
   GIT_TAG v1.14.0
 )

# Fetch pybind11
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.11.0
  OVERRIDE_FIND_PACKAGE
)

FetchContent_MakeAvailable(pybind11)

# Fetch fmt
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_TAG 10.2.1
  OVERRIDE_FIND_PACKAGE
)

FetchContent_MakeAvailable(fmt)

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

# Add subdirectories
add_subdirectory(src/spconv)
add_subdirectory(src/utils)

# Add tests
if(SPCONV_BuildTests)
    # Make the content available
    enable_testing()
    include(CTest)
    FetchContent_MakeAvailable(googletest)
    include(GoogleTest)
    add_subdirectory(test)
endif()
