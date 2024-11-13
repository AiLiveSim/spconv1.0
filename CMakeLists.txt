cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(SparseConv LANGUAGES CXX CUDA VERSION 1.0)

option(SPCONV_BuildTests "Build the unit tests when BUILD_TESTING is enabled." OFF)

set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES "52;60;61;70;75" CACHE STRING "Supported CUDA architectures" FORCE)

# Append the -allow-unsupported-compiler flag to the CUDA flags
if (CMAKE_GENERATOR MATCHES "Visual Studio")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
endif()

# The reason this needs to be set, take a look at https://stackoverflow.com/questions/62693218/how-to-solve-gtest-and-libtorch-linkage-conflict
# If this is not used, GTest linking fails if linking against PyTorch
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

# Set C++ standard to C++17 or later
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set strict compilation flags for C++ compilers
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # Strict flags for GCC and Clang
    add_compile_options(
        -Wall          # Enable all warnings
        -Wextra        # Enable extra warnings
        -Wno-unknown-pragmas # Disable unroll warnings
        -O3
        # -Wpedantic     # Enforce strict compliance with standards. Comment to remove unwanted  warning: style of line directive is a GCC extension when using clang
    )
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # Strict flags for MSVC
    add_compile_options(
        /W4            # Enable high-level warnings
        /WX            # Treat all warnings as errors
    )
endif()

# CUDA-specific strict flags (only if CUDA is found)
if (CMAKE_CUDA_COMPILER)
    # Standard warnings and error flags
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wall,-Wextra")

    # Optional: suppress specific CUDA warnings if needed
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe \"--diag_suppress=unsigned_compare_with_zero\"")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -O3")
endif()

# Include the FetchContent module
include(FetchContent)

# If a Conda environment is enabled, we look for the CUDA toolkit in that environment
if(DEFINED ENV{CONDA_PREFIX})
    message(STATUS "CONDA environment detected, path: $ENV{CONDA_PREFIX}")
    set(CUDAToolkit_ROOT "$ENV{CONDA_PREFIX}" CACHE PATH "Path to CUDA toolkit")
    set(CUDA_TOOLKIT_ROOT_DIR $ENV{CONDA_PREFIX} CACHE PATH "CUDA Toolkit directory path (used by older/other find_package implementations)")
    # Some Caffe2 find-package(s) need these if we want to find cudnn in Windows
    set(CUDNN_LIBRARY "$ENV{CONDA_PREFIX};$ENV{CONDA_PREFIX}/Library" CACHE PATH "CUDNN directory path (used by some Caffe2 find-packages)")
    set(CUDNN_INCLUDE_DIR "$ENV{CONDA_PREFIX};$ENV{CONDA_PREFIX}/Library" CACHE PATH "CUDNN directory path (used by some Caffe2 find-packages)")
    set(NVTOOLEXT_HOME $ENV{CONDA_PREFIX} CACHE PATH "Some pre-built PyTorch versions require this")
    # Boost should be installed in the environment
    list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}/Library/lib/cmake/Boost-1.85.0")
endif()

find_package(CUDAToolkit)

# Boost support
#include(cmake/Boost.cmake)

find_package(Boost CONFIG)

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
