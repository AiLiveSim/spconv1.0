cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(SparseConv LANGUAGES CXX CUDA VERSION 1.0)









# Option to build tests
option(SPCONV_BuildTests "Build the unit tests when BUILD_TESTING is enabled." OFF)

# Set C++ standard and disable extensions
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES "52;60;61;70;75" CACHE STRING "Supported CUDA architectures" FORCE)

# Append the -allow-unsupported-compiler flag to the CUDA flags if using Visual Studio
if (CMAKE_GENERATOR MATCHES "Visual Studio")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
endif()

# Fix for GTest linking issues with PyTorch
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

# Include the FetchContent module
include(FetchContent)

# Set paths for global CUDA installation (12.4)
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4" CACHE PATH "CUDA Toolkit directory path")
set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4" CACHE PATH "Path to CUDA toolkit")
set(CUDNN_LIBRARY "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/lib/x64/cudnn.lib" CACHE PATH "CUDNN library path")
set(CUDNN_INCLUDE_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include" CACHE PATH "CUDNN include directory path")
set(NVTOOLEXT_HOME "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4" CACHE PATH "Path for some pre-built PyTorch versions")

# Append Boost path
list(APPEND CMAKE_PREFIX_PATH "C:/Users/yuihs/Downloads/boost_1_82_0/lib/cmake/Boost-1.85.0")
list(APPEND CMAKE_PREFIX_PATH "C:/Users/yuihs/Downloads/libtorch-win-shared-with-deps-2.4.1+cu124/libtorch")
# Find required packages
find_package(CUDAToolkit REQUIRED)
set(Boost_ROOT "C:/Users/yuihs/Downloads/boost_1_82_0")
find_package(Boost REQUIRED)

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

# Fetch fmt
FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt
    GIT_TAG 10.2.1
    OVERRIDE_FIND_PACKAGE
)

# Make fetched content available
FetchContent_MakeAvailable(pybind11 fmt googletest)

# Instruct Caffe2 to use CUDNN
set(CAFFE2_USE_CUDNN True)
find_package(Torch REQUIRED)

# Add subdirectories
add_subdirectory(src/spconv)
add_subdirectory(src/utils)

# Add tests if option is enabled
if(SPCONV_BuildTests)
    enable_testing()
    include(CTest)
    include(GoogleTest)
    add_subdirectory(test)
endif()
