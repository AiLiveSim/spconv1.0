set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4" CACHE PATH "CUDA Toolkit directory path")
set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4" CACHE PATH "Path to CUDA toolkit")
set(CUDNN_LIBRARY "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/lib/x64/cudnn.lib" CACHE PATH "CUDNN library path")
set(CUDNN_INCLUDE_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include" CACHE PATH "CUDNN include directory path")
set(NVTOOLEXT_HOME "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4" CACHE PATH "Path for some pre-built PyTorch versions")

# Append Boost path
list(APPEND CMAKE_PREFIX_PATH "C:/Users/User/Downloads/boost_1_86_0/boost_1_86_0")
list(APPEND CMAKE_PREFIX_PATH "C:/Users/User/Downloads/libtorch-win-shared-with-deps-2.4.1+cu124/libtorch")