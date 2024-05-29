add_library(spconv SHARED
            all.cc
            indice.cc
            indice.cu 
            reordering.cc
            reordering.cu
            maxpool.cc
            maxpool.cu)

target_compile_definitions(spconv PRIVATE CUDA_NO_HALF __CUDA_NO_HALF_CONVERSIONS__ __CUDA_NO_HALF_OPERATORS__ __CUDA_NO_HALF2_OPERATORS__)

target_include_directories(spconv
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
    PRIVATE
        ${TORCH_INCLUDE_DIRS})

target_link_libraries(spconv
    torch
    CUDA::cudart 
    CUDA::cublas)

set_target_properties(spconv PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

install(TARGETS spconv DESTINATION lib)
