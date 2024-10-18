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

target_sources(spconv PRIVATE
    all.cc
    indice.cc
    indice.cu
    maxpool.cc
    maxpool.cu
    reordering.cc
    reordering.cu
    ${PROJECT_SOURCE_DIR}/include/spconv/reordering.h
    ${PROJECT_SOURCE_DIR}/include/spconv/reordering.h
    ${PROJECT_SOURCE_DIR}/include/spconv/reordering.cu.h
    ${PROJECT_SOURCE_DIR}/include/spconv/pool_ops.h
    ${PROJECT_SOURCE_DIR}/include/spconv/spconv_ops.h
    ${PROJECT_SOURCE_DIR}/include/spconv/geometry.h
    ${PROJECT_SOURCE_DIR}/include/spconv/indice.h
    ${PROJECT_SOURCE_DIR}/include/spconv/indice.cu.h
    ${PROJECT_SOURCE_DIR}/include/spconv/mp_helper.h
    ${PROJECT_SOURCE_DIR}/include/spconv/maxpool.h
    ${PROJECT_SOURCE_DIR}/include/tensorview/tensorview.h
    ${PROJECT_SOURCE_DIR}/include/tensorview/helper_launch.h
    ${PROJECT_SOURCE_DIR}/include/tensorview/helper_kernel.cu.h
    ${PROJECT_SOURCE_DIR}/include/tensorview/helper_launch.h
    ${PROJECT_SOURCE_DIR}/include/tensorview/tensorview.h
)

install(TARGETS spconv DESTINATION lib)
