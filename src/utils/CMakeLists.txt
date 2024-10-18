#--------------
# spconv_utils
#--------------
add_library(spconv_utils SHARED all.cc nms.cu)
set_target_properties(spconv_utils PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(spconv_utils PROPERTIES SOVERSION 1)
set_property(TARGET spconv_utils PROPERTY CXX_STANDARD 14)
set_property(TARGET spconv_utils PROPERTY CUDA_STANDARD 14)

set_target_properties(spconv_utils
    PROPERTIES PREFIX
        "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}")

target_include_directories(spconv_utils
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(spconv_utils
    CUDA::cudart
    pybind11::module
    Boost::boost)

target_sources(spconv_utils PRIVATE
    nms.cu
    all.cc
    ${PROJECT_SOURCE_DIR}/include/spconv/nms.h
    ${PROJECT_SOURCE_DIR}/include/spconv/point2voxel.h
    ${PROJECT_SOURCE_DIR}/include/spconv/box_iou.h
    ${PROJECT_SOURCE_DIR}/include/spconv/nms_gpu.h
)

install(TARGETS spconv_utils DESTINATION lib)
