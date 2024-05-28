#------------
# spconv_nms
#------------
add_library(spconv_nms SHARED nms.cu)
set_target_properties(spconv_nms PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(spconv_nms PROPERTIES SOVERSION 1)
set_property(TARGET spconv_nms PROPERTY CXX_STANDARD 14)
set_property(TARGET spconv_nms PROPERTY CUDA_STANDARD 14)

target_include_directories(spconv_nms
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(spconv_nms
    CUDA::cudart)

install(TARGETS spconv_nms DESTINATION lib)

#--------------
# spconv_utils
#--------------
find_package(Boost REQUIRED)

add_library(spconv_utils SHARED all.cc)
set_target_properties(spconv_utils PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(spconv_utils PROPERTIES SOVERSION 1)
set_property(TARGET spconv_utils PROPERTY CXX_STANDARD 14)
set_property(TARGET spconv_utils PROPERTY CUDA_STANDARD 14)

set_target_properties(spconv_utils
    PROPERTIES PREFIX
        "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}")

target_include_directories(spconv_nms
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(spconv_utils
    spconv_nms
    CUDA::cudart
    pybind11::module
    Boost::boost)

install(TARGETS spconv_utils DESTINATION lib)
