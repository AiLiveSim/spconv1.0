#add_library(catch_main OBJECT src/catch_main.cpp)
#target_link_libraries(catch_main PRIVATE Catch2::Catch2)
#file(GLOB files "src/test_*.cpp")

#foreach(file ${files})
#    get_filename_component(file_basename ${file} NAME_WE)
#    string(REGEX REPLACE "test_([^$]+)" "test-\\1" testcase ${file_basename})

#    add_executable(${testcase} ${file} $<TARGET_OBJECTS:catch_main>)
#    set_property(TARGET ${testcase} PROPERTY CXX_STANDARD 14)

#    target_link_libraries(${testcase}
#        pybind11::embed
#        spconv
#        Catch2::Catch2)
#    
#    target_include_directories(${testcase}
#        PRIVATE
#            ${PROJECT_SOURCE_DIR}/include)
#    
#    add_test(NAME "${testcase}" COMMAND ${testcase} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
#endforeach()

set(Python3_EXECUTABLE "/usr/bin/python3")
set(Python3_FIND_STRATEGY LOCATION)
#find_package(Python3 COMPONENTS Interpreter Development Development.Module Development.Embed NumPy REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
message(STATUS "Python3_EXECUTABLE: ${Python3_EXECUTABLE}")
message(STATUS "Python3_VERSION: ${Python3_VERSION}")
message(STATUS "Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3_LIBRARIES: ${Python3_LIBRARIES}")
message(STATUS "Python3_Development_FOUND: ${Python3_Development_FOUND}")
message(STATUS "Python3_Development.Embed_FOUND: ${Python3_Development.Embed_FOUND}")
message(STATUS "Python3_NUMPY_FOUND: ${Python3_NumPy_FOUND}")
message(STATUS "Python3_NumPy_INCLUDE_DIRS: ${Python3_NumPy_INCLUDE_DIRS}")

add_executable(test_conv_rule src/test_conv_rule.cpp)
target_link_libraries(test_conv_rule
    PRIVATE
        Catch2::Catch2WithMain
        pybind11::embed
        Python3::Python
        Python3::NumPy
        spconv)

target_include_directories(test_conv_rule
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include)
catch_discover_tests(test_conv_rule)
