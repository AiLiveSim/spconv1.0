add_executable(test_conv_rule src/test_conv_rule.cpp)

target_link_libraries(test_conv_rule
        pybind11::embed
        spconv
        GTest::gtest_main)

gtest_discover_tests(test_conv_rule)
