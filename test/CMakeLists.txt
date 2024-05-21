add_executable(test_conv_rule src/test_conv_rule.cpp)

target_link_libraries(test_conv_rule
    PRIVATE
        Catch2::Catch2WithMain
        pybind11::embed
        spconv)

catch_discover_tests(test_conv_rule)
