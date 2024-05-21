#include <algorithm>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <exception>
#include <numeric>
#include <tuple>
#include <iostream>

#include <catch2/catch_test_macros.hpp>

#include "spconv/spconv_ops.h"
#include "pybind11_utils.h"
#include "prettyprint.h"

namespace py = pybind11;

TEST_CASE("GetConvIndPair", "[SpConvNet]")
{
       auto inds = array2TensorView<int>(py::array(py::globals()["indices"]));
       auto inds_tensor = torch::from_blob(inds.data(), {inds.dim(0), inds.dim(1)}, torch::dtype(torch::kInt32));
       auto inds_gpu = inds_tensor.to(torch::Device(torch::kCPU));

       auto features = array2TensorView<float>(py::array(py::globals()["features"]));
       auto features_tensor = torch::from_blob(features.data(), {features.dim(0), features.dim(1)}, torch::dtype(torch::kFloat));
       auto features_gpu = features_tensor.to(torch::Device(torch::kCUDA, 0));
       auto filters = array2TensorView<float>(py::array(py::globals()["filters"]));
       auto filters_tensor = torch::from_blob(filters.data(), {filters.dim(0), filters.dim(1), filters.dim(2), filters.dim(3), filters.dim(4)}, torch::dtype(torch::kFloat));
       auto filters_gpu = filters_tensor.to(torch::Device(torch::kCUDA, 0));

       auto outputs = spconv::getIndicePair<3>(inds_gpu, 1, {46, 26, 26}, {50, 30, 30}, {3, 3, 3},
                                               {1, 1, 1}, {0, 0, 0}, {2, 2, 2}, {0, 0, 0}, false, false);
}
