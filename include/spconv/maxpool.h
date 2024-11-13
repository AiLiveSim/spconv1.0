// Copyright 2019 Yan Yan
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SPARSE_MAXPOOL_FUNCTOR_H_
#define SPARSE_MAXPOOL_FUNCTOR_H_
#include <tensorview/tensorview.h>
#include <ATen/ATen.h>
#include <spconv/mp_helper.h>

namespace spconv
{
    namespace functor
    {
        // General template declaration for SparseMaxPoolForwardFunctor
        template <typename Device, typename T, typename Index>
        struct SparseMaxPoolForwardFunctor
        {
            void operator()(tv::TensorView<T> outFeatures,
                            tv::TensorView<const T> inFeatures,
                            tv::TensorView<const Index> indices, int size);
        };

        // Specialization for GPU device
        template <typename T, typename Index>
        struct SparseMaxPoolForwardFunctor<tv::GPU, T, Index>
        {
            using vecload_type_t =
                std::conditional_t<std::is_same<T, at::Half>::value, int2, int4>;
            using kernel_block_t = mp_list_c<int, 64, 32, 16>;
            void operator()(const tv::GPU &d,
                            tv::TensorView<T> outFeatures,
                            tv::TensorView<const T> inFeatures,
                            tv::TensorView<const Index> indices, int size);
        };

        // General template declaration for SparseMaxPoolBackwardFunctor
        template <typename Device, typename T, typename Index>
        struct SparseMaxPoolBackwardFunctor
        {
            void operator()(tv::TensorView<const T> outFeatures,
                            tv::TensorView<const T> inFeatures,
                            tv::TensorView<const T> dout,
                            tv::TensorView<T> din,
                            tv::TensorView<const Index> indices, int size);
        };

        // Specialization for GPU device
        template <typename T, typename Index>
        struct SparseMaxPoolBackwardFunctor<tv::GPU, T, Index>
        {
            using vecload_type_t =
                std::conditional_t<std::is_same<T, at::Half>::value, int2, int4>;
            using kernel_block_t = mp_list_c<int, 64, 32, 16>;
            void operator()(const tv::GPU &d,
                            tv::TensorView<const T> outFeatures,
                            tv::TensorView<const T> inFeatures,
                            tv::TensorView<const T> dout,
                            tv::TensorView<T> din,
                            tv::TensorView<const Index> indices, int size);
        };

    } // namespace functor
} // namespace spconv

#endif