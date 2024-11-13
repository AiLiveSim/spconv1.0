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

#ifndef SPARSE_REORDERING_FUNCTOR_H_
#define SPARSE_REORDERING_FUNCTOR_H_
#include <tensorview/tensorview.h>
#include <ATen/ATen.h>
#include <spconv/mp_helper.h>

namespace spconv {
    namespace functor {
        // Base declaration for SparseGatherFunctor
        template <typename Device,typename T, typename Index>
        struct SparseGatherFunctor
        {
            void operator()(tv::TensorView<T> buffer,
                            tv::TensorView<const T> features,
                            tv::TensorView<const Index> indices,
                            int size);
        };

        // GPU specialization for SparseGatherFunctor
        template <typename T, typename Index>
        struct SparseGatherFunctor<tv::GPU, T, Index>
        {
            using vecload_type_t = std::conditional_t<std::is_same<T, at::Half>::value, int2, int4>;
            using kernel_block_t = mp_list_c<int, 64, 32, 16>;
            void operator()(const tv::GPU &d,  // Device parameter for GPU specialization
                            tv::TensorView<T> buffer,
                            tv::TensorView<const T> features,
                            tv::TensorView<const Index> indices,
                            int size);
        };

        // Base declaration for SparseScatterAddFunctor
        template <typename Device, typename T, typename Index>
        struct SparseScatterAddFunctor
        {
            void operator()(tv::TensorView<T> out_features,
                            tv::TensorView<const T> buffer,
                            tv::TensorView<const Index> indices,
                            int size);
        };

        // GPU specialization for SparseScatterAddFunctor
        template <typename T, typename Index>
        struct SparseScatterAddFunctor<tv::GPU, T, Index>
        {
            using vecload_type_t =
                std::conditional_t<std::is_same<T, at::Half>::value, int2, int4>;
            using kernel_block_t = mp_list_c<int, 64, 32, 16>;
            void operator()(const tv::GPU &d,  // Device parameter for GPU specialization
                            tv::TensorView<T> out_features,
                            tv::TensorView<const T> buffer,
                            tv::TensorView<const Index> indices,
                            int size);
        };
    } // namespace functor
} // namespace spconv

#endif