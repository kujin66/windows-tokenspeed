// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "flashinfer/attention/decode.cuh"
#include "flashinfer/attention/scheduler.cuh"
#include "flashinfer/comm/trtllm_allreduce.cuh"
#include "flashinfer/comm/trtllm_allreduce_fusion.cuh"
#include "flashinfer/comm/trtllm_reducescatter_fusion.cuh"
#include "tvm_ffi_utils.h"
#include <spdlog/fmt/bundled/format.h>

template <typename T>
__device__ int64_t sentinel_bits(T value);

template <>
__device__ int64_t sentinel_bits(half value) {
  return __half_as_ushort(value);
}
template <>
__device__ int64_t sentinel_bits(__nv_bfloat16 value) {
  return __bfloat16_as_ushort(value);
}
template <>
__device__ int64_t sentinel_bits(float value) {
  return static_cast<uint32_t>(__float_as_int(value));
}

template <typename T>
__global__ void read_sentinels(int64_t* output) {
  namespace ar = flashinfer::trtllm_allreduce;
  namespace arf = flashinfer::trtllm_allreduce_fusion;
  namespace rs = flashinfer::trtllm_reducescatter_fusion::utils;
  const T a = ar::neg_zero<T>::value;
  const T b = arf::neg_zero<T>::value;
  const T c = rs::neg_zero<T>::value;
  const T positive_zero = static_cast<T>(0.0f);
  output[0] = sentinel_bits(a);
  output[1] = ar::is_negative_zero(a);
  output[2] = ar::is_negative_zero(positive_zero);
  output[3] = sentinel_bits(b);
  output[4] = arf::is_negative_zero(b);
  output[5] = arf::is_negative_zero(positive_zero);
  output[6] = sentinel_bits(c);
  output[7] = rs::is_negative_zero(c);
  output[8] = rs::is_negative_zero(positive_zero);
}

void collective_sentinels(TensorView output) {
  TVM_FFI_ICHECK_EQ(output.numel(), 27);
  TVM_FFI_ICHECK(output.dtype() == dl_int64);
  int64_t* ptr = static_cast<int64_t*>(output.data_ptr());
  const auto stream = get_stream(output.device());
  read_sentinels<half><<<1, 1, 0, stream>>>(ptr);
  read_sentinels<__nv_bfloat16><<<1, 1, 0, stream>>>(ptr + 9);
  read_sentinels<float><<<1, 1, 0, stream>>>(ptr + 18);
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(collective_sentinels, collective_sentinels);

int64_t fmt_rounding_threshold(int64_t index) {
  TVM_FFI_ICHECK_GE(index, 0);
  TVM_FFI_ICHECK_LT(index, 8);
  return fmt::detail::fractional_part_rounding_thresholds(static_cast<int>(index));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fmt_rounding_threshold, fmt_rounding_threshold);

__global__ void half_math_kernel(const half* input, half* output, int64_t count) {
  const int index = threadIdx.x;
  if (index < count) {
    output[index] = flashinfer::math::ptx_exp2(input[index]);
    output[count + index] = flashinfer::math::tanh(input[index]);
  }
}

void half_math(TensorView input, TensorView output) {
  TVM_FFI_ICHECK(input.dtype() == dl_float16);
  TVM_FFI_ICHECK(output.dtype() == dl_float16);
  TVM_FFI_ICHECK_LE(input.numel(), 32);
  TVM_FFI_ICHECK_EQ(output.numel(), input.numel() * 2);
  half_math_kernel<<<1, 32, 0, get_stream(input.device())>>>(
      static_cast<const half*>(input.data_ptr()), static_cast<half*>(output.data_ptr()), input.numel());
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(half_math, half_math);
