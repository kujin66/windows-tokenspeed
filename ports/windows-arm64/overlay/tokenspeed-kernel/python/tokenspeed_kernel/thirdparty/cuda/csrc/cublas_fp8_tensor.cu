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


#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include "tvm_ffi_utils.h"

namespace {
void check(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string(operation) + " failed with cuBLAS status " + std::to_string(status));
}
void cuda_check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template <typename T, auto Destroy> struct Resource {
  T value = nullptr;
  ~Resource() { if (value) Destroy(value); }
};
struct DeviceGuard {
  int original = 0;
  explicit DeviceGuard(int device) { cuda_check(cudaGetDevice(&original)); cuda_check(cudaSetDevice(device)); }
  ~DeviceGuard() { cudaSetDevice(original); }
};
}

void cublas_fp8_tensor(TensorView a, TensorView b, TensorView a_scale, TensorView b_scale,
                       TensorView out, TensorView workspace, int64_t stream_address) {
  if (a.ndim() != 2 || b.ndim() != 2 || out.ndim() != 2)
    throw std::invalid_argument("FP8 GEMM requires rank-two matrices");
  int64_t m = a.size(0), k = a.size(1), n = b.size(1);
  if (m <= 0 || n <= 0 || k <= 0)
    throw std::invalid_argument("FP8 GEMM requires positive dimensions");
  if (b.size(0) != k || out.size(0) != m || out.size(1) != n)
    throw std::invalid_argument("FP8 GEMM matrix shapes disagree");
  if (a.stride(1) != 1 || a.stride(0) != k || b.stride(0) != 1 ||
      b.stride(1) != k || out.stride(1) != 1 || out.stride(0) != n)
    throw std::invalid_argument("FP8 GEMM requires dense TN storage");
  if (a_scale.numel() != 1 || b_scale.numel() != 1 || workspace.ndim() != 1 ||
      workspace.dtype() != dl_uint8 || workspace.stride(0) != 1)
    throw std::invalid_argument("FP8 GEMM requires scalar scales and byte workspace");
  for (TensorView tensor : {a, b, a_scale, b_scale, out, workspace}) {
    if (tensor.device().device_type != kDLCUDA ||
        tensor.device().device_id != a.device().device_id)
      throw std::invalid_argument("FP8 GEMM tensors must share a CUDA device");
  }
  if (a.dtype() != dl_float8_e4m3fn || b.dtype() != dl_float8_e4m3fn ||
      a_scale.dtype() != dl_float32 || b_scale.dtype() != dl_float32)
    throw std::invalid_argument("FP8 GEMM requires E4M3 operands and FP32 scales");
  cudaDataType_t output_type;
  if (out.dtype() == dl_bfloat16) output_type = CUDA_R_16BF;
  else if (out.dtype() == dl_float16) output_type = CUDA_R_16F;
  else throw std::invalid_argument("FP8 GEMM writes BF16 or FP16");
  DeviceGuard device(a.device().device_id);
  Resource<cublasLtHandle_t, cublasLtDestroy> handle;
  Resource<cublasLtMatmulDesc_t, cublasLtMatmulDescDestroy> operation;
  Resource<cublasLtMatrixLayout_t, cublasLtMatrixLayoutDestroy> left, right, result;
  Resource<cublasLtMatmulPreference_t, cublasLtMatmulPreferenceDestroy> preference;
  check(cublasLtCreate(&handle.value), "cublasLtCreate");
  check(cublasLtMatmulDescCreate(&operation.value, CUBLAS_COMPUTE_32F, CUDA_R_32F), "MatmulDescCreate");
  cublasOperation_t transpose = CUBLAS_OP_T;
  check(cublasLtMatmulDescSetAttribute(operation.value, CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)), "SetTranspose");
  uint8_t fast_accum = 0;
  check(cublasLtMatmulDescSetAttribute(operation.value, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast_accum, sizeof(fast_accum)), "SetAccumulation");
  // Compute out.T = weight @ activations.T using FP8's required TN layout.
  const void* weight_scale = b_scale.data_ptr();
  const void* input_scale = a_scale.data_ptr();
  check(cublasLtMatmulDescSetAttribute(operation.value, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &weight_scale, sizeof(weight_scale)), "SetWeightScale");
  check(cublasLtMatmulDescSetAttribute(operation.value, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &input_scale, sizeof(input_scale)), "SetInputScale");
  check(cublasLtMatrixLayoutCreate(&left.value, CUDA_R_8F_E4M3, k, n, k), "LeftLayout");
  check(cublasLtMatrixLayoutCreate(&right.value, CUDA_R_8F_E4M3, k, m, k), "RightLayout");
  check(cublasLtMatrixLayoutCreate(&result.value, output_type, n, m, n), "OutputLayout");
  check(cublasLtMatmulPreferenceCreate(&preference.value), "PreferenceCreate");
  size_t workspace_bytes = workspace.size(0);
  check(cublasLtMatmulPreferenceSetAttribute(preference.value, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes)), "SetWorkspace");
  cublasLtMatmulHeuristicResult_t algorithm{};
  int count = 0;
  check(cublasLtMatmulAlgoGetHeuristic(handle.value, operation.value, left.value, right.value, result.value, result.value, preference.value, 1, &algorithm, &count), "GetHeuristic");
  if (!count) throw std::runtime_error("No cuBLASLt FP8 algorithm supports these shapes");
  check(algorithm.state, "HeuristicAlgorithm");
  float alpha = 1.0f, beta = 0.0f;
  check(cublasLtMatmul(handle.value, operation.value, &alpha, b.data_ptr(), left.value, a.data_ptr(), right.value,
                      &beta, out.data_ptr(), result.value, out.data_ptr(), result.value, &algorithm.algo,
                      workspace.data_ptr(), workspace_bytes, reinterpret_cast<cudaStream_t>(stream_address)), "Matmul");
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(cublas_fp8_tensor, cublas_fp8_tensor);
