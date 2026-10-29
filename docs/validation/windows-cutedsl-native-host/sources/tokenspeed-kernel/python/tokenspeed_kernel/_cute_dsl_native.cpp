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

// Native forwarding ABI for generated CuTe CUDA-dialect host code. All CUDA
// structures and launch attributes pass through the installed SDK unchanged.
// This does not reimplement or link NVIDIA's x64 CuTe runtime library.
#include <cuda.h>
#include <cuda_runtime_api.h>

#define TS_CUDA_FORWARD(name, signature, arguments) \
  extern "C" int _##name signature { return static_cast<int>(name arguments); }

TS_CUDA_FORWARD(cudaLibraryLoadData,
    (cudaLibrary_t* library, const void* code, cudaJitOption* opts, void** values,
     unsigned count, cudaLibraryOption* libopts, void** libvalues, unsigned libcount),
    (library, code, opts, values, count, libopts, libvalues, libcount))
TS_CUDA_FORWARD(cudaLibraryGetKernel,
    (cudaKernel_t* kernel, cudaLibrary_t library, const char* name), (kernel, library, name))
TS_CUDA_FORWARD(cudaLibraryUnload, (cudaLibrary_t library), (library))
TS_CUDA_FORWARD(cudaGetDevice, (int* device), (device))
TS_CUDA_FORWARD(cudaSetDevice, (int device), (device))
TS_CUDA_FORWARD(cudaGetDeviceCount, (int* count), (count))
TS_CUDA_FORWARD(cudaDeviceGetAttribute,
    (int* value, cudaDeviceAttr attr, int device), (value, attr, device))
TS_CUDA_FORWARD(cuKernelGetAttribute,
    (int* value, CUfunction_attribute attr, CUkernel kernel, CUdevice device),
    (value, attr, kernel, device))
TS_CUDA_FORWARD(cudaFuncSetAttribute,
    (const void* kernel, cudaFuncAttribute attr, int value), (kernel, attr, value))
TS_CUDA_FORWARD(cudaKernelSetAttributeForDevice,
    (cudaKernel_t kernel, cudaFuncAttribute attr, int value, int device),
    (kernel, attr, value, device))
extern "C" int _cudaLaunchKernelEx(const cudaLaunchConfig_t* config, const void* kernel, void** args) {
  return static_cast<int>(cudaLaunchKernelExC(config, kernel, args));
}
extern "C" int _cudaLaunchKernel(const void* kernel, int gx, int gy, int gz,
    int bx, int by, int bz, void** args, size_t smem, cudaStream_t stream) {
  return static_cast<int>(cudaLaunchKernel(kernel, dim3(gx, gy, gz), dim3(bx, by, bz), args, smem, stream));
}
TS_CUDA_FORWARD(cudaOccupancyMaxActiveBlocksPerMultiprocessor,
    (int* count, const void* kernel, int threads, size_t smem), (count, kernel, threads, smem))
TS_CUDA_FORWARD(cuTensorMapEncodeTiled,
    (CUtensorMap* map, CUtensorMapDataType type, cuuint32_t rank, void* address,
     const cuuint64_t* dims, const cuuint64_t* strides, const cuuint32_t* box,
     const cuuint32_t* element_strides, CUtensorMapInterleave interleave,
     CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2, CUtensorMapFloatOOBfill oob),
    (map, type, rank, address, dims, strides, box, element_strides, interleave, swizzle, l2, oob))
TS_CUDA_FORWARD(cuTensorMapReplaceAddress, (CUtensorMap* map, void* address), (map, address))
TS_CUDA_FORWARD(cudaMalloc, (void** ptr, size_t size), (ptr, size))
TS_CUDA_FORWARD(cudaMallocAsync, (void** ptr, size_t size, cudaStream_t stream), (ptr, size, stream))
TS_CUDA_FORWARD(cudaFree, (void* ptr), (ptr))
TS_CUDA_FORWARD(cudaFreeAsync, (void* ptr, cudaStream_t stream), (ptr, stream))
TS_CUDA_FORWARD(cudaMemcpy,
    (void* dst, const void* src, size_t size, cudaMemcpyKind kind), (dst, src, size, kind))
TS_CUDA_FORWARD(cudaMemcpyAsync,
    (void* dst, const void* src, size_t size, cudaMemcpyKind kind, cudaStream_t stream),
    (dst, src, size, kind, stream))
TS_CUDA_FORWARD(cudaMemset, (void* ptr, int value, size_t size), (ptr, value, size))
TS_CUDA_FORWARD(cudaMemsetAsync,
    (void* ptr, int value, size_t size, cudaStream_t stream), (ptr, value, size, stream))
TS_CUDA_FORWARD(cudaDeviceSynchronize, (), ())
TS_CUDA_FORWARD(cudaStreamSynchronize, (cudaStream_t stream), (stream))
TS_CUDA_FORWARD(cudaGetLastError, (), ())
#undef TS_CUDA_FORWARD
