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

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

static_assert(_MSC_VER == 1951, "This regression requires the actual MSVC 14.51 compiler");
#ifndef _M_ARM64
#error This regression requires an ARM64 host target
#endif

__global__ void add(const int* a, const int* b, int* out, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) out[i] = a[i] + b[i];
}

int main() {
    constexpr int size = 4097;
    std::vector<int> a(size), b(size), result(size);
    for (int i = 0; i < size; ++i) { a[i] = i * 3; b[i] = 100 - i; }
    auto check = [](cudaError_t code) {
        if (code != cudaSuccess) {
            std::fprintf(stderr, "CUDA failure: %s\n", cudaGetErrorString(code));
            return false;
        }
        return true;
    };
    if (!check(cudaSetDevice(0))) return 1;
    int *da = nullptr, *db = nullptr, *dout = nullptr;
    std::size_t bytes = size * sizeof(int);
    if (!check(cudaMalloc(&da, bytes)) || !check(cudaMalloc(&db, bytes)) ||
        !check(cudaMalloc(&dout, bytes))) return 2;
    if (!check(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice)) ||
        !check(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice))) return 3;
    add<<<(size + 255) / 256, 256>>>(da, db, dout, size);
    if (!check(cudaGetLastError()) || !check(cudaDeviceSynchronize()) ||
        !check(cudaMemcpy(result.data(), dout, bytes, cudaMemcpyDeviceToHost))) return 4;
    if (!check(cudaFree(da)) || !check(cudaFree(db)) || !check(cudaFree(dout))) return 5;
    for (int i = 0; i < size; ++i) if (result[i] != a[i] + b[i]) return 6;
    std::printf("{\"passed\":true,\"elements\":%d,\"msc_ver\":%d,\"msc_full_ver\":%d,\"cuda_runtime_version\":%d}\n",
                size, _MSC_VER, _MSC_FULL_VER, CUDART_VERSION);
    return 0;
}
