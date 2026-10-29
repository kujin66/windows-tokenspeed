# Complete Windows ARM64 source inventory

These are complete source postimages from validated port `9a50ac82`,
applied to official upstream `864bcdd9`. Unchanged files come from that
pinned upstream tree. No source changes from the port within these five
components are omitted. SHA256 and before/after Git blob IDs are in
[source-manifest.json](source-manifest.json).

| Component | Changed/added files |
| --- | ---: |
| python | 16 |
| test | 4 |
| tokenspeed-engine | 9 |
| tokenspeed-kernel | 32 |
| tokenspeed-scheduler | 5 |

| Original path | Full source in this PR |
| --- | --- |
| python/pyproject.toml | [source](overlay/python/pyproject.toml) |
| python/tokenspeed/env.py | [source](overlay/python/tokenspeed/env.py) |
| python/tokenspeed/runtime/distributed/comm_backend/nccl.py | [source](overlay/python/tokenspeed/runtime/distributed/comm_backend/nccl.py) |
| python/tokenspeed/runtime/engine/async_llm.py | [source](overlay/python/tokenspeed/runtime/engine/async_llm.py) |
| python/tokenspeed/runtime/engine/rust_host.py | [source](overlay/python/tokenspeed/runtime/engine/rust_host.py) |
| python/tokenspeed/runtime/entrypoints/engine.py | [source](overlay/python/tokenspeed/runtime/entrypoints/engine.py) |
| python/tokenspeed/runtime/execution/distributed_initializer.py | [source](overlay/python/tokenspeed/runtime/execution/distributed_initializer.py) |
| python/tokenspeed/runtime/execution/drafter/__init__.py | [source](overlay/python/tokenspeed/runtime/execution/drafter/__init__.py) |
| python/tokenspeed/runtime/layers/attention/backends/__init__.py | [source](overlay/python/tokenspeed/runtime/layers/attention/backends/__init__.py) |
| python/tokenspeed/runtime/layers/attention/registry.py | [source](overlay/python/tokenspeed/runtime/layers/attention/registry.py) |
| python/tokenspeed/runtime/layers/quantization/compressed_tensors/compressed_tensors.py | [source](overlay/python/tokenspeed/runtime/layers/quantization/compressed_tensors/compressed_tensors.py) |
| python/tokenspeed/runtime/layers/quantization/compressed_tensors/gptq_marlin_moe.py | [source](overlay/python/tokenspeed/runtime/layers/quantization/compressed_tensors/gptq_marlin_moe.py) |
| python/tokenspeed/runtime/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py | [source](overlay/python/tokenspeed/runtime/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py) |
| python/tokenspeed/runtime/models/registry.py | [source](overlay/python/tokenspeed/runtime/models/registry.py) |
| python/tokenspeed/runtime/utils/common.py | [source](overlay/python/tokenspeed/runtime/utils/common.py) |
| python/tokenspeed/runtime/utils/process.py | [source](overlay/python/tokenspeed/runtime/utils/process.py) |
| test/runtime/test_compressed_tensors_fp8.py | [source](overlay/test/runtime/test_compressed_tensors_fp8.py) |
| test/runtime/test_compressed_tensors_nvfp4.py | [source](overlay/test/runtime/test_compressed_tensors_nvfp4.py) |
| test/runtime/test_runtime_plugins.py | [source](overlay/test/runtime/test_runtime_plugins.py) |
| test/runtime/test_rust_host.py | [source](overlay/test/runtime/test_rust_host.py) |
| tokenspeed-engine/crates/ts-grammar/CMakeLists.txt | [source](overlay/tokenspeed-engine/crates/ts-grammar/CMakeLists.txt) |
| tokenspeed-engine/crates/ts-grammar/build.rs | [source](overlay/tokenspeed-engine/crates/ts-grammar/build.rs) |
| tokenspeed-engine/crates/ts-host/build.rs | [source](overlay/tokenspeed-engine/crates/ts-host/build.rs) |
| tokenspeed-engine/crates/ts-host/src/main.rs | [source](overlay/tokenspeed-engine/crates/ts-host/src/main.rs) |
| tokenspeed-engine/crates/ts-metrics/src/prom.rs | [source](overlay/tokenspeed-engine/crates/ts-metrics/src/prom.rs) |
| tokenspeed-engine/crates/ts-scheduler/CMakeLists.txt | [source](overlay/tokenspeed-engine/crates/ts-scheduler/CMakeLists.txt) |
| tokenspeed-engine/crates/ts-scheduler/build.rs | [source](overlay/tokenspeed-engine/crates/ts-scheduler/build.rs) |
| tokenspeed-engine/python/tokenspeed_engine/__init__.py | [source](overlay/tokenspeed-engine/python/tokenspeed_engine/__init__.py) |
| tokenspeed-engine/setup.py | [source](overlay/tokenspeed-engine/setup.py) |
| tokenspeed-kernel/python/requirements/windows-cuda.txt | [source](overlay/tokenspeed-kernel/python/requirements/windows-cuda.txt) |
| tokenspeed-kernel/python/setup.py | [source](overlay/tokenspeed-kernel/python/setup.py) |
| tokenspeed-kernel/python/tokenspeed_kernel/_triton.py | [source](overlay/tokenspeed-kernel/python/tokenspeed_kernel/_triton.py) |
| tokenspeed-kernel/python/tokenspeed_kernel/ops/activation/__init__.py | [source](overlay/tokenspeed-kernel/python/tokenspeed_kernel/ops/activation/__init__.py) |
| tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv4/cuda.py | [source](overlay/tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv4/cuda.py) |
| tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/gdn/flashinfer.py | [source](overlay/tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/gdn/flashinfer.py) |
| tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/mha/cuda.py | [source](overlay/tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/mha/cuda.py) |
| tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/mla/cuda.py | [source](overlay/tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/mla/cuda.py) |
| tokenspeed-kernel/python/tokenspeed_kernel/ops/communication/deep_ep.py | [source](overlay/tokenspeed-kernel/python/tokenspeed_kernel/ops/communication/deep_ep.py) |
| tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/README.md | [source](overlay/tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/README.md) |
| tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/__init__.py | [source](overlay/tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/__init__.py) |
| tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/_cute_dsl/nvfp4_gemm_swiglu.py | [source](overlay/tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/_cute_dsl/nvfp4_gemm_swiglu.py) |
| tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/_flashinfer/fp8_quant.py | [source](overlay/tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/_flashinfer/fp8_quant.py) |
| tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/_triton/nvfp4.py | [source](overlay/tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/_triton/nvfp4.py) |
| tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/cublaslt.py | [source](overlay/tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/cublaslt.py) |
| tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/triton.py | [source](overlay/tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/triton.py) |
| tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/trtllm.py | [source](overlay/tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/trtllm.py) |
| tokenspeed-kernel/python/tokenspeed_kernel/ops/layernorm/__init__.py | [source](overlay/tokenspeed-kernel/python/tokenspeed_kernel/ops/layernorm/__init__.py) |
| tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/__init__.py | [source](overlay/tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/__init__.py) |
| tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/deep_gemm.py | [source](overlay/tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/deep_gemm.py) |
| tokenspeed-kernel/python/tokenspeed_kernel/ops/quantization/triton.py | [source](overlay/tokenspeed-kernel/python/tokenspeed_kernel/ops/quantization/triton.py) |
| tokenspeed-kernel/python/tokenspeed_kernel/ops/quantization/trtllm.py | [source](overlay/tokenspeed-kernel/python/tokenspeed_kernel/ops/quantization/trtllm.py) |
| tokenspeed-kernel/python/tokenspeed_kernel/ops/transform/faster_hadamard_transform/__init__.py | [source](overlay/tokenspeed-kernel/python/tokenspeed_kernel/ops/transform/faster_hadamard_transform/__init__.py) |
| tokenspeed-kernel/python/tokenspeed_kernel/ops/tuning.py | [source](overlay/tokenspeed-kernel/python/tokenspeed_kernel/ops/tuning.py) |
| tokenspeed-kernel/python/tokenspeed_kernel/platform.py | [source](overlay/tokenspeed-kernel/python/tokenspeed_kernel/platform.py) |
| tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cuda/csrc/cublas_fp8_tensor.cu | [source](overlay/tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cuda/csrc/cublas_fp8_tensor.cu) |
| tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cuda/csrc/include/flashinfer/math.cuh | [source](overlay/tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cuda/csrc/include/flashinfer/math.cuh) |
| tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cuda/csrc/include/flashinfer/vec_dtypes.cuh | [source](overlay/tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cuda/csrc/include/flashinfer/vec_dtypes.cuh) |
| tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cuda/cublas_fp8.py | [source](overlay/tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cuda/cublas_fp8.py) |
| tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cuda/cuda_ipc.py | [source](overlay/tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cuda/cuda_ipc.py) |
| tokenspeed-kernel/test/nvidia/ops/test_cublas_fp8_tensor.py | [source](overlay/tokenspeed-kernel/test/nvidia/ops/test_cublas_fp8_tensor.py) |
| tokenspeed-kernel/test/nvidia/ops/test_nvfp4_linear.py | [source](overlay/tokenspeed-kernel/test/nvidia/ops/test_nvfp4_linear.py) |
| tokenspeed-scheduler/CMakeLists.txt | [source](overlay/tokenspeed-scheduler/CMakeLists.txt) |
| tokenspeed-scheduler/csrc/cache/prefix/prefix_hasher.h | [source](overlay/tokenspeed-scheduler/csrc/cache/prefix/prefix_hasher.h) |
| tokenspeed-scheduler/csrc/cache/prefix/prefix_index.h | [source](overlay/tokenspeed-scheduler/csrc/cache/prefix/prefix_index.h) |
| tokenspeed-scheduler/csrc/resource/allocator/req_pool_allocator.cpp | [source](overlay/tokenspeed-scheduler/csrc/resource/allocator/req_pool_allocator.cpp) |
| tokenspeed-scheduler/csrc/utils.h | [source](overlay/tokenspeed-scheduler/csrc/utils.h) |
