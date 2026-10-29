# GEMM weight preparation

## Small dense projections

`mv` selects the registered small-row projection implementation. Its dispatcher
and registry both use the `("gemm", "mv")` operation. The gfx950 row-CTA GEMV
declares its minimum N and K through the shared shape traits.

On CDNA4, ordinary GEMM dispatch admits the measured single-row
`[1,8192] @ [4608,8192].T` projection alongside the existing multirow cases.
Callers that publish directly into a collective buffer pass that buffer as
`out` to the same `mv` operation.

## Per-token/per-channel FP8

Per-channel FP8 projections use FP8 E4M3 weights with FP32 scales per output
channel and dynamically quantize activations per token. This scale format is
distinct from block scaling. Native FP8 matrix arithmetic uses FP32
accumulators but need not match sequential FP32 arithmetic.

The portable token quantizer has an unbounded-row SM100-family registration,
so loading FP8 linears does not require TRT-LLM. The measured small-row
quantizer keeps its higher priority; a present TRT-LLM implementation still
takes larger batches before the portable entry.

`cublaslt_mm_fp8_tensor_scaled` is a native cuBLASLt implementation for
one FP32 device scale per E4M3 operand, with FP16/BF16 output. Its project
FFI library is compiled against the CUDA SDK, including native Windows ARM64;
it does not invoke FlashInfer JIT. FP32 accumulation explicitly disables
fast accumulation. Inputs use row-major A and column-major B. Output buffers
may have a padded row stride with a contiguous last dimension; those buffers
receive a copy from a contiguous result. Fixed-size scratch is isolated by
device and CUDA stream. The registry admits N divisible by 8 and K divisible
by 16, within SM100 through SM121. Real GPU validation was performed on SM121.

## SM12x NVFP4 linears

`nvfp4_linear` takes dense FP16/BF16 activations, packed E2M1 weights
`[N,K/2]`, linear E4M3 scales `[N,K/16]`, and the two scalar global
multipliers stored by compressed-tensors. It quantizes activations per
16-value group, accumulates with `tl.dot_scaled`, and divides by the product
of the global multipliers before converting to the output dtype. The SM12x
implementation emits native `mma.sync ... mxf4nvf4 ... ue4m3` instructions.
Weights remain packed; it does not materialize a BF16 weight matrix.

The runtime scheme invokes the same operation for each logical projection
in a fused layer, preserving unequal input and weight global scales rather
than replacing them by a shared maximum. Request row counts are runtime
parameters and opt out of Triton's integer specialization; changing a decode
batch does not compile another GPU program. This registration targets NVIDIA
SM120/SM121. FlashMLA and FlashInfer's Linux JIT are not prerequisites for it.

## DeepSeek V4 grouped output projection

The grouped-projection plan describes the projection geometry and source formats.

After loading and sharding, the operation selects preprocessing using the source
formats and operation traits. The layer runs that preprocessing through the
weight broker, which installs the transformed tensors and records their storage
layout. When no transformation is needed, the layer enrolls canonical storage.

Execution and warmup select kernels using the supplied tensors' recorded layouts,
formats, and operation traits. Participating layers require known storage layouts;
raw canonical callers can explicitly allow unknown storage. This policy is
independent of kernel overrides. Python validation runs during graph capture,
not device-side replay.

## Hopper block-FP8 DeepGEMM

`deepgemm_mm_fp8_blockscale` supports explicit `mm(..., override=...)` or
`solution="deepgemm"` selection. Serving can opt in with
`--dense-gemm-backend deepgemm` for standard block-FP8 dense linears. The
default `auto` route remains unchanged; routed experts use their own backend.

This optional DeepGEMM integration requires Hopper, contiguous E4M3 operands
A[M,K]/B[N,K], BF16 output, N divisible by 64, and K divisible by 128. It
converts canonical FP32 1x128 activation and 128x128 weight scales to the
required layout without UE8M0 rounding. Conversion is part of each call,
including graph capture; no persistent weight cache is created. Unsupported
forced calls fail explicitly. It does not claim batch-invariant numerics.
