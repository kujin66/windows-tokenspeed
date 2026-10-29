# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""NVFP4 checkpoint parsing, packed GPU arithmetic and fused-scale preservation."""

import pytest
import torch
from tokenspeed_kernel.ops.gemm import nvfp4_linear
from tokenspeed_kernel.ops.gemm._triton.nvfp4 import quantize_nvfp4_linear

from tokenspeed.runtime.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from tokenspeed.runtime.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_nvfp4 import (
    CompressedTensorsW4A4Nvfp4,
)
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader


def config(group_size):
    weights = dict(
        type="float",
        num_bits=4,
        group_size=group_size,
        strategy="tensor_group",
        symmetric=True,
        dynamic=False,
    )
    return CompressedTensorsConfig.from_config(
        dict(
            format="nvfp4-pack-quantized",
            ignore=["lm_head"],
            config_groups={
                "group_0": dict(
                    targets=["Linear"],
                    weights=weights,
                    input_activations={**weights, "dynamic": "local"},
                )
            },
        )
    )


def test_nvfp4_checkpoint_scheme():
    cfg = config(16)
    group = cfg.target_scheme_map["Linear"]
    scheme = cfg._get_scheme_from_parts(group["weights"], group["input_activations"])
    assert isinstance(scheme, CompressedTensorsW4A4Nvfp4)


def test_invalid_nvfp4_group_size_is_rejected():
    cfg = config(32)
    group = cfg.target_scheme_map["Linear"]
    with pytest.raises(NotImplementedError, match="group_size=16"):
        cfg._get_scheme_from_parts(group["weights"], group["input_activations"])


def test_missing_nvfp4_global_scales_are_rejected():
    layer = torch.nn.Module()
    scheme = CompressedTensorsW4A4Nvfp4()
    scheme.create_weights(
        layer, 128, 128, [64], 64, torch.bfloat16, default_weight_loader
    )
    with pytest.raises(ValueError, match="loaded, finite and positive"):
        scheme.process_weights_after_loading(layer)


def reference_quantize(x, inverse):
    blocks = x.float().reshape(x.shape[0], -1, 16)
    scales = (
        (blocks.abs().amax(-1) * inverse / 6).clamp(max=448).to(torch.float8_e4m3fn)
    )
    scaled = (
        blocks
        * torch.where(scales.float() > 0, inverse / scales.float(), 0)[:, :, None]
    )
    levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=x.device)
    distance = (scaled.abs()[..., None] - levels).abs()
    # Round ties to the even E2M1 code, independently of the GPU conversion.
    codes = torch.arange(8, device=x.device)
    score = torch.where(
        distance == distance.amin(-1, keepdim=True), codes + (codes % 2) * 16, 100
    )
    index = score.argmin(-1)
    signed = index | (torch.signbit(scaled).long() * 8)
    signed = signed.reshape(x.shape[0], x.shape[1])
    packed = (signed[:, ::2] | (signed[:, 1::2] << 4)).to(torch.uint8)
    decoded = levels[index] * torch.where(torch.signbit(scaled), -1.0, 1.0)
    decoded *= scales.float()[:, :, None] / inverse
    return packed, scales, decoded.reshape_as(x)


def decode_weight(packed, scales, inverse):
    levels = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        device=packed.device,
    )
    codes = torch.stack([packed & 15, packed >> 4], dim=-1).flatten(-2)
    return levels[codes.long()] * scales.float().repeat_interleave(16, 1) / inverse


requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12,
    reason="requires SM12x NVFP4 Tensor Cores",
)


@requires_gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch", [1, 7, 32])
@pytest.mark.parametrize("shape", [(17, 16), (64, 128), (65, 160)])
def test_nvfp4_gpu_against_decoded_oracle(dtype, batch, shape):
    torch.manual_seed(42)
    n, k = shape
    x = torch.randn(batch, k, device="cuda", dtype=dtype)
    wp = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
    ws = torch.randint(1, 20, (n, k // 16), device="cuda").to(torch.float8_e4m3fn)
    gx, gw = torch.tensor([400.0], device="cuda"), torch.tensor([73.0], device="cuda")
    expected_packed, expected_scales, xd = reference_quantize(x, gx)
    packed, scales = quantize_nvfp4_linear(x, gx)
    torch.testing.assert_close(packed, expected_packed, atol=0, rtol=0)
    torch.testing.assert_close(scales.float(), expected_scales.float(), atol=0, rtol=0)
    actual = nvfp4_linear(x, wp, ws, gx, gw, bias=None, solution="triton")
    reference = (xd @ decode_weight(wp, ws, gw).T).to(dtype)
    torch.testing.assert_close(
        actual, reference, atol=2e-3, rtol=8e-3 if dtype == torch.bfloat16 else 1e-3
    )


@requires_gpu
@pytest.mark.parametrize("with_bias", [False, True])
def test_fused_nvfp4_keeps_each_projection_global_scale(with_bias):
    torch.manual_seed(42)
    scheme = CompressedTensorsW4A4Nvfp4()
    layer = torch.nn.Module()
    scheme.create_weights(
        layer, 128, 128, [32, 64], 96, torch.bfloat16, default_weight_loader
    )
    layer = layer.cuda()
    layer.weight_packed.data.random_(0, 256)
    layer.weight_scale.data.copy_(torch.ones(96, 8, device="cuda"))
    layer.input_global_scale.data.copy_(torch.tensor([100.0, 500.0], device="cuda"))
    layer.weight_global_scale.data.copy_(torch.tensor([23.0, 71.0], device="cuda"))
    scheme.process_weights_after_loading(layer)
    x = torch.randn(7, 128, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(96, device="cuda", dtype=x.dtype) if with_bias else None
    actual = scheme.apply_weights(layer, x, bias)
    expected = []
    for i, (start, end) in enumerate([(0, 32), (32, 96)]):
        _, _, xd = reference_quantize(x, layer.input_global_scale[i])
        wd = decode_weight(
            layer.weight_packed[start:end],
            layer.weight_scale[start:end],
            layer.weight_global_scale[i],
        )
        value = (xd @ wd.T).to(x.dtype)
        if bias is not None:
            value += bias[start:end]
        expected.append(value)
    torch.testing.assert_close(actual, torch.cat(expected, -1), atol=2e-3, rtol=8e-3)


@requires_gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_nvfp4_zero_blocks_and_round_to_even(dtype):
    gain = torch.ones(1, device="cuda")
    values = torch.tensor(
        [
            [
                0.0,
                0.25,
                0.5,
                0.75,
                1.0,
                1.25,
                1.5,
                1.75,
                2.0,
                2.5,
                3.0,
                3.5,
                4.0,
                5.0,
                6.0,
                -0.75,
            ]
        ],
        device="cuda",
        dtype=dtype,
    )
    values = torch.cat([values, torch.zeros_like(values)])
    packed, scales = quantize_nvfp4_linear(values, gain)
    expected_packed, expected_scales, _ = reference_quantize(values, gain)
    torch.testing.assert_close(packed, expected_packed, atol=0, rtol=0)
    torch.testing.assert_close(scales.float(), expected_scales.float(), atol=0, rtol=0)
    output = nvfp4_linear(
        torch.zeros_like(values),
        packed,
        scales,
        gain,
        gain,
        bias=None,
        solution="triton",
    )
    torch.testing.assert_close(output, torch.zeros_like(output), atol=0, rtol=0)
