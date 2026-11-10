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

import pytest
import torch
from tokenspeed_kernel.thirdparty.cuda.marlin import gptq_marlin_repack
from tokenspeed_kernel.thirdparty.cuda.marlin_moe import (
    marlin_make_workspace,
    marlin_permute_scales,
    moe_align_block_size,
    moe_wna16_marlin_gemm,
    mxfp4_marlin_process_scales,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.cuda is None,
    reason="NVIDIA CUDA required",
)


@pytest.mark.parametrize(
    "tokens,block", [(1, 8), (8, 16), (64, 32), (96, 48), (128, 64)]
)
@pytest.mark.parametrize("is_ep", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_native_mxfp4_routes_match_independent_dequantization(
    tokens, block, is_ep, weighted
):
    torch.manual_seed(29)
    torch.set_float32_matmul_precision("highest")
    experts, size_k, size_n, routes = 4, 256, 128, 2
    codes = torch.randint(
        0, 16, (experts, size_k, size_n), device="cuda", dtype=torch.int64
    )
    scale_bytes = torch.randint(
        124, 128, (experts, size_k // 32, size_n), device="cuda", dtype=torch.uint8
    )
    shifts = (torch.arange(8, device="cuda", dtype=torch.int64) * 4).view(1, 1, 8, 1)
    packed = (
        (codes.reshape(experts, size_k // 8, 8, size_n) << shifts)
        .sum(2)
        .to(torch.int32)
    )
    empty_perm = torch.empty(0, device="cuda", dtype=torch.int32)
    repacked = torch.stack(
        [
            gptq_marlin_repack(packed[e], empty_perm, size_k, size_n, 4)
            for e in range(experts)
        ]
    )
    scales = torch.stack(
        [
            mxfp4_marlin_process_scales(
                marlin_permute_scales(scale_bytes[e], size_k, size_n, 32)
            )
            for e in range(experts)
        ]
    )
    ids = (
        (torch.arange(tokens * routes, device="cuda") % experts)
        .reshape(tokens, routes)
        .to(torch.int32)
    )
    if is_ep:
        ids.view(-1)[::3] = -1
    sorted_ids, expert_ids, padded = moe_align_block_size(ids, block, experts)
    # GEMM2 reads one activation row per route; GEMM1 broadcasts each token.
    rows = tokens * routes if weighted else tokens
    top_k = 1 if weighted else routes
    activation = (torch.randn(rows, size_k, device="cuda") * 0.05).to(torch.bfloat16)
    # Exact powers of two keep the independent reference free of double rounding.
    route_weights = torch.exp2(
        -(torch.arange(tokens * routes, device="cuda") % 4).float()
    ).reshape(tokens, routes)
    actual = moe_wna16_marlin_gemm(
        activation,
        None,
        repacked,
        scales,
        marlin_make_workspace(activation.device),
        sorted_ids,
        expert_ids,
        padded,
        route_weights,
        moe_block_size=block,
        top_k=top_k,
        mul_topk_weights=weighted,
        is_ep=is_ep,
        size_m=rows,
        size_n=size_n,
        size_k=size_k,
        use_fp32_reduce=True,
    )
    magnitudes = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device="cuda")
    weights = magnitudes[codes & 7] * torch.where((codes & 8) == 0, 1.0, -1.0)
    weights *= torch.exp2(scale_bytes.float() - 127).repeat_interleave(32, dim=1)
    expected = torch.zeros(tokens * routes, size_n, device="cuda")
    for expert in range(experts):
        selected = (ids.reshape(-1) == expert).nonzero().flatten()
        input_rows = selected if weighted else selected // routes
        result = activation.float().index_select(0, input_rows) @ weights[expert]
        if weighted:
            result *= route_weights.reshape(-1).index_select(0, selected)[:, None]
        expected.index_copy_(0, selected, result)
    torch.testing.assert_close(
        actual.float(), expected.to(torch.bfloat16).float(), rtol=1e-2, atol=2e-3
    )
