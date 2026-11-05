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

"""Exercise the selected dense backend under changed-input CUDA graph replay."""

import pytest
import torch
from tokenspeed_kernel.ops.activation import silu_and_mul
from tokenspeed_kernel.ops.layernorm import rmsnorm


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("residual_enabled", [False, True])
def test_selected_norm_graph(batch, residual_enabled):
    torch.manual_seed(20261102)
    x = torch.randn(batch, 1024, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x) if residual_enabled else None
    weight = torch.randn(1024, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        rmsnorm(x, weight, 1e-6, residual=residual)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = rmsnorm(x, weight, 1e-6, residual=residual)
    for seed in range(3):
        torch.manual_seed(seed + 19)
        x.normal_()
        if residual is not None:
            residual.normal_()
        summed = x.float() + residual.float() if residual is not None else x.float()
        expected = (
            summed
            * torch.rsqrt(summed.square().mean(-1, keepdim=True) + 1e-6)
            * weight.float()
        ).to(x.dtype)
        graph.replay()
        actual = result[0] if isinstance(result, tuple) else result
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.02)
        if residual is not None:
            torch.testing.assert_close(result[1], summed.to(x.dtype), rtol=0, atol=0)


@pytest.mark.parametrize("batch", [1, 4])
def test_selected_swiglu_graph(batch):
    x = torch.randn(batch, 2048, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        silu_and_mul(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = silu_and_mul(x)
    for _ in range(3):
        x.normal_()
        graph.replay()
        expected = (
            torch.nn.functional.silu(x[:, :1024].float()) * x[:, 1024:].float()
        ).to(x.dtype)
        torch.testing.assert_close(out, expected, rtol=0.01, atol=0.02)


def _paged_inputs(batch, length):
    q_heads, kv_heads, head_dim, page_size = 16, 8, 128, 64
    k = torch.randn(
        batch, length, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    v = torch.randn_like(k)
    k_cache = k.reshape(batch * length // page_size, page_size, kv_heads, head_dim)
    v_cache = v.reshape_as(k_cache)
    table = torch.arange(
        batch * length // page_size, device="cuda", dtype=torch.int32
    ).reshape(batch, -1)
    lengths = torch.full((batch,), length, device="cuda", dtype=torch.int32)
    return k, v, k_cache, v_cache, table, lengths


def _attention_reference(q, k, v, *, causal):
    key = k.repeat_interleave(q.shape[2] // k.shape[2], dim=2)
    value = v.repeat_interleave(q.shape[2] // v.shape[2], dim=2)
    scores = q.float().transpose(1, 2) @ key.float().transpose(1, 2).transpose(-1, -2)
    scores *= q.shape[-1] ** -0.5
    if causal:
        mask = torch.ones(
            q.shape[1], k.shape[1], device=q.device, dtype=torch.bool
        ).triu(1)
        scores.masked_fill_(mask, -torch.inf)
    return (
        (scores.softmax(-1) @ value.float().transpose(1, 2)).transpose(1, 2).to(q.dtype)
    )


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("length", [128, 512])
def test_sm121_fa2_prefill(batch, length):
    from tokenspeed_kernel.ops.attention.mha._sm121_fa2 import (
        flashinfer_fa2_mha_extend_sm121,
    )

    q = torch.randn(batch, length, 16, 128, device="cuda", dtype=torch.bfloat16)
    k, v, kc, vc, table, lengths = _paged_inputs(batch, length)
    cu = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * length
    result = flashinfer_fa2_mha_extend_sm121(
        q=q.reshape(-1, 16, 128),
        k_cache=kc,
        v_cache=vc,
        page_table=table,
        cache_seqlens=lengths,
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        max_seqlen_q=length,
        max_seqlen_k=length,
        is_causal=True,
        window_left=-1,
        logit_cap=0,
        sinks=None,
        return_lse=False,
        softmax_scale=128**-0.5,
        q_scale=None,
        k_scale=None,
        v_scale=None,
        enable_pdl=False,
    )
    expected = _attention_reference(q, k, v, causal=True).reshape_as(result)
    torch.testing.assert_close(result, expected, rtol=0.01, atol=0.01)


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("length", [128, 512])
def test_sm121_decode_graph(batch, length):
    from tokenspeed_kernel.ops.attention.mha.flashinfer import (
        flashinfer_trtllm_mha_decode_with_kvcache,
    )

    q = torch.randn(batch, 16, 128, device="cuda", dtype=torch.bfloat16)
    k, v, kc, vc, table, lengths = _paged_inputs(batch, length)

    def decode():
        return flashinfer_trtllm_mha_decode_with_kvcache(
            q,
            kc,
            vc,
            table,
            lengths,
            length,
            enable_pdl=False,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        decode()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = decode()
    for _ in range(3):
        q.normal_()
        graph.replay()
        expected = _attention_reference(q[:, None], k, v, causal=False).reshape_as(out)
        torch.testing.assert_close(out, expected, rtol=0.01, atol=0.01)
