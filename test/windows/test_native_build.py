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

"""Installed-wheel and CUDA graph checks for the minimal ARM64 build."""

import hashlib
import importlib.metadata
import struct
import sys
import zipfile
from pathlib import Path

import pytest

from tools.windows.build_wheels import audit_wheel, pe_machine


def load_cuda_library(group):
    import tvm_ffi

    distribution = importlib.metadata.distribution("tokenspeed-kernel")
    path = Path(
        distribution.locate_file(
            f"tokenspeed_kernel/thirdparty/cuda/objs/{group}/{group}.so"
        )
    ).resolve()
    assert path.is_relative_to(Path(sys.prefix).resolve())
    return tvm_ffi.load_module(str(path))


def test_rejects_non_pe():
    with pytest.raises(ValueError):
        pe_machine(b"not a PE file")


@pytest.mark.parametrize("machine", [0xAA64, 0x8664])
def test_wheel_machine_validation(tmp_path, machine):
    data = bytearray(128)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 60, 64)
    data[64:68] = b"PE\0\0"
    struct.pack_into("<H", data, 68, machine)
    path = tmp_path / "example-1-cp313-cp313-win_arm64.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("example/native.so", data)
    if machine == 0xAA64:
        assert audit_wheel(path)["native_files"] == ["example/native.so"]
    else:
        with pytest.raises(ValueError, match="Not ARM64"):
            audit_wheel(path)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows installed-wheel check")
def test_installed_native_packages():
    import tokenspeed_engine
    import tokenspeed_scheduler

    prefix = Path(sys.prefix).resolve()
    for module in (tokenspeed_engine, tokenspeed_scheduler):
        assert Path(module.__file__).resolve().is_relative_to(prefix)
    host = tokenspeed_engine.host_binary()
    assert host.is_file()
    assert pe_machine(host.read_bytes()) == 0xAA64
    assert importlib.metadata.version("tokenspeed")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows CNG SHA256 check")
def test_prefix_hash_matches_sha256():
    import tokenspeed_scheduler as ts

    config = ts.SchedulerConfig()
    config.prefix_granularity = 2
    config.num_device_pages = 8
    config.num_host_pages = 0
    config.num_snapshot_pages = 1
    config.max_scheduled_tokens = 16
    config.max_batch_size = 4
    config.disable_l2_cache = True
    config.cache_groups = [
        ts.CacheGroupConfig(
            group_id="full_attention",
            block_granularity=2,
            total_pages=8,
            attn_kind=ts.AttnKind.Full,
        )
    ]
    scheduler = ts.Scheduler(config)
    tokens = [0, 2147483647, 42, 256, 7]
    expected, prior = [], b""
    for offset in (0, 2):
        payload = (
            struct.pack("<I", len(prior))
            + prior
            + struct.pack("<I2i", 2, *tokens[offset : offset + 2])
        )
        prior = hashlib.sha256(payload).digest()
        expected.append(prior.hex())
    assert scheduler.prefix_hashes_for_tokens(tokens) == expected


@pytest.mark.skipif(sys.platform != "win32", reason="Windows ARM64 CUDA build check")
@pytest.mark.parametrize("batch", [1, 4])
def test_rmsnorm_cuda_graph(batch):
    import torch

    rmsnorm_fused_parallel = load_cuda_library(
        "rmsnorm_fused_parallel"
    ).rmsnorm_fused_parallel

    # A configured GPU runner must fail, rather than silently skip, without CUDA.
    assert torch.cuda.is_available()
    x = torch.randn(batch, 1024, device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(1024, device="cuda", dtype=torch.bfloat16)
    out1, out2 = torch.empty_like(x), torch.empty_like(x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        rmsnorm_fused_parallel(x, weight, out1, x, weight, out2, 1e-6, False)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rmsnorm_fused_parallel(x, weight, out1, x, weight, out2, 1e-6, False)
    for _ in range(3):
        x.normal_()
        graph.replay()
        expected = (
            x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        ).to(x.dtype)
        torch.testing.assert_close(out1, expected, rtol=0.01, atol=0.01)
        torch.testing.assert_close(out2, expected, rtol=0.01, atol=0.01)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows ARM64 CUDA build check")
def test_chain_verifier_cuda_graph():
    import torch

    verify_chain_greedy = load_cuda_library("sampling_chain").verify_chain_greedy

    assert torch.cuda.is_available()
    batch, width = 4, 4
    target = (torch.arange(batch * width, dtype=torch.int64, device="cuda") + 100).view(
        batch, width
    )
    candidates = torch.full((batch, width), -1, dtype=torch.int32, device="cuda")
    predicts = torch.empty(batch * width, dtype=torch.int32, device="cuda")
    indices = torch.empty_like(candidates)
    count = torch.empty(batch, dtype=torch.int32, device="cuda")

    def verify():
        predicts.fill_(-1)
        indices.fill_(-1)
        verify_chain_greedy(
            predicts, indices, count, candidates, target, batch, width, False
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        verify()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        verify()
    for accepted in (3, 0, 1):
        candidates.fill_(-1)
        candidates[:, 1 : accepted + 1].copy_(target[:, :accepted])
        graph.replay()
        assert count.cpu().tolist() == [accepted] * batch
        for row in range(batch):
            expected_indices = list(range(row * width, row * width + accepted + 1)) + [
                -1
            ] * (width - accepted - 1)
            assert indices[row].cpu().tolist() == expected_indices
            expected_predicts = list(
                range(100 + row * width, 100 + row * width + accepted + 1)
            ) + [-1] * (width - accepted - 1)
            assert predicts.view(batch, width)[row].cpu().tolist() == expected_predicts
