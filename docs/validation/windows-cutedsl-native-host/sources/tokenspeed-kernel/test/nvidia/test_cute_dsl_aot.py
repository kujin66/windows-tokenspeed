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

"""Validate configured AOT selection through the real TokenSpeed family API."""

import json
import os
import shutil
from pathlib import Path

import pytest
import torch
from tokenspeed_kernel.cute_dsl_aot import RouterAOTCatalog, configured_router_aot
from tokenspeed_kernel.ops.gemm import kimi3_router_projection
from tokenspeed_kernel.ops.gemm.cute_dsl_aot import cute_dsl_aot_router_supported
from tokenspeed_kernel.platform import pdl_enabled


@pytest.fixture
def catalog():
    config = os.environ.get("TOKENSPEED_CUTE_DSL_AOT_CONFIG")
    if config is None or os.name != "nt" or not torch.cuda.is_available():
        pytest.skip("Requires configured Windows AOT artifacts and CUDA")
    torch.cuda.set_device(0)
    old_pdl = pdl_enabled()
    pdl_enabled(False)
    configured_router_aot.cache_clear()
    bank = configured_router_aot()
    yield bank
    torch.cuda.synchronize()
    pdl_enabled(old_pdl)
    # Keep loaded modules alive: graphs may own their device functions.


@pytest.mark.parametrize("m", [1, 2, 4])
def test_auto_router_graph_and_changed_inputs(catalog, tmp_path, m):
    torch.manual_seed(178 + m)
    a = torch.randn(m, 7168, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(896, 7168, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(m, 896, dtype=torch.float32, device="cuda")
    assert cute_dsl_aot_router_supported(a, b)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        assert kimi3_router_projection(a, b, out=out, solution="auto") is out
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    reference = (a.cpu().double() @ b.cpu().double().T).float()
    torch.testing.assert_close(out.cpu(), reference, atol=1e-3, rtol=1e-4)
    graph = torch.cuda.CUDAGraph()
    graph.enable_debug_mode()
    with torch.cuda.graph(graph):
        kimi3_router_projection(a, b, out=out, solution="auto")
    for _ in range(3):
        a.normal_()
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        reference = (a.cpu().double() @ b.cpu().double().T).float()
        torch.testing.assert_close(out.cpu(), reference, atol=1e-3, rtol=1e-4)
    dot = tmp_path / f"auto-m{m}.dot"
    graph.debug_dump(str(dot))
    captured = dot.read_text(encoding="utf-8")
    kernel = catalog.modules[(0, (m, 896, 7168))]
    assert kernel.metadata["entry"] in captured
    assert captured.count('label="{KERNEL') == 1
    pdl_enabled(True)
    assert not cute_dsl_aot_router_supported(a, b)
    with pytest.raises(ValueError):
        kimi3_router_projection(a, b, out=out, solution="cute_dsl_aot")


def test_uncovered_shape_falls_back_and_forced_aot_fails(catalog):
    a = torch.randn(5, 7168, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(896, 7168, dtype=torch.bfloat16, device="cuda")
    assert not catalog.supports(a, b)
    with pytest.raises((ValueError, RuntimeError)):
        kimi3_router_projection(a, b, solution="cute_dsl_aot")
    # The existing auto path uses cuBLAS at M>4; an uncovered shape still runs.
    actual = kimi3_router_projection(a, b, solution="auto")
    reference = torch.mm(a, b.T, out_dtype=torch.float32)
    torch.testing.assert_close(actual, reference, atol=1e-3, rtol=1e-4)


def test_source_mismatch_and_duplicate_policy_rejected(catalog, tmp_path):
    manifest = json.loads((catalog.directory / "manifest.json").read_text())
    manifest["source_sha256"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    config = {
        "schema": 1,
        "directory": ".",
        "selections": [{"shape_mnk": [1, 896, 7168], "block_size": 128}],
    }
    policy = tmp_path / "selection.json"
    policy.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="source"):
        RouterAOTCatalog(policy)
    shutil.copyfile(catalog.directory / "manifest.json", tmp_path / "manifest.json")
    config["selections"] *= 2
    policy.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="duplicate"):
        RouterAOTCatalog(policy)


def test_corrupt_cubin_rejected_before_launch(catalog, tmp_path):
    manifest = json.loads((catalog.directory / "manifest.json").read_text())
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    shape, block = next(iter(catalog.blocks.items()))
    selected = next(
        k
        for k in manifest["kernels"]
        if tuple(k["shape_mnk"]) == shape and k["block"] == [block, 1, 1]
    )
    (tmp_path / (selected["name"] + ".cubin")).write_bytes(b"corrupt")
    policy = tmp_path / "selection.json"
    policy.write_text(
        json.dumps(
            {
                "schema": 1,
                "directory": ".",
                "selections": [{"shape_mnk": list(shape), "block_size": block}],
            }
        )
    )
    bank = RouterAOTCatalog(policy)
    with pytest.raises(ValueError, match="hash"):
        bank.preload(device=torch.device("cuda:0"))


def test_unwarmed_catalog_refuses_module_load_inside_capture(catalog):
    bank = RouterAOTCatalog(Path(os.environ["TOKENSPEED_CUTE_DSL_AOT_CONFIG"]))
    a = torch.randn(1, 7168, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(896, 7168, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(1, 896, dtype=torch.float32, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with pytest.raises(RuntimeError, match="Warm up"):
            bank.run(a, b, out=out)
        out.zero_()
    graph.replay()
    torch.testing.assert_close(out, torch.zeros_like(out), atol=0, rtol=0)
    assert not bank.modules


def test_absent_configuration_preserves_fallback(catalog, monkeypatch):
    with monkeypatch.context() as environment:
        environment.delenv("TOKENSPEED_CUTE_DSL_AOT_CONFIG")
        configured_router_aot.cache_clear()
        assert configured_router_aot() is None
        a = torch.randn(5, 7168, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(896, 7168, dtype=torch.bfloat16, device="cuda")
        assert not cute_dsl_aot_router_supported(a, b)
        result = kimi3_router_projection(a, b, solution="auto")
        torch.testing.assert_close(result, torch.mm(a, b.T, out_dtype=torch.float32))
    configured_router_aot.cache_clear()
