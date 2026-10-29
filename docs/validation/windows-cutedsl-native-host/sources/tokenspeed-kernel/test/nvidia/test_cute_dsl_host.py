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

"""Real Windows ARM64 execution of cross-compiled CuTe host programs."""

import json
import os
import shutil
import sys
from pathlib import Path

import pytest
import torch
from tokenspeed_kernel.cute_dsl_aot import (
    RouterHostCatalog,
    configured_router_aot,
    prepare_router_host,
)
from tokenspeed_kernel.cute_dsl_compile import pe_machine
from tokenspeed_kernel.cute_dsl_host import (
    CuTeHostProgram,
    build_host_artifact,
    compile_host_program,
)
from tokenspeed_kernel.ops.gemm import kimi3_router_projection
from tokenspeed_kernel.platform import pdl_enabled


@pytest.fixture(scope="module")
def build_options(tmp_path_factory):
    compiler = os.environ.get("TOKENSPEED_CUTE_DSL_COMPILER_PYTHON")
    if (
        os.name != "nt"
        or not torch.cuda.is_available()
        or compiler is None
        or shutil.which("cl") is None
    ):
        pytest.skip(
            "Requires configured Windows CuTe compiler, GPU and native MSVC environment"
        )
    if pe_machine(Path(sys.executable)) != 0xAA64:
        pytest.skip("Requires native ARM64 Python")
    torch.cuda.set_device(0)
    root = Path(__file__).resolve().parents[3]
    return dict(
        compiler_python=Path(compiler),
        arch="sm_121a",
        cache=tmp_path_factory.mktemp("host-cache"),
        dependencies=(root / "tools/windows/cute_dsl_aot/elementwise_factory.py",),
        cuda_home=Path(os.environ["CUDA_PATH"]),
        cxx=Path(shutil.which("cl")),
    )


@pytest.fixture(scope="module")
def programs(build_options):
    root = Path(__file__).resolve().parents[3] / "tools/windows/cute_dsl_aot"
    items = {
        "affine": (
            root / "elementwise_factory.py",
            "compile_affine",
            {"block_size": 128},
        ),
        "tensor": (root / "host_factory.py", "compile_tensor", {"divisibility": 1}),
        "tensor_div4": (
            root / "host_factory.py",
            "compile_tensor",
            {"divisibility": 4},
        ),
        "sequence": (root / "host_factory.py", "compile_sequence", {}),
        "cluster": (root / "host_factory.py", "compile_cluster", {"use_pdl": False}),
        "cluster_pdl": (root / "host_factory.py", "compile_cluster", {"use_pdl": True}),
    }
    return {
        name: compile_host_program(
            **build_options,
            source=source,
            factory=factory,
            parameters=parameters,
            device=torch.device("cuda:0"),
        )
        for name, (source, factory, parameters) in items.items()
    }


def test_automatic_scalar_signature_and_nondefault_stream(programs):
    p = programs["affine"]
    for n in (1, 127, 128, 129, 4096):
        a = torch.arange(n * 2, dtype=torch.float32, device="cuda")
        out = torch.empty(n, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            p(a=a, c=out, n=n, stride=2, scale=2.0, offset=1.0)
        torch.cuda.current_stream().wait_stream(stream)
        torch.testing.assert_close(out, a[::2] * 2 + 1, atol=0, rtol=0)
    with pytest.raises(TypeError):
        p(a, out, n, 2, 2.0, 1.0, unexpected=True)
    with pytest.raises(ValueError, match="i32 range"):
        p(a, out, 2**31, 2, 2.0, 1.0)


def test_automatic_dynamic_tensor_descriptors_and_graphs(programs, tmp_path):
    p = programs["tensor"]
    for n in (1, 127, 128, 129, 4096):
        a = torch.arange(n * 2, dtype=torch.float32, device="cuda")[::2]
        out = torch.empty(n, device="cuda")
        p(a, out)
        graph = torch.cuda.CUDAGraph()
        graph.enable_debug_mode()
        with torch.cuda.graph(graph):
            p(a, out)
        for _ in range(3):
            a.add_(1)
            out.fill_(float("nan"))
            graph.replay()
            torch.testing.assert_close(out, a * 2 + 1, atol=0, rtol=0)
        dot = tmp_path / f"tensor-{n}.dot"
        graph.debug_dump(str(dot))
        assert dot.read_text(encoding="utf-8").count('label="{KERNEL') == 1
        del graph


def test_tensor_contract_rejection(programs):
    p = programs["tensor"]
    a = torch.empty(128, device="cuda")
    with pytest.raises(ValueError, match="symbolic tensor dimensions"):
        p(a, torch.empty(129, device="cuda"))
    with pytest.raises(ValueError, match="static shape/stride"):
        p(a, torch.empty(256, device="cuda")[::2])
    with pytest.raises(ValueError, match="dtype"):
        p(a.double(), a)
    with pytest.raises(ValueError, match="alignment"):
        p(torch.empty(129, device="cuda")[1:], a)
    with pytest.raises(ValueError, match="divisibility"):
        programs["tensor_div4"](
            torch.empty(129, device="cuda"), torch.empty(129, device="cuda")
        )
    programs["tensor_div4"](a, torch.empty_like(a))


@pytest.mark.parametrize("repetitions", [0, 1, 3])
def test_preserved_runtime_host_loop_and_launch_order(programs, tmp_path, repetitions):
    p = programs["sequence"]
    a = torch.arange(257, dtype=torch.float32, device="cuda")
    temporary, out = torch.empty_like(a), torch.empty_like(a)
    p(a, temporary, out, a.numel(), repetitions)
    graph = torch.cuda.CUDAGraph()
    graph.enable_debug_mode()
    with torch.cuda.graph(graph):
        p(a, temporary, out, a.numel(), repetitions)
    for _ in range(3):
        a.add_(1)
        graph.replay()
        expected = a
        for _ in range(repetitions + 1):
            expected = expected * 2 + 1
        torch.testing.assert_close(out, expected, atol=0, rtol=0)
    dot = tmp_path / f"sequence-{repetitions}.dot"
    graph.debug_dump(str(dot))
    assert dot.read_text(encoding="utf-8").count('label="{KERNEL') == repetitions + 2


@pytest.mark.parametrize("name", ["cluster", "cluster_pdl"])
def test_preserved_cluster_and_pdl_launch(programs, name):
    p = programs[name]
    for n in (1, 129, 257, 1024):
        a = torch.arange(n, dtype=torch.float32, device="cuda")
        out = torch.empty_like(a)
        p(a, out, n)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            p(a, out, n)
        for _ in range(3):
            a.add_(1)
            graph.replay()
            expected = a + (torch.arange(n, device="cuda") // 128 % 2).float()
            torch.testing.assert_close(out, expected, atol=0, rtol=0)
        del graph


def test_cached_native_link_and_corrupt_dll_rejected(build_options, programs, tmp_path):
    source = (
        Path(__file__).resolve().parents[3]
        / "tools/windows/cute_dsl_aot/elementwise_factory.py"
    )
    options = dict(
        **build_options,
        source=source,
        factory="compile_affine",
        parameters={"block_size": 128},
    )
    directory = build_host_artifact(**options)
    timestamp = (directory / "host.dll").stat().st_mtime_ns
    assert build_host_artifact(**options) == directory
    assert (directory / "host.dll").stat().st_mtime_ns == timestamp
    assert pe_machine(directory / "host.dll") == 0xAA64
    manifest = json.loads((directory / "artifact.json").read_text(encoding="utf-8"))
    for filename in (
        "artifact.json",
        "kernel.cubin",
        "kernel.ptx",
        *manifest["host_files"],
    ):
        shutil.copyfile(directory / filename, tmp_path / filename)
    (tmp_path / "host.dll").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="host file hash"):
        CuTeHostProgram(
            tmp_path,
            device=torch.device("cuda:0"),
            cuda_home=build_options["cuda_home"],
        )


def test_host_preloading_refused_inside_capture(build_options, programs):
    source = (
        Path(__file__).resolve().parents[3]
        / "tools/windows/cute_dsl_aot/elementwise_factory.py"
    )
    options = dict(
        **build_options,
        source=source,
        factory="compile_affine",
        parameters={"block_size": 128},
    )
    out = torch.empty(1, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with pytest.raises(RuntimeError, match="before CUDA graph capture"):
            compile_host_program(**options, device=torch.device("cuda:0"))
        out.zero_()
    graph.replay()


def test_native_tma_descriptor_and_shared_memory_copy(build_options):
    source = (
        Path(__file__).resolve().parents[3]
        / "tools/windows/cute_dsl_aot/host_factory.py"
    )
    p = compile_host_program(
        **build_options,
        source=source,
        factory="compile_tma",
        parameters={},
        device=torch.device("cuda:0"),
    )
    for n in (256, 512, 2048):
        a = torch.randn(n, device="cuda")
        out = torch.empty_like(a)
        p(a, out)
        torch.testing.assert_close(out, a, atol=0, rtol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            p(a, out)
        for _ in range(3):
            a.normal_()
            out.fill_(float("nan"))
            graph.replay()
            torch.testing.assert_close(out, a, atol=0, rtol=0)
        del graph


def test_programmatic_dependent_sequence(build_options, tmp_path):
    source = (
        Path(__file__).resolve().parents[3]
        / "tools/windows/cute_dsl_aot/host_factory.py"
    )
    p = compile_host_program(
        **build_options,
        source=source,
        factory="compile_pdl_sequence",
        parameters={},
        device=torch.device("cuda:0"),
    )
    n = 1024
    a = torch.arange(n, dtype=torch.float32, device="cuda")
    temporary, out = torch.empty_like(a), torch.empty_like(a)
    p(a, temporary, out, n)
    graph = torch.cuda.CUDAGraph()
    graph.enable_debug_mode()
    with torch.cuda.graph(graph):
        p(a, temporary, out, n)
    for _ in range(3):
        a.add_(1)
        graph.replay()
        expected = a + 2 * (torch.arange(n, device="cuda") // 128 % 2).float()
        torch.testing.assert_close(out, expected, atol=0, rtol=0)
    dot = tmp_path / "pdl-sequence.dot"
    graph.debug_dump(str(dot))
    assert dot.read_text(encoding="utf-8").count('label="{KERNEL') == 2


@pytest.fixture(scope="module")
def native_router_policy(build_options, tmp_path_factory):
    policy = tmp_path_factory.mktemp("native-router") / "selection.json"
    return prepare_router_host(
        compiler_python=build_options["compiler_python"],
        cache=build_options["cache"],
        cuda_home=build_options["cuda_home"],
        cxx=build_options["cxx"],
        blocks={1: 128, 2: 128, 4: 64},
        configuration=policy,
    )


@pytest.mark.parametrize("m", [1, 2, 4])
def test_native_router_selected_through_family_api(
    native_router_policy, monkeypatch, tmp_path, m
):
    monkeypatch.setenv("TOKENSPEED_CUTE_DSL_AOT_CONFIG", str(native_router_policy))
    old_pdl = pdl_enabled()
    pdl_enabled(False)
    configured_router_aot.cache_clear()
    try:
        a = torch.randn(m, 7168, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(896, 7168, dtype=torch.bfloat16, device="cuda")
        out = torch.empty(m, 896, device="cuda")
        assert kimi3_router_projection(a, b, out=out, solution="auto") is out
        graph = torch.cuda.CUDAGraph()
        graph.enable_debug_mode()
        with torch.cuda.graph(graph):
            kimi3_router_projection(a, b, out=out, solution="auto")
        for _ in range(3):
            a.normal_()
            out.fill_(float("nan"))
            graph.replay()
            reference = (a.cpu().double() @ b.cpu().double().T).float()
            torch.testing.assert_close(out.cpu(), reference, atol=1e-3, rtol=1e-4)
        bank = configured_router_aot()
        program = bank.modules[(0, (m, 896, 7168))]
        dot = tmp_path / f"native-router-m{m}.dot"
        graph.debug_dump(str(dot))
        captured = dot.read_text(encoding="utf-8")
        assert program.metadata["entries"][0]["name"] in captured
        assert captured.count('label="{KERNEL') == 1
        uncovered = torch.randn(5, 7168, dtype=torch.bfloat16, device="cuda")
        actual = kimi3_router_projection(uncovered, b, solution="auto")
        torch.testing.assert_close(
            actual, torch.mm(uncovered, b.T, out_dtype=torch.float32)
        )
        if m == 1:
            overlapping = a.view(torch.float32)[:, :896]
            with pytest.raises(ValueError, match="overlap"):
                kimi3_router_projection(a, b, out=overlapping, solution="cute_dsl_aot")
        pdl_enabled(True)
        with pytest.raises(ValueError):
            kimi3_router_projection(a, b, out=out, solution="cute_dsl_aot")
    finally:
        pdl_enabled(old_pdl)
        configured_router_aot.cache_clear()


def test_native_router_rejects_duplicate_policy(
    native_router_policy, tmp_path, build_options
):
    policy = json.loads(native_router_policy.read_text(encoding="utf-8"))
    policy["selections"] *= 2
    config = tmp_path / "duplicate.json"
    config.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        RouterHostCatalog(config)
    original = native_router_policy.read_bytes()
    with pytest.raises(FileExistsError):
        prepare_router_host(
            compiler_python=build_options["compiler_python"],
            cache=build_options["cache"],
            cuda_home=build_options["cuda_home"],
            cxx=build_options["cxx"],
            blocks={1: 128},
            configuration=native_router_policy,
        )
    assert native_router_policy.read_bytes() == original


def test_native_router_rejects_mismatched_factory(native_router_policy, tmp_path):
    policy = json.loads(native_router_policy.read_text(encoding="utf-8"))
    policy["selections"] = policy["selections"][:1]
    selection = policy["selections"][0]
    directory = Path(selection["program"])
    manifest = json.loads((directory / "artifact.json").read_text(encoding="utf-8"))
    for filename in ("kernel.cubin", "kernel.ptx", *manifest["host_files"]):
        shutil.copyfile(directory / filename, tmp_path / filename)
    manifest["source_sha256"] = "0" * 64
    (tmp_path / "artifact.json").write_text(json.dumps(manifest), encoding="utf-8")
    selection["program"] = str(tmp_path)
    config = tmp_path / "selection.json"
    config.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(ValueError, match="installed source/contract"):
        RouterHostCatalog(config)
