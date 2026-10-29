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

"""Real Windows-local CuTe compilation and native ARM64 device execution tests."""

import ctypes
import os
import py_compile
from pathlib import Path

import pytest
import torch
from tokenspeed_kernel.cute_dsl_compile import (
    compile_artifact,
    compile_cached_artifact,
    pe_machine,
    validate_artifact,
)
from tokenspeed_kernel.cute_dsl_runtime import CuTeKernel, compile_kernel


@pytest.fixture(scope="module")
def compiler():
    value = os.environ.get("TOKENSPEED_CUTE_DSL_COMPILER_PYTHON")
    if os.name != "nt" or value is None or not torch.cuda.is_available():
        pytest.skip(
            "Requires Windows, CUDA and an explicitly configured compiler Python"
        )
    return Path(value)


@pytest.mark.parametrize("operation,block", [("add", 128), ("multiply", 256)])
def test_local_compile_and_native_graph_execution(compiler, tmp_path, operation, block):
    factory = (
        Path(__file__).resolve().parents[3]
        / "tools/windows/cute_dsl_aot/elementwise_factory.py"
    )
    artifact = tmp_path / operation
    result = compile_artifact(
        compiler_python=compiler,
        source=factory,
        factory="compile_elementwise",
        parameters={"operation": operation, "block_size": block},
        arch="sm_121a",
        directory=artifact,
    )
    assert result["compiler"]["os"] == "win32"
    assert int(result["compiler"]["python_pe_machine"], 16) == pe_machine(compiler)
    assert len(result["entries"]) == 1
    assert len(result["entries"][0]["ptx_parameters"].split(",")) == 4
    torch.cuda.set_device(0)
    torch.empty(0, device="cuda")
    driver = ctypes.WinDLL("nvcuda.dll")
    ptr = ctypes.c_void_p
    signatures = {
        "cuModuleLoadData": [ctypes.POINTER(ptr), ptr],
        "cuModuleGetFunction": [ctypes.POINTER(ptr), ptr, ctypes.c_char_p],
        "cuLaunchKernel": [
            ptr,
            *([ctypes.c_uint] * 7),
            ptr,
            ctypes.POINTER(ptr),
            ctypes.POINTER(ptr),
        ],
        "cuModuleUnload": [ptr],
    }
    for name, signature in signatures.items():
        function = getattr(driver, name)
        function.argtypes, function.restype = signature, ctypes.c_int
    module, function = ptr(), ptr()
    image = ctypes.create_string_buffer((artifact / "kernel.cubin").read_bytes())
    assert driver.cuModuleLoadData(ctypes.byref(module), image) == 0
    try:
        entry = result["entries"][0]["name"]
        assert (
            driver.cuModuleGetFunction(ctypes.byref(function), module, entry.encode())
            == 0
        )
        for n in (1, 127, 128, 129, 257, 4096):
            a = torch.randn(n, dtype=torch.float32, device="cuda")
            b = torch.randn_like(a)
            out = torch.empty_like(a)

            def launch():
                values = [
                    ptr(a.data_ptr()),
                    ptr(b.data_ptr()),
                    ptr(out.data_ptr()),
                    ctypes.c_int32(n),
                ]
                args = (ptr * 4)(*[ctypes.cast(ctypes.byref(v), ptr) for v in values])
                stream = ptr(torch.cuda.current_stream().cuda_stream)
                code = driver.cuLaunchKernel(
                    function,
                    (n + block - 1) // block,
                    1,
                    1,
                    block,
                    1,
                    1,
                    0,
                    stream,
                    args,
                    None,
                )
                assert code == 0

            launch()
            expected = a + b if operation == "add" else a * b
            torch.testing.assert_close(out, expected, atol=0, rtol=0)
            graph = torch.cuda.CUDAGraph()
            graph.enable_debug_mode()
            with torch.cuda.graph(graph):
                launch()
            for _ in range(3):
                a.normal_()
                b.normal_()
                out.fill_(float("nan"))
                graph.replay()
                expected = a + b if operation == "add" else a * b
                torch.testing.assert_close(out, expected, atol=0, rtol=0)
            dot_path = tmp_path / f"{operation}-{n}.dot"
            graph.debug_dump(str(dot_path))
            assert entry in dot_path.read_text(encoding="utf-8")
            del graph
    finally:
        torch.cuda.synchronize()
        assert driver.cuModuleUnload(module) == 0


def test_failed_compile_does_not_publish_artifact(compiler, tmp_path):
    source = tmp_path / "broken_factory.py"
    source.write_text(
        "def compile_kernel(parameters):\n    raise ValueError('intentional invalid compilation input')\n"
    )
    artifact = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="intentional invalid"):
        compile_artifact(
            compiler_python=compiler,
            source=source,
            factory="compile_kernel",
            parameters={},
            arch="sm_121a",
            directory=artifact,
        )
    assert not artifact.exists()


def test_compiler_refuses_cuda_graph_capture(compiler, tmp_path):
    output = torch.empty(1, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with pytest.raises(RuntimeError, match="before CUDA graph capture"):
            compile_artifact(
                compiler_python=compiler,
                source=tmp_path / "not-read.py",
                factory="compile_kernel",
                parameters={},
                arch="sm_121a",
                directory=tmp_path / "not-created",
            )
        output.zero_()
    graph.replay()
    assert not (tmp_path / "not-created").exists()


@pytest.fixture(scope="module")
def custom_artifacts(compiler, tmp_path_factory):
    root = tmp_path_factory.mktemp("custom-cute")
    source = (
        Path(__file__).resolve().parents[3]
        / "tools/windows/cute_dsl_aot/elementwise_factory.py"
    )
    artifacts = {}
    for name, parameters in (("affine", {"block_size": 128}), ("transpose", {})):
        directory = root / name
        manifest = compile_artifact(
            compiler_python=compiler,
            source=source,
            factory=f"compile_{name}",
            parameters=parameters,
            arch="sm_121a",
            directory=directory,
        )
        artifacts[name] = (directory, manifest["entries"][0]["name"])
    torch.cuda.set_device(0)
    return artifacts


def test_generic_runtime_scalar_abi_streams_and_graphs(custom_artifacts, tmp_path):
    directory, entry = custom_artifacts["affine"]
    kernel = CuTeKernel(
        directory,
        entry=entry,
        argument_types=("ptr", "ptr", "i32", "i64", "f32", "f32"),
        device=torch.device("cuda:0"),
        shared_memory_bytes=0,
    )
    try:
        for n in (1, 127, 128, 129, 4096):
            for stride in (1, 2):
                a = torch.arange(n * stride, dtype=torch.float32, device="cuda")
                out = torch.empty(n, device="cuda")
                grid = ((n + 127) // 128, 1, 1)

                def launch():
                    kernel.launch(
                        a,
                        out,
                        n,
                        stride,
                        2.0,
                        1.0,
                        grid=grid,
                        block=(128, 1, 1),
                        shared_memory_bytes=0,
                    )

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    launch()
                torch.cuda.current_stream().wait_stream(stream)
                torch.testing.assert_close(out, a[::stride] * 2 + 1, atol=0, rtol=0)
                graph = torch.cuda.CUDAGraph()
                graph.enable_debug_mode()
                with torch.cuda.graph(graph):
                    launch()
                for _ in range(3):
                    a.add_(1)
                    out.fill_(float("nan"))
                    graph.replay()
                    torch.testing.assert_close(out, a[::stride] * 2 + 1, atol=0, rtol=0)
                dot = tmp_path / f"affine-{n}-{stride}.dot"
                graph.debug_dump(str(dot))
                assert entry in dot.read_text(encoding="utf-8")
                del graph
    finally:
        kernel.close()
    with pytest.raises(RuntimeError, match="closed"):
        launch()


def test_generic_runtime_two_dimensional_launch(custom_artifacts):
    directory, entry = custom_artifacts["transpose"]
    kernel = CuTeKernel(
        directory,
        entry=entry,
        argument_types=("ptr", "ptr", "i32", "i32"),
        device=torch.device("cuda:0"),
        shared_memory_bytes=0,
    )
    try:
        for m, n in ((1, 1), (15, 17), (33, 65)):
            a = torch.randn(m, n, device="cuda")
            out = torch.empty(n, m, device="cuda")
            grid = ((n + 15) // 16, (m + 15) // 16, 1)
            kernel.launch(
                a, out, m, n, grid=grid, block=(16, 16, 1), shared_memory_bytes=0
            )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                kernel.launch(
                    a, out, m, n, grid=grid, block=(16, 16, 1), shared_memory_bytes=0
                )
            for _ in range(3):
                a.normal_()
                graph.replay()
                torch.testing.assert_close(out, a.T, atol=0, rtol=0)
            del graph
    finally:
        kernel.close()


def test_generic_runtime_rejects_bad_abi_and_launch(custom_artifacts):
    directory, entry = custom_artifacts["affine"]
    kinds = ("ptr", "ptr", "i32", "i64", "f32", "f32")
    with pytest.raises(ValueError, match="width"):
        CuTeKernel(
            directory,
            entry=entry,
            argument_types=(*kinds[:2], "i64", *kinds[3:]),
            device=torch.device("cuda:0"),
            shared_memory_bytes=0,
        )
    kernel = CuTeKernel(
        directory,
        entry=entry,
        argument_types=kinds,
        device=torch.device("cuda:0"),
        shared_memory_bytes=0,
    )
    a = torch.empty(128, device="cuda")
    try:
        options = dict(grid=(1, 1, 1), block=(128, 1, 1), shared_memory_bytes=0)
        with pytest.raises(ValueError, match="i32 range"):
            kernel.launch(a, a, 2**31, 1, 1.0, 0.0, **options)
        with pytest.raises(ValueError, match="preloaded CUDA device"):
            kernel.launch(a.cpu(), a, 128, 1, 1.0, 0.0, **options)
        with pytest.raises(ValueError, match="alignment"):
            kernel.launch(a[1:], a, 127, 1, 1.0, 0.0, **options)
        with pytest.raises(ValueError, match="Block"):
            kernel.launch(a, a, 128, 1, 1.0, 0.0, **{**options, "block": (64, 1, 1)})
        with pytest.raises(ValueError, match="shared memory"):
            kernel.launch(
                a, a, 128, 1, 1.0, 0.0, **{**options, "shared_memory_bytes": 1}
            )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            with pytest.raises(RuntimeError, match="before CUDA graph capture"):
                CuTeKernel(
                    directory,
                    entry=entry,
                    argument_types=kinds,
                    device=torch.device("cuda:0"),
                    shared_memory_bytes=0,
                )
            with pytest.raises(RuntimeError, match="during CUDA graph capture"):
                kernel.close()
            a.zero_()
        graph.replay()
    finally:
        kernel.close()


def test_compile_cache_reuse_invalidation_and_corruption(compiler, tmp_path):
    example = (
        Path(__file__).resolve().parents[3]
        / "tools/windows/cute_dsl_aot/elementwise_factory.py"
    )
    helper = tmp_path / "choice.py"
    helper.write_text("operation = 'add'".ljust(24) + "\n", encoding="utf-8")
    py_compile.compile(str(helper), doraise=True)
    helper_stat = helper.stat()
    source = tmp_path / "factory.py"
    source.write_text(
        f"import sys\nsys.path.insert(0, {str(example.parent)!r})\n"
        "from elementwise_factory import compile_elementwise\nimport choice\n"
        "def build(parameters):\n    return compile_elementwise({**parameters, 'operation': choice.operation})\n",
        encoding="utf-8",
    )
    options = dict(
        compiler_python=compiler,
        source=source,
        factory="build",
        parameters={"block_size": 128},
        arch="sm_121a",
        cache=tmp_path / "cache",
        dependencies=(example, helper),
    )
    first = compile_cached_artifact(**options)
    timestamp = (first / "kernel.cubin").stat().st_mtime_ns
    assert compile_cached_artifact(**options) == first
    assert (first / "kernel.cubin").stat().st_mtime_ns == timestamp
    loaded = compile_kernel(
        **options,
        entry=None,
        argument_types=("ptr", "ptr", "ptr", "i32"),
        device=torch.device("cuda:0"),
        shared_memory_bytes=0,
    )
    try:
        a = torch.ones(128, device="cuda")
        b, out = torch.full_like(a, 2), torch.empty_like(a)
        loaded.launch(
            a, b, out, 128, grid=(1, 1, 1), block=(128, 1, 1), shared_memory_bytes=0
        )
        torch.testing.assert_close(out, a + b, atol=0, rtol=0)
    finally:
        loaded.close()
    assert (first / "kernel.cubin").stat().st_mtime_ns == timestamp
    helper.write_text("operation = 'multiply'".ljust(24) + "\n", encoding="utf-8")
    os.utime(helper, ns=(helper_stat.st_atime_ns, helper_stat.st_mtime_ns))
    assert helper.stat().st_size == helper_stat.st_size
    second = compile_cached_artifact(**options)
    assert first != second
    for directory, expected_operation in ((first, "add"), (second, "multiply")):
        manifest = validate_artifact(directory)
        kernel = CuTeKernel(
            directory,
            entry=manifest["entries"][0]["name"],
            argument_types=("ptr", "ptr", "ptr", "i32"),
            device=torch.device("cuda:0"),
            shared_memory_bytes=0,
        )
        try:
            a, b = torch.randn(129, device="cuda"), torch.randn(129, device="cuda")
            out = torch.empty_like(a)
            kernel.launch(
                a, b, out, 129, grid=(2, 1, 1), block=(128, 1, 1), shared_memory_bytes=0
            )
            torch.testing.assert_close(
                out, a + b if expected_operation == "add" else a * b, atol=0, rtol=0
            )
        finally:
            kernel.close()
    source.write_text(
        source.read_text(encoding="utf-8") + "\n# new source revision\n",
        encoding="utf-8",
    )
    third = compile_cached_artifact(**options)
    assert third not in (first, second)
    (third / "kernel.cubin").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        compile_cached_artifact(**options)
    with pytest.raises(ValueError, match="hash mismatch"):
        CuTeKernel(
            third,
            entry="ignored",
            argument_types=(),
            device=torch.device("cuda:0"),
            shared_memory_bytes=0,
        )
