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

"""Installed-wheel and CUDA graph checks for Windows ARM64 wheels."""

import hashlib
import importlib.metadata
import os
import shutil
import struct
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

import pytest

from tools.windows.build_wheels import audit_wheel, pe_machine, run_build_command
from tools.windows.prepare_public_dependencies import PACKAGES, adapt_windows_headers


@pytest.mark.parametrize("patch_eol", [b"\n", b"\r\n"])
def test_header_patch_is_independent_of_enclosing_git_repository(tmp_path, patch_eol):
    package = PACKAGES[2]
    wheel_input = os.environ.get("TOKENSPEED_FLASHINFER_TEST_WHEEL")
    if wheel_input is None:
        wheel_input = str(
            Path(os.environ["TOKENSPEED_FLASHINFER_DATA_DIR"]).parent
            / package["filename"]
        )
    wheel = Path(wheel_input)
    assert hashlib.sha256(wheel.read_bytes()).hexdigest() == package["sha256"]
    patch = (
        Path(__file__).resolve().parents[2]
        / "tools/windows/flashinfer-build-compat.patch"
    )
    names = [
        line.split(" b/", 1)[1]
        for line in patch.read_text().splitlines()
        if line.startswith("diff --git ")
    ]
    canonical_data = patch.read_text(encoding="utf-8").encode("utf-8")
    canonical_patch = tmp_path / "canonical.patch"
    canonical_patch.write_bytes(canonical_data)
    checkout_patch = tmp_path / "checkout.patch"
    checkout_patch.write_bytes(canonical_data.replace(b"\n", patch_eol))
    outside = tmp_path / "standalone-headers"
    with zipfile.ZipFile(wheel) as archive:
        for name in names:
            target = outside / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(f"flashinfer/data/{name}"))
    checkout = tmp_path / "checkout"
    subprocess.run(["git", "init", str(checkout)], check=True, capture_output=True)
    inside = checkout / "outputs/public-dependencies/flashinfer-headers"
    shutil.copytree(outside, inside)
    expected = adapt_windows_headers(outside, patch=canonical_patch)
    actual = adapt_windows_headers(inside, patch=checkout_patch)
    assert actual == expected
    assert all(r["original_sha256"] != r["patched_sha256"] for r in actual)
    for record in actual:
        assert (inside / record["path"]).read_bytes() == (
            outside / record["path"]
        ).read_bytes()
    # Exercise the same process-only configuration used by runner checkout.
    assert checkout.resolve().is_relative_to(tmp_path.resolve())
    cleanup_env = dict(os.environ)
    config_count = int(cleanup_env.get("GIT_CONFIG_COUNT", "0"))
    cleanup_env[f"GIT_CONFIG_KEY_{config_count}"] = "core.longpaths"
    cleanup_env[f"GIT_CONFIG_VALUE_{config_count}"] = "true"
    cleanup_env["GIT_CONFIG_COUNT"] = str(config_count + 1)
    subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={checkout.as_posix()}",
            "clean",
            "-ffd",
            "--",
            "outputs",
        ],
        cwd=checkout,
        env=cleanup_env,
        check=True,
        capture_output=True,
    )
    assert not (checkout / "outputs").exists()


@pytest.mark.parametrize(
    "message",
    [
        "error[E0308]: mismatched types",
        "failed to remove temporary directory: (os error 5) at x.temp-archive",
        "failed to open output file: (os error 32)",
    ],
)
def test_build_command_preserves_failure(tmp_path, message):
    marker = tmp_path / "first-attempt"
    script = (
        "from pathlib import Path; import sys; "
        "Path(sys.argv[1]).open('x').close(); print(sys.argv[2]); sys.exit(7)"
    )
    command = (sys.executable, "-c", script, str(marker), message)
    with pytest.raises(subprocess.CalledProcessError) as failure:
        run_build_command(command, cwd=tmp_path, env=dict(os.environ))
    assert failure.value.returncode == 7
    assert message.encode() in failure.value.output


@pytest.mark.skipif(sys.platform != "win32", reason="Windows file sharing semantics")
@pytest.mark.parametrize("transient", [True, False])
def test_build_command_handles_real_archive_lock(tmp_path, transient):
    import ctypes
    from ctypes import wintypes

    target = tmp_path / "cargo.temp-archive"
    target.write_bytes(b"archive")
    marker = tmp_path / "sharing-violation"
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    # Permit reads/writes but omit FILE_SHARE_DELETE, reproducing OS error 32.
    handle = kernel32.CreateFileW(str(target), 0x80000000, 3, None, 3, 0x80, None)
    assert handle != ctypes.c_void_p(-1).value, ctypes.get_last_error()
    release_allowed = threading.Event()

    def release_after_violation():
        try:
            deadline = time.monotonic() + 10
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            if not transient:
                # Bound the test too: an unbounded retry would recover and fail
                # the expected-error assertion once this safety timeout expires.
                release_allowed.wait(10)
        finally:
            kernel32.CloseHandle(handle)

    release = threading.Thread(target=release_after_violation)
    release.start()
    script = """
from pathlib import Path
import sys
try:
    Path(sys.argv[1]).unlink()
except OSError as error:
    Path(sys.argv[2]).write_text(str(error.winerror))
    print(f"failed to remove temporary directory: (os error {error.winerror}) at {sys.argv[1]}")
    raise SystemExit(1)
"""
    try:
        command = (sys.executable, "-c", script, str(target), str(marker))
        if transient:
            run_build_command(command, cwd=tmp_path, env=dict(os.environ))
            assert not target.exists()
        else:
            with pytest.raises(subprocess.CalledProcessError) as failure:
                run_build_command(command, cwd=tmp_path, env=dict(os.environ))
            assert failure.value.returncode == 1
            assert target.exists()
    finally:
        release_allowed.set()
        release.join()
    assert marker.read_text() == "32"


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
    licenses = [
        Path(module.__file__).parent / "licenses" / "OPENSSL-LICENSE.txt"
        for module in (tokenspeed_engine, tokenspeed_scheduler)
    ]
    assert licenses[0].read_bytes() == licenses[1].read_bytes()
    assert b"Apache License" in licenses[0].read_bytes()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows OpenSSL SHA256 check")
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
