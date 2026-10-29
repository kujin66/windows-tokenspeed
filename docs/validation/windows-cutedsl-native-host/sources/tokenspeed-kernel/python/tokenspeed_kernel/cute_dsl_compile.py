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

"""Compile a user CuTe factory in a Windows compiler process; export device code.

This supports the official x64 compiler under Windows ARM64 system emulation.
It is not an ARM64 compiler port or a replacement for the CuTe DSL API.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


def pe_machine(executable: Path) -> int:
    """Read the actual executable architecture (platform.machine may report ARM64)."""
    with executable.open("rb") as stream:
        if stream.read(2) != b"MZ":
            raise ValueError("The compiler must be a Windows PE executable")
        stream.seek(0x3C)
        offset = struct.unpack("<I", stream.read(4))[0]
        stream.seek(offset)
        if stream.read(4) != b"PE\0\0":
            raise ValueError("Invalid Windows PE header")
        return struct.unpack("<H", stream.read(2))[0]


def _require_preparation() -> None:
    if sys.platform != "win32":
        raise RuntimeError("This compiler runner requires Windows")
    torch_module = sys.modules.get("torch")
    if (
        torch_module is not None
        and torch_module.cuda.is_initialized()
        and torch_module.cuda.is_current_stream_capturing()
    ):
        raise RuntimeError("Compile CuTe artifacts before CUDA graph capture")
    monitor = sys.modules.get("tokenspeed_kernel.compile_monitor")
    if monitor is not None and monitor.is_serving():
        raise RuntimeError("Compile CuTe artifacts during preparation, before serving")


def validate_artifact(directory: Path) -> dict:
    """Read an exported device artifact and verify both device-code hashes."""
    directory = Path(directory)
    manifest = json.loads((directory / "artifact.json").read_text(encoding="utf-8"))
    if manifest["schema"] != 1:
        raise ValueError("Unsupported CuTe artifact schema")
    for extension in ("cubin", "ptx"):
        payload = (directory / f"kernel.{extension}").read_bytes()
        if hashlib.sha256(payload).hexdigest() != manifest[f"{extension}_sha256"]:
            raise ValueError(f"CuTe {extension} hash mismatch")
    for name, digest in manifest.get("host_files", {}).items():
        if Path(name).name != name:
            raise ValueError("Host artifact files must be basenames")
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"CuTe host file hash mismatch: {name}")
    return manifest


def _compile_artifact(
    *,
    compiler_python: Path,
    source: Path,
    factory: str,
    parameters: dict,
    arch: str,
    directory: Path,
    worker_action: str,
) -> dict:
    """Compile a factory on demand, outside CUDA capture and serving threads.

    Args:
        compiler_python: Windows x64 or ARM64 Python with official CuTe installed.
        source: Python module defining factory(parameters) -> cute.compile result.
        factory: Name of that callable. The supplied source runs as Python code.
        parameters: JSON-serializable explicit compilation arguments.
        arch: Explicit NVIDIA target, such as sm_121a.
        directory: New destination directory; an existing destination is rejected.

    Returns:
        Manifest with compiler provenance, PTX entries and device CUBIN hash.
        Consumers must supply the kernel's own ABI, launch geometry and checks.
    """
    _require_preparation()
    compiler_python, source, directory = (
        Path(p).resolve() for p in (compiler_python, source, directory)
    )
    if pe_machine(compiler_python) not in (0x8664, 0xAA64):
        raise ValueError("Expected an x64 or ARM64 Windows compiler Python")
    if not re.fullmatch(r"sm_[0-9]+[af]?", arch):
        raise ValueError("An explicit sm_* NVIDIA architecture is required")
    if not source.is_file() or not factory.isidentifier():
        raise ValueError("Expected a source file and a factory identifier")
    if directory.exists():
        raise FileExistsError(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="cute-compile-", dir=directory.parent
    ) as temp:
        staging = Path(temp)
        command = [
            str(compiler_python),
            "-I",
            str(Path(__file__).resolve()),
            worker_action,
            "--source",
            str(source),
            "--factory",
            factory,
            "--parameters",
            json.dumps(parameters),
            "--arch",
            arch,
            "--directory",
            str(staging),
        ]
        completed = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        log = completed.stdout + completed.stderr
        (directory.parent / (directory.name + "-compile.log")).write_text(
            log, encoding="utf-8"
        )
        if completed.returncode:
            raise RuntimeError(
                f"Windows CuTe compilation failed ({completed.returncode}): {log[-6000:]}"
            )
        manifest = validate_artifact(staging)
        if manifest["source_sha256"] != hashlib.sha256(source.read_bytes()).hexdigest():
            raise RuntimeError("Compiler source changed during compilation")
        if manifest["arch"] != arch or manifest["compiler"]["os"] != "win32":
            raise RuntimeError("Compiler reported a different target/host")
        if (
            hashlib.sha256((staging / "kernel.cubin").read_bytes()).hexdigest()
            != manifest["cubin_sha256"]
        ):
            raise RuntimeError("Compiler artifact hash mismatch")
        # Publish the directory only after every compiler/artifact check succeeds.
        staging.rename(directory)
        return manifest


def compile_artifact(
    *,
    compiler_python: Path,
    source: Path,
    factory: str,
    parameters: dict,
    arch: str,
    directory: Path,
) -> dict:
    """Export a user factory's device code in an isolated Windows compiler.

    The source defines factory(parameters) returning a cute.compile result.
    compiler_python selects the compiler interpreter; arch selects the GPU.
    directory must be new. The result is a hash-validated device manifest;
    the caller supplies the device ABI, launch geometry and operation contract.
    """
    return _compile_artifact(
        compiler_python=compiler_python,
        source=source,
        factory=factory,
        parameters=parameters,
        arch=arch,
        directory=directory,
        worker_action="worker",
    )


def _compile_cached_artifact(
    *,
    compiler_python: Path,
    source: Path,
    factory: str,
    parameters: dict,
    arch: str,
    cache: Path,
    dependencies: tuple[Path, ...],
    worker_action: str,
) -> Path:
    """Compile on a cache miss, or verify and reuse a device artifact.

    Arguments match compile_artifact, except cache is a content-addressed root.
    dependencies explicitly lists imported source/helper files and external
    build inputs; include every file that can affect the factory's result.
    The key also includes the source, factory, parameters, target, worker,
    compiler executable, installed package records and CUDA/build environment.
    This preparation API performs file I/O and a compiler-environment probe
    even on a hit. Retain the loaded kernel for capture and serving.
    """
    _require_preparation()
    compiler_python, source, cache = (
        Path(p).resolve() for p in (compiler_python, source, cache)
    )
    # Probe the isolated interpreter; native ARM64 packages are not the compiler.
    probe = subprocess.run(
        [
            str(compiler_python),
            "-I",
            "-c",
            "import hashlib, importlib.metadata as m, json, sys; "
            "print(json.dumps([sys.version, sorted("
            "(d.metadata['Name'], d.version, hashlib.sha256("
            "(d.read_text('RECORD') or '').encode()).hexdigest()) "
            "for d in m.distributions())]))",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    inputs = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (source, *(Path(p).resolve() for p in dependencies))
    }
    identity = {
        "worker_action": worker_action,
        "inputs": inputs,
        "source": str(source),
        "factory": factory,
        "parameters": parameters,
        "arch": arch,
        "compiler_python": str(compiler_python),
        "compiler_exe_sha256": hashlib.sha256(compiler_python.read_bytes()).hexdigest(),
        "environment": json.loads(probe.stdout),
        "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "build_environment": {
            k: v
            for k, v in os.environ.items()
            if k.upper().startswith(("CUTE_DSL_", "CUDA_")) or k.upper() == "PATH"
        },
    }
    if worker_action == "host-worker":
        identity["host_exporter_sha256"] = hashlib.sha256(
            Path(__file__).with_name("cute_dsl_host_export.py").read_bytes()
        ).hexdigest()
    key = hashlib.sha256(
        json.dumps(identity, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()
    directory = cache / key
    if directory.exists():
        manifest = validate_artifact(directory)
        if (
            json.loads((directory / "cache-inputs.json").read_text(encoding="utf-8"))
            != identity
        ):
            raise ValueError("CuTe cache identity mismatch")
        if (
            manifest["source_sha256"] != inputs[str(source)]
            or manifest["factory"] != factory
            or manifest["parameters"] != parameters
            or manifest["arch"] != arch
        ):
            raise ValueError("CuTe cache manifest mismatch")
        return directory
    cache.mkdir(parents=True, exist_ok=True)
    # Each process builds privately. Rename publishes a complete entry atomically.
    with tempfile.TemporaryDirectory(prefix="cute-cache-", dir=cache) as temp:
        staging = Path(temp) / "artifact"
        _compile_artifact(
            compiler_python=compiler_python,
            source=source,
            factory=factory,
            parameters=parameters,
            arch=arch,
            directory=staging,
            worker_action=worker_action,
        )
        if any(
            hashlib.sha256(Path(p).read_bytes()).hexdigest() != digest
            for p, digest in inputs.items()
        ):
            raise RuntimeError("CuTe build inputs changed during compilation")
        (staging / "cache-inputs.json").write_text(
            json.dumps(identity, indent=2, sort_keys=True), encoding="utf-8"
        )
        try:
            staging.rename(directory)
        except FileExistsError:
            # Another process published the same specialization first.
            validate_artifact(directory)
            if (
                json.loads(
                    (directory / "cache-inputs.json").read_text(encoding="utf-8")
                )
                != identity
            ):
                raise ValueError("CuTe cache identity mismatch")
    return directory


def compile_cached_artifact(
    *,
    compiler_python: Path,
    source: Path,
    factory: str,
    parameters: dict,
    arch: str,
    cache: Path,
    dependencies: tuple[Path, ...],
) -> Path:
    """Compile or verify a cached device artifact. Declare all imported inputs.

    Parameters match compile_artifact except cache is a content-addressed root,
    and dependencies lists helper sources/external build inputs. Factories must
    be deterministic; installed compiler packages and toolchains are immutable.
    Call only during preparation; retain loaded kernels for serving and capture.
    """
    return _compile_cached_artifact(
        compiler_python=compiler_python,
        source=source,
        factory=factory,
        parameters=parameters,
        arch=arch,
        cache=cache,
        dependencies=dependencies,
        worker_action="worker",
    )


def compile_cached_host_artifact(
    *,
    compiler_python: Path,
    source: Path,
    factory: str,
    parameters: dict,
    arch: str,
    cache: Path,
    dependencies: tuple[Path, ...],
) -> Path:
    """Compile the device code and its original host launcher for Windows ARM64.

    Arguments and cache/preparation contracts match compile_cached_artifact.
    The factory uses string cute.compile options. Exported C signatures and
    tensor contracts are inferred from the real compilation, not a launch trace.
    """
    return _compile_cached_artifact(
        compiler_python=compiler_python,
        source=source,
        factory=factory,
        parameters=parameters,
        arch=arch,
        cache=cache,
        dependencies=dependencies,
        worker_action="host-worker",
    )


def worker(args):
    if sys.platform != "win32":
        raise RuntimeError("The compiler worker must execute on Windows")
    os.environ["CUTE_DSL_ARCH"] = args.arch
    os.environ["CUTE_DSL_DUMP_DIR"] = str(args.directory)
    os.chdir(args.directory)
    source_hash = hashlib.sha256(args.source.read_bytes()).hexdigest()
    # User factories can import adjacent helpers; dependencies belong in cache keys.
    sys.dont_write_bytecode = True
    # Existing timestamp-based .pyc files can hide same-size source edits. Use
    # a fresh bytecode location so the content-addressed build reads real source.
    sys.pycache_prefix = str(args.directory / "python-bytecode")
    sys.path.insert(0, str(args.source.parent))
    host_export = None
    if args.action == "host-worker":
        export_spec = importlib.util.spec_from_file_location(
            "tokenspeed_host_export",
            Path(__file__).with_name("cute_dsl_host_export.py"),
        )
        host_export = importlib.util.module_from_spec(export_spec)
        sys.modules[export_spec.name] = host_export
        export_spec.loader.exec_module(host_export)
        host_export.install_compiler_hook()
    spec = importlib.util.spec_from_file_location("user_cute_factory", args.source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    compiled = getattr(module, args.factory)(json.loads(args.parameters))
    ptx = compiled.__ptx__
    if isinstance(ptx, bytes):
        ptx = ptx.decode("utf-8")
    cubin = bytes(compiled.__cubin__)
    entries = re.findall(r"\.entry\s+([^\s(]+)\s*\((.*?)\)", ptx, re.S)
    if not entries or not re.search(rf"\.target\s+{re.escape(args.arch)}\b", ptx):
        raise ValueError("Compiler did not export the requested target kernels")
    (args.directory / "kernel.cubin").write_bytes(cubin)
    # Hash and publish identical bytes; Windows text mode otherwise adds CRLF.
    (args.directory / "kernel.ptx").write_bytes(ptx.encode("utf-8"))
    manifest = {
        "schema": 1,
        "arch": args.arch,
        "source_sha256": source_hash,
        "factory": args.factory,
        "parameters": json.loads(args.parameters),
        "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cubin_sha256": hashlib.sha256(cubin).hexdigest(),
        "ptx_sha256": hashlib.sha256(ptx.encode("utf-8")).hexdigest(),
        "entries": [
            {"name": name, "ptx_parameters": params} for name, params in entries
        ],
        "compiler": {
            "os": sys.platform,
            "python": sys.version,
            "python_pe_machine": hex(pe_machine(Path(sys.executable))),
            "cutedsl": importlib.metadata.version("nvidia-cutlass-dsl"),
            "cuda_bindings": importlib.metadata.version("cuda-bindings"),
        },
        "scope": "Windows-local compiler process; x64 may use system emulation; device code only, caller owns ABI and launch contract",
    }
    if host_export is not None:
        manifest["host"] = host_export.export_host(compiled, args.directory)
        manifest["host_files"] = {
            name: hashlib.sha256((args.directory / name).read_bytes()).hexdigest()
            for name in ("host.h", "host.o", "adapter.cpp")
        }
    (args.directory / "artifact.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for name in ("compile", "worker", "host-worker"):
        child = commands.add_parser(name)
        if name == "compile":
            child.add_argument("--compiler-python", type=Path, required=True)
        child.add_argument("--source", type=Path, required=True)
        child.add_argument("--factory", required=True)
        child.add_argument("--parameters", required=True)
        child.add_argument("--arch", required=True)
        child.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    args.source, args.directory = args.source.resolve(), args.directory.resolve()
    if args.action in ("worker", "host-worker"):
        worker(args)
    else:
        result = compile_artifact(
            compiler_python=args.compiler_python,
            source=args.source,
            factory=args.factory,
            parameters=json.loads(args.parameters),
            arch=args.arch,
            directory=args.directory,
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
