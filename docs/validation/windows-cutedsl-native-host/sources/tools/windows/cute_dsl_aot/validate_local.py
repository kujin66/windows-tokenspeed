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

"""Rebuild Windows-local artifacts and retain real GPU/compiler test evidence."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tokenspeed-kernel/python"))

from tokenspeed_kernel.cute_dsl_compile import pe_machine  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler-python", type=Path, required=True)
    parser.add_argument("--work-directory", type=Path, required=True)
    parser.add_argument("--evidence-directory", type=Path, required=True)
    parser.add_argument("--suite", choices=("device", "host", "all"), required=True)
    parser.add_argument(
        "--router-blocks",
        type=int,
        nargs=3,
        required=True,
        choices=(64, 128, 256, 512),
        metavar=("M1", "M2", "M4"),
    )
    args = parser.parse_args()
    if sys.platform != "win32" or pe_machine(Path(sys.executable)) != 0xAA64:
        parser.error("Validation requires native Windows ARM64 Python")
    work, evidence = args.work_directory.resolve(), args.evidence_directory.resolve()
    if work.exists() or evidence.exists():
        parser.error(
            "Choose new work and evidence directories; existing data is preserved"
        )
    work.mkdir(parents=True)
    evidence.mkdir(parents=True)
    compiler = args.compiler_python.resolve()
    source = (
        ROOT
        / "tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cute_dsl/ll_bf16/_kernel.py"
    )
    router = work / "router"
    build = subprocess.run(
        [
            str(compiler),
            "-I",
            str(Path(__file__).with_name("export_router.py")),
            "--source",
            str(source),
            "--output",
            str(router),
            "--block-sizes",
            "64",
            "128",
            "256",
            "512",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    (evidence / "router-compile.log").write_text(
        build.stdout + build.stderr, encoding="utf-8"
    )
    if build.returncode:
        raise SystemExit(
            f"Router compilation failed; see {evidence / 'router-compile.log'}"
        )
    config = router / "selection.json"
    config.write_text(
        json.dumps(
            {
                "schema": 1,
                "directory": ".",
                "selections": [
                    {"shape_mnk": [m, 896, 7168], "block_size": block}
                    for m, block in zip((1, 2, 4), args.router_blocks)
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "tokenspeed-kernel/python")
    env["TOKENSPEED_CUTE_DSL_COMPILER_PYTHON"] = str(compiler)
    env["TOKENSPEED_CUTE_DSL_AOT_CONFIG"] = str(config)
    device_tests = [
        "tokenspeed-kernel/test/nvidia/test_cute_dsl_windows_compile.py",
        "tokenspeed-kernel/test/nvidia/test_cute_dsl_aot.py",
    ]
    host_tests = ["tokenspeed-kernel/test/nvidia/test_cute_dsl_host.py"]
    tests = {
        "device": device_tests,
        "host": host_tests,
        "all": device_tests + host_tests,
    }[args.suite]
    command = [
        sys.executable,
        "-m",
        "pytest",
        *tests,
        "-q",
        "-p",
        "no:cacheprovider",
        f"--basetemp={work / 'pytest'}",
        f"--junitxml={evidence / 'tests.xml'}",
    ]
    run = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    (evidence / "tests.log").write_text(run.stdout + run.stderr, encoding="utf-8")
    shutil.copyfile(router / "manifest.json", evidence / "router-manifest.json")
    shutil.copyfile(config, evidence / "router-selection.json")
    # Keep compiler provenance and captured device symbols, without shipping CUBINs.
    for pattern in ("artifact.json", "*.dot", "selection.json", "*-build.log"):
        for path in (work / "pytest").rglob(pattern):
            target = evidence / "artifacts" / path.relative_to(work / "pytest")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    suites = list(ET.parse(evidence / "tests.xml").getroot().iter("testsuite"))
    counts = {
        key: sum(int(s.get(key, 0)) for s in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    sources = [
        Path(__file__),
        Path(__file__).with_name("elementwise_factory.py"),
        Path(__file__).with_name("export_router.py"),
        ROOT / "tokenspeed-kernel/python/tokenspeed_kernel/cute_dsl_compile.py",
        ROOT / "tokenspeed-kernel/python/tokenspeed_kernel/cute_dsl_runtime.py",
        ROOT / "tokenspeed-kernel/python/tokenspeed_kernel/cute_dsl_aot.py",
        ROOT / "tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/cute_dsl_aot.py",
        ROOT / "tokenspeed-kernel/python/tokenspeed_kernel/ops/gemm/__init__.py",
        source,
        *(ROOT / test for test in tests),
    ]
    if args.suite in ("host", "all"):
        sources.extend(
            [
                Path(__file__).with_name("host_factory.py"),
                ROOT / "tokenspeed-kernel/python/tokenspeed_kernel/cute_dsl_host.py",
                ROOT
                / "tokenspeed-kernel/python/tokenspeed_kernel/cute_dsl_host_export.py",
                ROOT
                / "tokenspeed-kernel/python/tokenspeed_kernel/_cute_dsl_native.cpp",
                ROOT
                / "tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cute_dsl/ll_bf16/_windows_host.py",
                ROOT / "tokenspeed-kernel/python/setup.py",
            ]
        )
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Windows x64 compiler under system emulation; native ARM64 device execution and configured router auto dispatch; no WSL or model throughput claim",
        "execution_python_pe_machine": hex(pe_machine(Path(sys.executable))),
        "compiler_python_pe_machine": hex(pe_machine(compiler)),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "test_exit_code": run.returncode,
        "suite": args.suite,
        "native_host_dlls": {
            p.relative_to(work).as_posix(): hex(pe_machine(p))
            for p in (work / "pytest").rglob("host.dll")
            if "native" in p.parts
        },
        **counts,
        "source_sha256": {
            str(p.relative_to(ROOT))
            .replace("\\", "/"): hashlib.sha256(p.read_bytes())
            .hexdigest()
            for p in sources
        },
    }
    (evidence / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    checksums = [
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(evidence).as_posix()}"
        for p in sorted(evidence.rglob("*"))
        if p.is_file()
    ]
    (evidence / "checksums.sha256").write_text(
        "\n".join(checksums) + "\n", encoding="utf-8"
    )
    print(run.stdout)
    print(f"Evidence: {evidence}")
    raise SystemExit(
        run.returncode or int(any(counts[k] for k in ("failures", "errors", "skipped")))
    )


if __name__ == "__main__":
    main()
