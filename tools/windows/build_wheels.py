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

"""Build the four project wheels in a prepared native ARM64 venv."""

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import runpy
import shlex
import shutil
import struct
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def stage_git_source(root: Path, destination: Path, *, paths: tuple[str, ...]) -> None:
    """Extract the tracked source revision for Windows package versioning."""
    archive = subprocess.check_output(
        ["git", "-c", f"safe.directory={root.as_posix()}", "archive", "HEAD", *paths],
        cwd=root,
    )
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        # Match core.symlinks=false checkouts. tarfile creates file symlinks
        # on Windows even when the target is a directory; Rust cannot resolve
        # that reparse point. The package build already handles link-text files.
        members = bundle.getmembers()
        bundle.extractall(
            destination,
            members=[item for item in members if not item.issym()],
            filter="data",
        )
        for item in members:
            if item.issym():
                validated = tarfile.data_filter(item, str(destination))
                link = destination / validated.name
                link.parent.mkdir(parents=True, exist_ok=True)
                link.write_text(item.linkname, encoding="utf-8")


def prepare_runtime_documents(source: Path) -> None:
    # Match the upstream runtime build without modifying tracked checkout files.
    for name in ("README.md", "LICENSE"):
        shutil.copyfile(source / name, source / "python" / name)


def stage_runtime_source(root: Path, destination: Path) -> None:
    destination.mkdir()
    stage_git_source(root, destination, paths=("python", "README.md", "LICENSE"))
    prepare_runtime_documents(destination)


def run_build_command(
    command: tuple[str, ...], *, cwd: Path, env: dict[str, str]
) -> None:
    """Retry only Windows Rust temporary-archive sharing violations, at most twice."""
    for attempt in range(3):
        result = subprocess.run(
            command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        sys.stdout.buffer.write(result.stdout)
        sys.stdout.buffer.flush()
        if result.returncode == 0:
            return
        archive_lock = os.name == "nt" and all(
            message in result.stdout
            for message in (
                b"failed to remove temporary directory",
                b"temp-archive",
                b"(os error 32)",
            )
        )
        if (
            not archive_lock
            or b"error[E" in result.stdout
            or b"test result: FAILED" in result.stdout
            or attempt == 2
        ):
            raise subprocess.CalledProcessError(
                result.returncode, command, output=result.stdout
            )
        print(
            f"Windows Rust temporary archive is locked; retrying ({attempt + 2}/3).",
            flush=True,
        )
        time.sleep(attempt + 1)


def pe_machine(contents: bytes) -> int:
    """Read the COFF machine field; reject truncated or non-PE inputs."""
    if len(contents) < 64 or contents[:2] != b"MZ":
        raise ValueError("Missing DOS header")
    offset = struct.unpack_from("<I", contents, 60)[0]
    if offset + 6 > len(contents) or contents[offset : offset + 4] != b"PE\0\0":
        raise ValueError("Missing PE signature")
    return struct.unpack_from("<H", contents, offset + 4)[0]


def audit_wheel(path: Path) -> dict:
    """Verify every native payload, including PE libraries named .so."""
    with zipfile.ZipFile(path) as archive:
        native = [
            n for n in archive.namelist() if n.endswith((".pyd", ".dll", ".exe", ".so"))
        ]
        for name in native:
            if pe_machine(archive.read(name)) != 0xAA64:
                raise ValueError(f"Not ARM64: {path.name}:{name}")
        if native and not path.name.endswith("-win_arm64.whl"):
            raise ValueError(
                f"Native payload in an incorrectly tagged wheel: {path.name}"
            )
    return {
        "wheel": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "native_files": native,
    }


def command_main(arguments: list[str]) -> None:
    """Run a native command with the same bounded archive-lock recovery."""
    parser = argparse.ArgumentParser(description=command_main.__doc__)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(arguments)
    if not args.command:
        parser.error("Supply a native build or test command")
    run_build_command(tuple(args.command), cwd=ROOT, env=dict(os.environ))


COMPONENTS = {
    "tokenspeed-scheduler": "tokenspeed-scheduler",
    "tokenspeed-engine": "tokenspeed-engine",
    "tokenspeed-kernel": "tokenspeed-kernel/python",
    "tokenspeed": "python",
}


def revision() -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
    ).strip()


def build_environment(context: dict, output: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        PYTHONPATH="",
        PYO3_PYTHON=sys.executable,
        CMAKE_GENERATOR="Ninja",
        CUDA_HOME=context["cuda_home"],
        TOKENSPEED_KERNEL_BACKEND="cuda",
        TOKENSPEED_KERNEL_GIT_SHA=context["revision"],
        TOKENSPEED_CUDA_ARCH=context["cuda_arch"],
        FLASHINFER_CUDA_ARCH_LIST=context["cuda_arch"],
        DISTUTILS_USE_SDK="1",
        MSSdk="1",
        TOKENSPEED_KERNEL_NIGHTLY="true" if context["nightly_date"] else "false",
        TOKENSPEED_KERNEL_VERSION_DATE=context["nightly_date"],
        TOKENSPEED_KERNEL_RELEASE_CANDIDATE=context["release_candidate"],
    )
    env.setdefault("CARGO_TARGET_DIR", str(output / "cargo"))
    return env


def prepare_build(output: Path, *, cuda_arch: str, cuda_home: str) -> None:
    date = os.environ.get("TOKENSPEED_WINDOWS_NIGHTLY_DATE", "")
    candidate = os.environ.get("TOKENSPEED_KERNEL_RELEASE_CANDIDATE", "")
    if candidate and not date:
        raise ValueError("A release candidate requires an explicit nightly date")
    output.mkdir(parents=True, exist_ok=False)
    (output / "wheels").mkdir()
    context = dict(
        revision=revision(),
        python=sys.executable,
        cuda_arch=cuda_arch,
        cuda_home=cuda_home,
        nightly_date=date,
        release_candidate=candidate,
        source="root",
        package_versions={},
        completed={},
    )
    if date:
        source = output / "source"
        source.mkdir()
        stage_git_source(ROOT, source, paths=())
        prepare_runtime_documents(source)
        context["source"] = "staged"
        env = build_environment(context, output)
        kernel = (
            subprocess.check_output(
                [sys.executable, "setup.py", "--version"],
                cwd=source / "tokenspeed-kernel/python",
                env=env,
                text=True,
                encoding="utf-8",
            )
            .strip()
            .splitlines()[-1]
        )
        prepare = runpy.run_path(str(ROOT / "tools/windows/prepare_nightly.py"))[
            "prepare"
        ]
        context["package_versions"] = prepare(source, date=date, kernel_version=kernel)
    else:
        stage_runtime_source(ROOT, output / "runtime-source")
    (output / "build-state.json").write_text(
        json.dumps(context, indent=2), encoding="utf-8"
    )
    print(
        f"Prepared project backend inputs for {context['revision']} (nightly={date or 'false'})",
        flush=True,
    )


def load_context(output: Path) -> dict:
    context = json.loads((output / "build-state.json").read_text(encoding="utf-8"))
    if context["revision"] != revision():
        raise ValueError("Build state belongs to another source revision")
    if Path(context["python"]).resolve() != Path(sys.executable).resolve():
        raise ValueError("Component builds must use the prepared interpreter")
    if context["nightly_date"] != os.environ.get("TOKENSPEED_WINDOWS_NIGHTLY_DATE", ""):
        raise ValueError("Component build channel/date changed after preparation")
    if context["release_candidate"] != os.environ.get(
        "TOKENSPEED_KERNEL_RELEASE_CANDIDATE", ""
    ):
        raise ValueError("Component release candidate changed after preparation")
    if context["source"] not in {"root", "staged"}:
        raise ValueError("Invalid prepared source layout")
    return context


def component_command(
    component: str, *, source: Path, output: Path, python: str
) -> tuple[str, ...]:
    if component not in COMPONENTS:
        raise ValueError(f"Unsupported component: {component}")
    command = [
        python,
        "-m",
        "pip",
        "wheel",
        str(source / COMPONENTS[component]),
        "--no-build-isolation",
        "--no-deps",
        "--no-index",
        "--wheel-dir",
        str(output / "wheels"),
    ]
    if component == "tokenspeed-scheduler":
        command.extend(
            [
                f"--config-settings=build-dir={output / 'scheduler'}",
                "--config-settings=cmake.args=-DTOKENSPEED_SCHEDULER_BUILD_TESTS=ON;-DTOKENSPEED_SCHEDULER_BUILD_PYTHON=ON;-DINSTALL_GTEST=OFF",
                "--config-settings=cmake.define.TOKENSPEED_SCHEDULER_BUILD_TESTS=ON",
                "--config-settings=cmake.define.INSTALL_GTEST=OFF",
            ]
        )
    else:
        stage = output / (
            "stage-tokenspeed-kernel-python"
            if component == "tokenspeed-kernel"
            else f"stage-{COMPONENTS[component]}"
        )
        command.extend(
            [
                "--config-settings=--build-option=--keep-temp",
                "--config-settings=--build-option="
                + shlex.quote(f"--bdist-dir={stage.as_posix()}"),
            ]
        )
    return tuple(command)


def build_component(output: Path, *, component: str) -> None:
    context = load_context(output)
    if component in context["completed"]:
        raise ValueError(f"Component already built: {component}")
    source = ROOT if context["source"] == "root" else output / "source"
    if component == "tokenspeed" and context["source"] == "root":
        source = output / "runtime-source"
    command = component_command(
        component, source=source, output=output, python=sys.executable
    )
    print(f"Building {component} through its project backend", flush=True)
    run_build_command(command, cwd=ROOT, env=build_environment(context, output))
    (wheel,) = (output / "wheels").glob(component.replace("-", "_") + "-*.whl")
    context["completed"][component] = audit_wheel(wheel)
    (output / "build-state.json").write_text(
        json.dumps(context, indent=2), encoding="utf-8"
    )


def finalize_build(output: Path) -> None:
    context = load_context(output)
    if set(context["completed"]) != set(COMPONENTS):
        raise ValueError("Every project component must complete before finalization")
    built = sorted((output / "wheels").glob("*.whl"))
    if len(built) != 4:
        raise ValueError("Expected exactly four project wheels")
    audits = [audit_wheel(path) for path in built]
    if sorted(audits, key=lambda a: a["wheel"]) != sorted(
        context["completed"].values(), key=lambda a: a["wheel"]
    ):
        raise ValueError("A completed component wheel changed before finalization")
    native_groups = sorted(
        Path(name).parent.name
        for a in audits
        for name in a["native_files"]
        if "/thirdparty/cuda/objs/" in name
    )
    if len(native_groups) != 24 or len(set(native_groups)) != 24:
        raise ValueError(f"Expected all 24 native CUDA groups, found {native_groups}")
    diff = subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "diff", "HEAD", "--binary"],
        cwd=ROOT,
    )
    report = dict(
        revision=context["revision"],
        nightly_date=context["nightly_date"],
        package_versions=context["package_versions"],
        build_dependencies={
            name: importlib.metadata.version(name)
            for name in ("torch", "apache-tvm-ffi", "tokenspeed-triton")
        },
        tracked_diff_sha256=hashlib.sha256(diff).hexdigest(),
        python=platform.python_version(),
        cuda_arch=context["cuda_arch"],
        kernel_groups=native_groups,
        wheels=audits,
    )
    (output / "build.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    run_build_command(
        (
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            "--force-reinstall",
            *map(str, built),
        ),
        cwd=ROOT,
        env=build_environment(context, output),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adapt project backends for explicit shared-CI component tasks"
    )
    actions = parser.add_subparsers(dest="action", required=True)
    prepare = actions.add_parser("prepare")
    prepare.add_argument("--cuda-arch", required=True)
    build = actions.add_parser("build")
    build.add_argument("--component", choices=tuple(COMPONENTS), required=True)
    finish = actions.add_parser("finalize")
    for sub in (prepare, build, finish):
        sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "win32" or platform.machine().lower() not in {
        "arm64",
        "aarch64",
    }:
        parser.error("Run with native Windows ARM64 Python")
    if sys.prefix == sys.base_prefix:
        parser.error("Run inside a dedicated venv")
    output = args.output.resolve()
    if args.action == "prepare":
        for command in (
            "cl.exe",
            "clang.exe",
            "link.exe",
            "cargo.exe",
            "cmake.exe",
            "ninja.exe",
            "git.exe",
        ):
            if not shutil.which(command):
                parser.error(f"Missing {command}; activate an ARM64 MSVC environment")
        cuda = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        if not cuda or not (Path(cuda) / "bin/nvcc.exe").is_file():
            parser.error("Set CUDA_HOME or CUDA_PATH to the Windows CUDA SDK")
        prepare_build(output, cuda_arch=args.cuda_arch, cuda_home=cuda)
    elif args.action == "build":
        build_component(output, component=args.component)
    else:
        finalize_build(output)


if __name__ == "__main__":
    if sys.argv[1:2] == ["run"]:
        command_main(sys.argv[2:])
    else:
        main()
