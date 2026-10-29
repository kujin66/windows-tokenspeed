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

"""Measure warm, fixed-length Engine requests on Windows ARM64 or WSL2 ARM64."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCHEMA = 2


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def summarize(samples):
    if not samples or any(
        s["seconds"] <= 0 or s["output_tokens"] <= 0 for s in samples
    ):
        raise ValueError("Positive durations and output counts are required")
    seconds = sorted(s["seconds"] for s in samples)
    total_tokens = sum(s["output_tokens"] for s in samples)
    return {
        "rounds": len(samples),
        "median_seconds": statistics.median(seconds),
        "p95_seconds": seconds[math.ceil(0.95 * len(seconds)) - 1],
        "mean_seconds": statistics.mean(seconds),
        "output_tokens_per_second": total_tokens / sum(seconds),
        "total_output_tokens": total_tokens,
        "total_seconds": sum(seconds),
    }


def verify_outputs(results, batch_size, output_len):
    if isinstance(results, dict):
        results = [results]
    if len(results) != batch_size:
        raise ValueError(f"Expected batch size {batch_size}, got {len(results)}")
    ids = [r["output_ids"] for r in results]
    if any(len(row) != output_len for row in ids):
        raise ValueError(
            f"Expected {output_len} output tokens per request, got {[len(row) for row in ids]}"
        )
    if any(r["meta_info"]["completion_tokens"] != output_len for r in results):
        raise ValueError("Completion metadata disagrees with requested length")
    return ids


def check_consistency(reference, actual, *, policy):
    if policy not in ("record", "require"):
        raise ValueError(f"Unknown consistency policy: {policy}")
    consistent = reference == actual
    if not consistent and policy == "require":
        raise ValueError("Greedy output differs from the reference request")
    return consistent


def git(root, *args):
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root), *args],
        text=True,
    ).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--upstream-base", required=True)
    parser.add_argument(
        "--platform", choices=("windows-arm64", "wsl2-arm64"), required=True
    )
    parser.add_argument("--attention-backend", choices=("triton",), required=True)
    parser.add_argument(
        "--consistency-policy", choices=("record", "require"), required=True
    )
    parser.add_argument("--input-lengths", type=int, nargs="+", required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", required=True)
    parser.add_argument("--output-length", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if (
        min(
            args.input_lengths
            + args.batch_sizes
            + [args.output_length, args.warmup, args.rounds]
        )
        <= 0
    ):
        parser.error("Lengths, batch sizes, warmup and rounds must be positive")
    if platform.machine().lower() not in ("arm64", "aarch64"):
        parser.error("An ARM64 interpreter is required")
    if args.platform == "windows-arm64" and sys.platform != "win32":
        parser.error("windows-arm64 requires Windows")
    if args.platform == "wsl2-arm64" and "microsoft" not in platform.release().lower():
        parser.error("wsl2-arm64 requires a WSL kernel")
    model = args.model.resolve()
    source = args.source_root.resolve()
    git(source, "merge-base", "--is-ancestor", args.upstream_base, "HEAD")
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    weights = sorted(model.glob("*.safetensors"))
    if not weights:
        parser.error("Expected local safetensors checkpoint")
    model_files = {p.name: sha256(p) for p in [model / "config.json", *weights]}
    # Reserve one prefix page for the scheduler/decode window beyond the workload.
    max_len = max(args.input_lengths) + args.output_length + 64
    engine_config = {
        "seed": 20261028,
        "dtype": "bfloat16",
        "world_size": 1,
        "max_model_len": max_len,
        "max_num_seqs": max(args.batch_sizes),
        "max_total_tokens": max(512, max_len * max(args.batch_sizes)),
        "gpu_memory_utilization": 0.25,
        "attention_backend": args.attention_backend,
        "enforce_eager": True,
        "disable_prefill_graph": True,
        "grammar_backend": "none",
        "sampling_backend": "greedy",
        "disable_kvstore": True,
        "retraction_snapshot_ratio": 0,
        "enable_prefix_caching": False,
        "log_level": "warning",
    }
    workload = {
        "input_lengths": args.input_lengths,
        "batch_sizes": args.batch_sizes,
        "output_length": args.output_length,
        "warmup": args.warmup,
        "rounds": args.rounds,
        "consistency_policy": args.consistency_policy,
        "seed": 20261028,
        "engine": engine_config,
        "sampling": {
            "temperature": 0,
            "max_new_tokens": args.output_length,
            "ignore_eos": True,
        },
        "timing": "blocking Engine.generate wall time, includes prefill and decode; excludes model load, warmup and output verification",
    }
    import torch

    from tokenspeed.runtime.entrypoints.engine import Engine

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable")
    dependencies = {}
    for name in (
        "tokenspeed",
        "tokenspeed-engine",
        "tokenspeed-scheduler",
        "tokenspeed-kernel",
        "torch",
        "triton-windows",
        "triton",
        "tokenspeed-triton",
        "transformers",
        "apache-tvm-ffi",
        "flashinfer-python",
    ):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    props = torch.cuda.get_device_properties(0)
    report = {
        "schema": SCHEMA,
        "platform": args.platform,
        "captured_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "source_commit": git(source, "rev-parse", "HEAD"),
        "upstream_base": args.upstream_base,
        "source_status": git(source, "status", "--porcelain"),
        "benchmark_sha256": sha256(Path(__file__)),
        "environment": {
            "os": platform.system(),
            "kernel": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch_cuda": torch.version.cuda,
            "dependencies": dependencies,
        },
        "gpu": {
            "name": props.name,
            "sm": [props.major, props.minor],
            "total_memory": props.total_memory,
        },
        "model_files": model_files,
        "quantization": config.get("quantization_config"),
        "workload": workload,
        "cases": [],
        "passed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    engine = Engine(model=str(model), **engine_config)
    try:
        for input_len in args.input_lengths:
            for batch_size in args.batch_sizes:
                rng = random.Random(20261028 + input_len * 100 + batch_size)
                inputs = [
                    [
                        rng.randrange(1000, min(32000, config["vocab_size"]))
                        for _ in range(input_len)
                    ]
                    for _ in range(batch_size)
                ]
                reference = None
                samples = []
                case = {
                    "input_length": input_len,
                    "batch_size": batch_size,
                    "input_ids_sha256": hashlib.sha256(
                        json.dumps(inputs).encode()
                    ).hexdigest(),
                    "output_ids": None,
                    "warmup_output_ids": [],
                    "samples": samples,
                    "greedy_output_ids_consistent": True,
                }
                report["cases"].append(case)
                for _ in range(args.warmup):
                    result = engine.generate(
                        input_ids=inputs, sampling_params=workload["sampling"]
                    )
                    ids = verify_outputs(result, batch_size, args.output_length)
                    case["warmup_output_ids"].append(ids)
                    if reference is None:
                        reference = ids
                        case["output_ids"] = reference
                    case["greedy_output_ids_consistent"] &= check_consistency(
                        reference, ids, policy=args.consistency_policy
                    )
                for _ in range(args.rounds):
                    start = time.perf_counter()
                    result = engine.generate(
                        input_ids=inputs, sampling_params=workload["sampling"]
                    )
                    elapsed = time.perf_counter() - start
                    ids = verify_outputs(result, batch_size, args.output_length)
                    samples.append(
                        {
                            "seconds": elapsed,
                            "output_tokens": batch_size * args.output_length,
                            "output_ids": ids,
                        }
                    )
                    case["greedy_output_ids_consistent"] &= check_consistency(
                        reference, ids, policy=args.consistency_policy
                    )
                case["summary"] = summarize(samples)
                args.report.write_text(
                    json.dumps(report, indent=2) + "\n", encoding="utf-8"
                )
                print(
                    f"input={input_len} batch={batch_size}: {case['summary']['output_tokens_per_second']:.2f} output tok/s",
                    flush=True,
                )
        report["passed"] = True
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except Exception as error:
        report["error"] = str(error)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        raise
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
