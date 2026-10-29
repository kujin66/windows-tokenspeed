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

"""Matched fixed-length text-output generation through two independent engines."""

import argparse
import hashlib
import importlib.metadata as md
import inspect
import json
import os
import platform
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, choices in [
        ("engine", ("tokenspeed", "vllm")),
        ("backend", ("triton", "flashinfer", "auto")),
        ("platform", ("windows-arm64", "wsl2-arm64")),
        ("execution", ("graph", "eager")),
    ]:
        parser.add_argument("--" + name, choices=choices, required=True)
    for name in ("model", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("input-lengths", "batch-sizes"):
        parser.add_argument("--" + name, nargs="+", type=int, required=True)
    for name in ("output-length", "warmup", "rounds"):
        parser.add_argument("--" + name, type=int, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(f"Preserve previous run: {args.report}")
    if platform.machine().lower() not in ("arm64", "aarch64"):
        parser.error("Native ARM64 Python required")
    if args.platform == "windows-arm64" and sys.platform != "win32":
        parser.error("Expected native Windows")
    if args.platform == "wsl2-arm64" and "microsoft" not in platform.release().lower():
        parser.error("Expected WSL2 Linux")
    if (
        min(
            args.input_lengths
            + args.batch_sizes
            + [args.output_length, args.rounds, args.warmup]
        )
        < 1
    ):
        parser.error("Counts must be positive")
    if max(args.input_lengths) + args.output_length > 704 or max(args.batch_sizes) > 4:
        parser.error("Explicit limits: 704 total tokens and batch 4")
    if (args.engine == "vllm") != (args.backend == "auto"):
        parser.error(
            "vLLM uses auto; TokenSpeed explicitly selects triton or flashinfer"
        )
    model = args.model.resolve()
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3":
        parser.error("This cohort requires dense Qwen3")
    report = dict(
        schema=1,
        captured_at=datetime.now(timezone.utc).isoformat(),
        engine_name=args.engine,
        backend=args.backend,
        execution=args.execution,
        platform=args.platform,
        os=platform.platform(),
        machine=platform.machine(),
        python=platform.python_version(),
        benchmark_sha256=digest(__file__),
        seed=20261028,
        input_lengths=args.input_lengths,
        batch_sizes=args.batch_sizes,
        output_length=args.output_length,
        warmup=args.warmup,
        rounds=args.rounds,
        model_files={
            p.name: digest(p)
            for p in [model / "config.json", *sorted(model.glob("*.safetensors"))]
        },
        aot_router=dict(
            active=False,
            reason="The AOT helper is not registered in the production engine. Dense Qwen3-0.6B has no MoE router or matching [M,7168] x [896,7168] projection.",
        ),
        timing="Blocking text-output generate wall time including prefill, decode and immediate ID/count checks; loading, warmup and model hashing excluded",
        cases=[],
        passed=False,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    save()
    engine = None
    try:
        import torch

        torch.cuda.set_device(0)
        torch.set_num_threads(1)
        props = torch.cuda.get_device_properties(0)
        report["gpu"] = dict(
            name=props.name,
            sm=[props.major, props.minor],
            multiprocessors=props.multi_processor_count,
        )
        report["torch_cuda"] = torch.version.cuda
        report["packages"] = {}
        for name in (
            "torch",
            "vllm",
            "tokenspeed",
            "tokenspeed-kernel",
            "tokenspeed-engine",
            "tokenspeed-scheduler",
            "flashinfer-python",
            "transformers",
            "triton",
            "triton-windows",
            "tokenspeed-triton",
            "apache-tvm-ffi",
            "nvidia-cutlass-dsl",
        ):
            try:
                report["packages"][name] = md.version(name)
            except md.PackageNotFoundError:
                pass
        if args.engine == "tokenspeed":
            import tokenspeed_kernel.ops.activation as activation
            import tokenspeed_kernel.ops.attention.mha as attention
            import tokenspeed_kernel.ops.layernorm as norm

            from tokenspeed.runtime.entrypoints.engine import Engine

            report["sources"] = {}
            for name, item in (
                ("Engine", Engine),
                ("layernorm", norm),
                ("activation", activation),
                ("attention_mha", attention),
            ):
                path = Path(inspect.getfile(item))
                report["sources"][name] = dict(path=str(path), sha256=digest(path))
            report["source_root_override"] = os.environ.get("PYTHONPATH", "")
            settings = dict(
                seed=20261028,
                dtype="bfloat16",
                world_size=1,
                max_model_len=704,
                max_num_seqs=4,
                max_total_tokens=2816,
                gpu_memory_utilization=0.25,
                attention_backend=args.backend,
                enforce_eager=args.execution == "eager",
                disable_prefill_graph=True,
                grammar_backend="none",
                sampling_backend="greedy",
                disable_kvstore=True,
                retraction_snapshot_ratio=0,
                enable_prefix_caching=False,
                log_level="warning",
            )
            if args.execution == "graph":
                settings.update(
                    cudagraph_capture_sizes=[1, 4], max_cudagraph_capture_size=4
                )
            sampling = dict(
                temperature=0, max_new_tokens=args.output_length, ignore_eos=True
            )
            report["engine_config"], report["sampling"] = settings, sampling
            report["graph_policy"] = (
                "decode capture [1,4], prefill graphs disabled"
                if args.execution == "graph"
                else "eager"
            )
            save()
            engine = Engine(model=str(model), **settings)

            def generate(inputs):
                outputs = engine.generate(input_ids=inputs, sampling_params=sampling)
                outputs = outputs if isinstance(outputs, list) else [outputs]
                ids = [row["output_ids"] for row in outputs]
                if len(ids) != len(inputs) or any(
                    len(row) != args.output_length for row in ids
                ):
                    raise ValueError("Wrong request/output count")
                if any(
                    row["meta_info"]["completion_tokens"] != args.output_length
                    for row in outputs
                ):
                    raise ValueError("Completion metadata mismatch")
                return ids

        else:
            from vllm import LLM, SamplingParams

            settings = dict(
                model=str(model),
                dtype="bfloat16",
                tensor_parallel_size=1,
                seed=20261028,
                max_model_len=704,
                max_num_seqs=4,
                max_num_batched_tokens=2816,
                gpu_memory_utilization=0.03,
                kv_cache_memory_bytes=335544320,
                enable_prefix_caching=False,
                enable_chunked_prefill=True,
                enforce_eager=args.execution == "eager",
                skip_tokenizer_init=False,
                generation_config="vllm",
                disable_log_stats=True,
            )
            if args.execution == "graph":
                settings["compilation_config"] = dict(
                    cudagraph_capture_sizes=[1, 4], max_cudagraph_capture_size=4
                )
            sampling = dict(
                temperature=0,
                max_tokens=args.output_length,
                ignore_eos=True,
                detokenize=True,
            )
            report["engine_config"], report["sampling"] = settings, sampling
            report["plugins"] = [
                dict(name=e.name, value=e.value)
                for e in md.entry_points(group="vllm.general_plugins")
            ]
            save()
            engine = LLM(**settings)
            actual = engine.llm_engine.vllm_config
            report["resolved"] = dict(
                dtype=str(actual.model_config.dtype),
                quantization=actual.model_config.quantization,
                enforce_eager=actual.model_config.enforce_eager,
                compilation_mode=str(actual.compilation_config.mode),
                cudagraph_mode=str(actual.compilation_config.cudagraph_mode),
                capture_sizes=actual.compilation_config.cudagraph_capture_sizes,
                attention_config=str(actual.attention_config),
            )
            if (
                args.execution == "graph"
                and report["resolved"]["cudagraph_mode"] == "CUDAGraphMode.NONE"
            ):
                raise RuntimeError("vLLM silently disabled CUDA graphs")
            parameters = SamplingParams(**sampling)

            def generate(inputs):
                outputs = engine.generate(
                    [dict(prompt_token_ids=ids) for ids in inputs],
                    parameters,
                    use_tqdm=False,
                )
                if len(outputs) != len(inputs):
                    raise ValueError("Wrong request count")
                ids = []
                for prompt, output in zip(inputs, outputs, strict=True):
                    if output.prompt_token_ids != prompt or len(output.outputs) != 1:
                        raise ValueError("Wrong request order/count")
                    row = list(output.outputs[0].token_ids)
                    if len(row) != args.output_length:
                        raise ValueError("Wrong output length")
                    ids.append(row)
                return ids

        for length in args.input_lengths:
            for batch in args.batch_sizes:
                rng = random.Random(report["seed"] + length * 100 + batch)
                inputs = [
                    [rng.randrange(1000, 32000) for _ in range(length)]
                    for _ in range(batch)
                ]
                case = dict(
                    input_length=length,
                    batch_size=batch,
                    input_ids=inputs,
                    warmup_ids=[],
                    samples=[],
                )
                report["cases"].append(case)
                for _ in range(args.warmup):
                    case["warmup_ids"].append(generate(inputs))
                for _ in range(args.rounds):
                    start = time.perf_counter()
                    ids = generate(inputs)
                    case["samples"].append(
                        dict(seconds=time.perf_counter() - start, output_ids=ids)
                    )
                durations = [s["seconds"] for s in case["samples"]]
                case["tokens_per_second"] = (
                    batch * args.output_length * args.rounds / sum(durations)
                )
                case["median_seconds"] = statistics.median(durations)
                all_ids = case["warmup_ids"] + [
                    s["output_ids"] for s in case["samples"]
                ]
                case["repeat_consistent"] = all(ids == all_ids[0] for ids in all_ids)
                print(
                    f'{args.platform} {args.engine}/{args.backend} {args.execution} input={length} batch={batch}: {case["tokens_per_second"]:.2f} tok/s repeat={case["repeat_consistent"]}',
                    flush=True,
                )
                save()
        if args.engine == "vllm":
            loaded = [
                n
                for n in sys.modules
                if n == "tokenspeed" or n.startswith("tokenspeed.")
            ]
            report["tokenspeed_engine_modules_loaded"] = loaded
            if loaded:
                raise RuntimeError(f"vLLM imported the TokenSpeed engine: {loaded}")
        report["passed"] = True
        save()
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        save()
        raise
    finally:
        if engine is not None:
            if args.engine == "tokenspeed":
                engine.shutdown()
            else:
                engine.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
