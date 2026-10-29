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

"""Measure native vLLM generation without importing or calling TokenSpeed."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def assert_no_tokenspeed():
    loaded = [name for name in sys.modules if name.startswith("tokenspeed")]
    if loaded:
        raise RuntimeError(f"vLLM baseline imported TokenSpeed: {loaded}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--execution", choices=("eager", "graph"), required=True)
    parser.add_argument("--attention", choices=("auto", "TRITON_ATTN"), required=True)
    parser.add_argument(
        "--graph-policy", choices=("default", "decode-only"), required=True
    )
    parser.add_argument("--output-policy", choices=("tokens", "text"), required=True)
    parser.add_argument("--memory-utilization", type=float, required=True)
    parser.add_argument("--input-lengths", type=int, nargs="+", required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", required=True)
    parser.add_argument("--output-length", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(f"Preserve existing report: {args.report}")
    assert_no_tokenspeed()
    report = dict(
        captured_at=datetime.now(timezone.utc).isoformat(),
        engine_name="vllm",
        tokenspeed_integration=False,
        execution=args.execution,
        attention_requested=args.attention,
        graph_policy=args.graph_policy,
        output_policy=args.output_policy,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        platform=platform.platform(),
        python=platform.python_version(),
        input_lengths=args.input_lengths,
        batch_sizes=args.batch_sizes,
        output_length=args.output_length,
        seed=20261028,
        warmup=args.warmup,
        rounds=args.rounds,
        passed=False,
        cases=[],
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    save()
    llm = None
    try:
        import torch
        from vllm import LLM, SamplingParams

        torch.cuda.set_device(0)
        report["torch"] = dict(version=torch.__version__, cuda=torch.version.cuda)
        report["gpu"] = torch.cuda.get_device_name(0)
        report["packages"] = (
            {
                name: importlib.metadata.version(name)
                for name in (
                    "vllm",
                    "transformers",
                    "flashinfer-python",
                    "triton-windows",
                )
            }
            if sys.platform == "win32"
            else {
                name: importlib.metadata.version(name)
                for name in ("vllm", "transformers", "flashinfer-python", "triton")
            }
        )
        report["general_plugins"] = [
            dict(name=entry.name, value=entry.value)
            for entry in importlib.metadata.entry_points(group="vllm.general_plugins")
        ]
        model = args.model.resolve()
        report["model_files"] = {}
        for path in [model / "config.json", *sorted(model.glob("*.safetensors"))]:
            with path.open("rb") as stream:
                report["model_files"][path.name] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
        configuration = dict(
            model=str(model),
            dtype="bfloat16",
            tensor_parallel_size=1,
            seed=20261028,
            max_model_len=704,
            max_num_seqs=4,
            max_num_batched_tokens=2816,
            gpu_memory_utilization=args.memory_utilization,
            kv_cache_memory_bytes=335544320,
            enable_prefix_caching=False,
            enable_chunked_prefill=True,
            enforce_eager=args.execution == "eager",
            skip_tokenizer_init=args.output_policy == "tokens",
            generation_config="vllm",
            disable_log_stats=True,
        )
        if args.attention != "auto":
            configuration["attention_config"] = dict(backend=args.attention)
        if args.execution == "graph":
            compilation = dict(
                cudagraph_capture_sizes=[1, 4], max_cudagraph_capture_size=4
            )
            if args.graph_policy == "decode-only":
                compilation["cudagraph_mode"] = "FULL_DECODE_ONLY"
                compilation["mode"] = 0
            configuration["compilation_config"] = compilation
        sampling = dict(
            temperature=0,
            max_tokens=args.output_length,
            ignore_eos=True,
            detokenize=args.output_policy == "text",
        )
        report["engine"] = configuration
        report["sampling"] = sampling
        save()
        llm = LLM(**configuration)
        config = llm.llm_engine.vllm_config
        report["actual_engine_class"] = (
            f"{type(llm.llm_engine).__module__}.{type(llm.llm_engine).__name__}"
        )
        report["resolved"] = dict(
            dtype=str(config.model_config.dtype),
            quantization=config.model_config.quantization,
            enforce_eager=config.model_config.enforce_eager,
            prefix_caching=config.cache_config.enable_prefix_caching,
            compilation_mode=str(config.compilation_config.mode),
            cudagraph_mode=str(config.compilation_config.cudagraph_mode),
            capture_sizes=config.compilation_config.cudagraph_capture_sizes,
            attention_config=str(config.attention_config),
        )
        graph_mode = report["resolved"]["cudagraph_mode"].removeprefix("CUDAGraphMode.")
        if args.execution == "graph" and graph_mode not in (
            "FULL",
            "PIECEWISE",
            "FULL_DECODE_ONLY",
            "FULL_AND_PIECEWISE",
        ):
            raise RuntimeError("Requested vLLM graph mode was silently disabled")
        params = SamplingParams(**sampling)
        for length in args.input_lengths:
            for batch in args.batch_sizes:
                if length + args.output_length > 704 or batch > 4:
                    raise ValueError("Workload exceeds explicit engine limits")
                rng = random.Random(20261028 + length * 100 + batch)
                inputs = [
                    [rng.randrange(1000, 32000) for _ in range(length)]
                    for _ in range(batch)
                ]
                prompts = [dict(prompt_token_ids=ids) for ids in inputs]
                case = dict(
                    input_length=length,
                    batch_size=batch,
                    input_ids=inputs,
                    warmup_ids=[],
                    samples=[],
                )
                report["cases"].append(case)

                def generate():
                    outputs = llm.generate(prompts, params, use_tqdm=False)
                    if len(outputs) != batch:
                        raise ValueError("Wrong returned batch")
                    result = []
                    for prompt, output in zip(inputs, outputs, strict=True):
                        if (
                            output.prompt_token_ids != prompt
                            or len(output.outputs) != 1
                        ):
                            raise ValueError("Returned request order/sequence mismatch")
                        ids = list(output.outputs[0].token_ids)
                        if len(ids) != args.output_length:
                            raise ValueError("Wrong returned output length")
                        result.append(ids)
                    return result

                for _ in range(args.warmup):
                    case["warmup_ids"].append(generate())
                for _ in range(args.rounds):
                    start = time.perf_counter()
                    ids = generate()
                    case["samples"].append(
                        dict(seconds=time.perf_counter() - start, output_ids=ids)
                    )
                case["tokens_per_second"] = (
                    batch
                    * args.output_length
                    * args.rounds
                    / sum(sample["seconds"] for sample in case["samples"])
                )
                all_ids = case["warmup_ids"] + [
                    sample["output_ids"] for sample in case["samples"]
                ]
                case["repeat_consistent"] = all(ids == all_ids[0] for ids in all_ids)
                print(
                    f"vLLM {args.execution} attention={args.attention} input={length} batch={batch}: {case['tokens_per_second']:.2f} tok/s consistent={case['repeat_consistent']}",
                    flush=True,
                )
                save()
        assert_no_tokenspeed()
        report["tokenspeed_modules_loaded"] = []
        report["passed"] = True
        save()
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        save()
        raise
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
