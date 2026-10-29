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

"""Measure Transformers generation without importing TokenSpeed packages."""

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

import torch
from transformers import (
    AutoModelForCausalLM,
    DynamicCache,
    GenerationConfig,
    StaticCache,
)


def assert_no_tokenspeed():
    loaded = sorted(name for name in sys.modules if name.startswith("tokenspeed"))
    if loaded:
        raise RuntimeError(
            f"No-TokenSpeed baseline imported unexpected modules: {loaded}"
        )


def prompt_ids(length, batch):
    rng = random.Random(20261028 + length * 100 + batch)
    return [[rng.randrange(1000, 32000) for _ in range(length)] for _ in range(batch)]


def validate_output(ids, batch, output_length):
    if len(ids) != batch or any(len(row) != output_length for row in ids):
        raise ValueError("Wrong generated batch or output length")
    if any(
        not isinstance(token, int) or not 0 <= token < 151936
        for row in ids
        for token in row
    ):
        raise ValueError("Invalid generated token ID")


def replace_rmsnorm(model):
    import flashinfer
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

    class FlashInferRMSNorm(Qwen3RMSNorm):
        def forward(self, hidden_states):
            shape = hidden_states.shape
            result = flashinfer.rmsnorm(
                hidden_states.reshape(-1, shape[-1]),
                self.weight,
                eps=self.variance_epsilon,
            )
            return result.reshape(shape)

    replaced = []
    with torch.inference_mode():
        for name, old in list(model.named_modules()):
            if not isinstance(old, Qwen3RMSNorm):
                continue
            replacement = FlashInferRMSNorm(
                old.weight.numel(), eps=old.variance_epsilon
            )
            replacement.weight = old.weight
            sample = torch.randn(
                4, old.weight.numel(), dtype=torch.bfloat16, device="cuda"
            )
            expected = (
                sample.float()
                * torch.rsqrt(
                    sample.float().square().mean(-1, keepdim=True)
                    + old.variance_epsilon
                )
                * old.weight.float()
            ).to(sample.dtype)
            actual = replacement(sample)
            torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
            replaced.append(
                dict(
                    name=name,
                    width=old.weight.numel(),
                    fp32_max_abs_error=(actual.float() - expected.float())
                    .abs()
                    .max()
                    .item(),
                    original_hf_max_abs_difference=(
                        actual.float() - old(sample).float()
                    )
                    .abs()
                    .max()
                    .item(),
                )
            )
            model.set_submodule(name, replacement)
    return dict(
        flashinfer_version=importlib.metadata.version("flashinfer-python"),
        flashinfer_module=flashinfer.__file__,
        module_count=len(replaced),
        modules=replaced,
        scope="Only Qwen3 RMSNorm replaced; attention remains PyTorch SDPA; MLP and weights unchanged",
    )


class StaticRunner:
    """Use the unmodified HF model/cache with an explicit greedy decode loop."""

    def __init__(self, model, input_length, batch, output_length, capacity, execution):
        self.model = model
        self.input_length = input_length
        self.batch = batch
        self.output_length = output_length
        self.execution = execution
        self.capacity = capacity
        self.cache = StaticCache(config=model.config, max_cache_len=capacity)
        self.positions = torch.arange(capacity, device="cuda")
        self.prefill_positions = torch.arange(input_length, device="cuda").unsqueeze(0)
        self.prefill_mask = (
            (self.positions.unsqueeze(0) <= self.prefill_positions.t())
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.token = torch.empty((batch, 1), dtype=torch.long, device="cuda")
        self.position = torch.full(
            (1, 1), input_length, dtype=torch.long, device="cuda"
        )
        self.output_index = torch.ones(1, dtype=torch.long, device="cuda")
        self.output = torch.empty(
            (batch, output_length), dtype=torch.long, device="cuda"
        )
        self.graph = None
        self.graph_validation = None

    def prefill(self, inputs):
        self.cache.reset()
        ids = torch.tensor(inputs, dtype=torch.long, device="cuda")
        result = self.model(
            input_ids=ids,
            attention_mask={"full_attention": self.prefill_mask},
            position_ids=self.prefill_positions,
            past_key_values=self.cache,
            use_cache=True,
            logits_to_keep=1,
        )
        first = result.logits[:, -1].argmax(dim=-1, keepdim=True)
        self.token.copy_(first)
        self.output[:, :1].copy_(first)
        self.position.fill_(self.input_length)
        self.output_index.fill_(1)

    def step(self):
        mask = (self.positions.unsqueeze(0) <= self.position).unsqueeze(0).unsqueeze(0)
        result = self.model(
            input_ids=self.token,
            attention_mask={"full_attention": mask},
            position_ids=self.position,
            past_key_values=self.cache,
            use_cache=True,
            logits_to_keep=1,
        )
        token = result.logits[:, -1].argmax(dim=-1, keepdim=True)
        self.output.index_copy_(1, self.output_index, token)
        self.token.copy_(token)
        self.position.add_(1)
        self.output_index.add_(1)

    def generate(self, inputs):
        self.prefill(inputs)
        for _ in range(self.output_length - 1):
            if self.execution == "graph":
                self.graph.replay()
            else:
                self.step()
        return self.output.cpu().tolist()

    def prepare(self, inputs):
        self.prefill(inputs)
        self.step()
        if self.execution == "graph":
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.prefill(inputs)
                    self.step()
            torch.cuda.current_stream().wait_stream(stream)
            self.prefill(inputs)
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                self.step()
            graph_ids = self.generate(inputs)
            self.execution = "eager"
            eager_ids = self.generate(inputs)
            self.execution = "graph"
            equal = graph_ids == eager_ids
            lengths = [
                int(layer.get_seq_length().item()) for layer in self.cache.layers
            ]
            expected = self.input_length + self.output_length - 1
            if not equal or any(length != expected for length in lengths):
                raise ValueError(
                    f"Graph/eager validation failed: equal={equal}, cache_lengths={lengths}"
                )
            self.graph_validation = dict(
                graph_ids=graph_ids,
                eager_ids=eager_ids,
                exact_match=equal,
                cache_lengths=lengths,
                expected_cache_length=expected,
            )


def dynamic_loop(model, inputs, output_length):
    ids = torch.tensor(inputs, dtype=torch.long, device="cuda")
    cache = DynamicCache(config=model.config)
    result = model(
        input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1
    )
    token = result.logits[:, -1].argmax(dim=-1, keepdim=True)
    output = [token]
    for _ in range(output_length - 1):
        result = model(
            input_ids=token, past_key_values=cache, use_cache=True, logits_to_keep=1
        )
        token = result.logits[:, -1].argmax(dim=-1, keepdim=True)
        output.append(token)
    return torch.cat(output, dim=1).cpu().tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--implementation",
        choices=("generate", "loop-dynamic", "loop-static-eager", "loop-static-graph"),
        required=True,
    )
    parser.add_argument(
        "--normalization", choices=("torch", "flashinfer"), required=True
    )
    parser.add_argument("--diagnostics", choices=("none", "cpu"), required=True)
    parser.add_argument("--input-lengths", type=int, nargs="+", required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", required=True)
    parser.add_argument("--output-length", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--capacity", type=int, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(f"Preserve existing report: {args.report}")
    assert_no_tokenspeed()
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.manual_seed(20261028)
    model_path = args.model.resolve()
    files = [model_path / "config.json", *sorted(model_path.glob("*.safetensors"))]
    hashes = {}
    for path in files:
        with path.open("rb") as stream:
            hashes[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    report = dict(
        captured_at=datetime.now(timezone.utc).isoformat(),
        implementation=args.implementation,
        normalization=args.normalization,
        platform=platform.platform(),
        python=platform.python_version(),
        torch=torch.__version__,
        torch_cuda=torch.version.cuda,
        transformers=importlib.metadata.version("transformers"),
        gpu=torch.cuda.get_device_name(0),
        dtype="bfloat16",
        attention_backend="sdpa",
        torch_num_threads=torch.get_num_threads(),
        model_files=hashes,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        seed=20261028,
        warmup=args.warmup,
        rounds=args.rounds,
        output_length=args.output_length,
        capacity=args.capacity,
        input_lengths=args.input_lengths,
        batch_sizes=args.batch_sizes,
        passed=False,
        cases=[],
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    save()
    try:
        model = (
            AutoModelForCausalLM.from_pretrained(
                str(model_path),
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                local_files_only=True,
            )
            .to("cuda")
            .eval()
        )
        if args.normalization == "flashinfer":
            if not args.implementation.startswith("loop-static"):
                raise ValueError("RMSNorm ablation requires the static-cache control")
            report["rmsnorm_ablation"] = replace_rmsnorm(model)
        config = GenerationConfig(
            do_sample=False,
            temperature=None,
            top_k=None,
            top_p=None,
            min_new_tokens=args.output_length,
            max_new_tokens=args.output_length,
            eos_token_id=None,
            pad_token_id=0,
            use_cache=True,
            cache_implementation="dynamic",
            disable_compile=True,
        )
        report["generation_config"] = config.to_dict()
        report["model_class"] = f"{type(model).__module__}.{type(model).__name__}"
        report["actual_attention_backend"] = model.config._attn_implementation
        with torch.inference_mode():
            for length in args.input_lengths:
                for batch in args.batch_sizes:
                    if length + args.output_length > args.capacity:
                        raise ValueError("Cache capacity is too small")
                    inputs = prompt_ids(length, batch)
                    case = dict(
                        input_length=length,
                        batch_size=batch,
                        input_ids=inputs,
                        warmup_ids=[],
                        samples=[],
                    )
                    report["cases"].append(case)
                    runner = None
                    if args.implementation.startswith("loop-static"):
                        execution = (
                            "graph"
                            if args.implementation.endswith("graph")
                            else "eager"
                        )
                        runner = StaticRunner(
                            model,
                            length,
                            batch,
                            args.output_length,
                            args.capacity,
                            execution,
                        )
                        runner.prepare(inputs)
                        case["graph_validation"] = runner.graph_validation

                    def generate():
                        if runner is not None:
                            result = runner.generate(inputs)
                        elif args.implementation == "loop-dynamic":
                            result = dynamic_loop(model, inputs, args.output_length)
                        else:
                            ids = torch.tensor(inputs, dtype=torch.long, device="cuda")
                            outputs = model.generate(
                                input_ids=ids,
                                attention_mask=torch.ones_like(ids),
                                generation_config=config,
                                eos_token_id=None,
                                do_sample=False,
                            )
                            result = outputs[:, length:].cpu().tolist()
                        validate_output(result, batch, args.output_length)
                        return result

                    for _ in range(args.warmup):
                        case["warmup_ids"].append(generate())
                    for _ in range(args.rounds):
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        result = generate()
                        elapsed = time.perf_counter() - start
                        case["samples"].append(dict(seconds=elapsed, output_ids=result))
                    case["tokens_per_second"] = (
                        batch
                        * args.output_length
                        * args.rounds
                        / sum(sample["seconds"] for sample in case["samples"])
                    )
                    all_ids = case["warmup_ids"] + [
                        sample["output_ids"] for sample in case["samples"]
                    ]
                    case["repeat_consistent"] = all(
                        ids == all_ids[0] for ids in all_ids
                    )
                    print(
                        f"{args.implementation} input={length} batch={batch}: {case['tokens_per_second']:.2f} tok/s consistent={case['repeat_consistent']}",
                        flush=True,
                    )
                    save()
                    del runner
                    torch.cuda.empty_cache()
            if args.diagnostics == "cpu":
                length, batch = args.input_lengths[0], args.batch_sizes[0]
                inputs = prompt_ids(length, batch)
                diagnostic_runner = None
                if args.implementation.startswith("loop-static"):
                    execution = (
                        "graph" if args.implementation.endswith("graph") else "eager"
                    )
                    diagnostic_runner = StaticRunner(
                        model,
                        length,
                        batch,
                        args.output_length,
                        args.capacity,
                        execution,
                    )
                    diagnostic_runner.prepare(inputs)

                def diagnostic_generate():
                    if diagnostic_runner is not None:
                        return diagnostic_runner.generate(inputs)
                    if args.implementation == "loop-dynamic":
                        return dynamic_loop(model, inputs, args.output_length)
                    ids = torch.tensor(inputs, dtype=torch.long, device="cuda")
                    return (
                        model.generate(
                            input_ids=ids,
                            attention_mask=torch.ones_like(ids),
                            generation_config=config,
                            eos_token_id=None,
                            do_sample=False,
                        )[:, length:]
                        .cpu()
                        .tolist()
                    )

                diagnostic_generate()
                torch.cuda.synchronize()
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU],
                    record_shapes=False,
                ) as profile:
                    diagnostic_ids = diagnostic_generate()
                report["cpu_diagnostic"] = dict(
                    input_length=length,
                    batch_size=batch,
                    output_ids=diagnostic_ids,
                    included_in_throughput=False,
                    gpu_kernel_time_collected=False,
                    events=[
                        dict(
                            name=event.key,
                            count=event.count,
                            self_cpu_time_us=event.self_cpu_time_total,
                        )
                        for event in profile.key_averages()
                    ],
                )
        assert_no_tokenspeed()
        report["tokenspeed_modules_loaded"] = []
        report["passed"] = True
        save()
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        save()
        raise


if __name__ == "__main__":
    main()
