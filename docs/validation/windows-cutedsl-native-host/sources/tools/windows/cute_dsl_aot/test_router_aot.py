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

"""Real GPU correctness and paired non-eager latency measurements."""

import argparse
import inspect
import hashlib
import importlib.util
import json
import platform
import statistics
import tempfile
from pathlib import Path

import torch
from windows_router_aot import RouterAOT


def capture(fn, count):
    warm = torch.cuda.Stream()
    warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
        for _ in range(10):
            fn()
    torch.cuda.current_stream().wait_stream(warm)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(count):
            fn()
    return graph


def graph_times(graphs, *, nodes, rounds, replays):
    samples = {name: [] for name in graphs}
    for graph in graphs.values():
        for _ in range(20):
            graph.replay()
    torch.cuda.synchronize()
    for trial in range(rounds):
        names = list(graphs)
        if trial % 2:
            names.reverse()
        for name in names:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            start.record()
            for _ in range(replays):
                graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / (nodes * replays))
    return {
        name: {
            "median_us": statistics.median(values),
            "min_us": min(values),
            "max_us": max(values),
            "samples_us": values,
        }
        for name, values in samples.items()
    }


def require_value_error(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("Invalid input was accepted")


def confirm_tuning(directory, results, *, rounds, nodes, replays):
    confirmation = []
    for m in (1, 2, 4):
        candidates = [r for r in results if r["shape_mnk"] == [m, 896, 7168]]
        chosen = min(candidates, key=lambda r: r["timing"]["cute_aot"]["median_us"])
        chosen_meta = next(
            item
            for item in json.loads((directory / "manifest.json").read_text())["kernels"]
            if item["name"] == chosen["name"]
        )
        bs = chosen_meta["block"][0]
        tuned = RouterAOT(
            directory,
            shape_mnk=(m, 896, 7168),
            block_size=bs,
            device=torch.device("cuda:0"),
        )
        original = RouterAOT(
            directory,
            shape_mnk=(m, 896, 7168),
            block_size=256,
            device=torch.device("cuda:0"),
        )
        a = torch.randn(m, 7168, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(896, 7168, dtype=torch.bfloat16, device="cuda")
        output = [
            torch.empty(m, 896, dtype=torch.float32, device="cuda") for _ in range(3)
        ]
        try:
            graphs = {
                "tuned_aot": capture(lambda: tuned(a, b, out=output[0]), nodes),
                "original_aot_bs256": capture(
                    lambda: original(a, b, out=output[1]), nodes
                ),
                "torch_cublas": capture(
                    lambda: torch.mm(a, b.t(), out_dtype=torch.float32, out=output[2]),
                    nodes,
                ),
            }
            for graph in graphs.values():
                graph.replay()
            reference = (a.cpu().double() @ b.cpu().double().t()).float()
            for value in output:
                torch.testing.assert_close(value.cpu(), reference, atol=1e-3, rtol=1e-4)
            timing = graph_times(graphs, nodes=nodes, rounds=rounds, replays=replays)
            row = {
                "shape_mnk": [m, 896, 7168],
                "selected_block_size": bs,
                "correctness_passed": True,
                "timing": timing,
                "cublas_over_tuned": timing["torch_cublas"]["median_us"]
                / timing["tuned_aot"]["median_us"],
                "original_over_tuned": timing["original_aot_bs256"]["median_us"]
                / timing["tuned_aot"]["median_us"],
            }
            confirmation.append(row)
            print(
                f"Confirmation M={m}, block={bs}: cuBLAS/AOT={row['cublas_over_tuned']:.3f}, bs256/AOT={row['original_over_tuned']:.3f}",
                flush=True,
            )
            del graphs
        finally:
            tuned.close()
            original.close()
    return confirmation


def run(directory, *, rounds, nodes, replays, confirm):
    assert platform.machine().lower() in {"arm64", "aarch64"}
    assert (
        importlib.util.find_spec("cutlass") is None
    ), "Test must prove execution without native cutlass"
    torch.cuda.set_device(0)
    torch.manual_seed(20261029)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    results = []
    for item in manifest["kernels"]:
        m, n, k = item["shape_mnk"]
        loader = RouterAOT(
            directory,
            shape_mnk=(m, n, k),
            block_size=item["block"][0],
            device=torch.device("cuda:0"),
        )
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(m, n, device="cuda", dtype=torch.float32)
        base_out = torch.empty_like(out)

        def aot():
            return loader(a, b, out=out)

        def baseline():
            return torch.mm(a, b.t(), out_dtype=torch.float32, out=base_out)

        try:
            correctness_graph = capture(aot, 1)
            baseline_graph = capture(baseline, 1)
            max_abs = 0.0
            for seed in range(3):
                a.normal_()
                b.normal_()
                reference = (a.cpu().double() @ b.cpu().double().t()).float()
                out.fill_(float("nan"))
                aot()
                torch.testing.assert_close(out.cpu(), reference, atol=1e-3, rtol=1e-4)
                out.fill_(float("nan"))
                correctness_graph.replay()
                baseline_graph.replay()
                torch.testing.assert_close(out.cpu(), reference, atol=1e-3, rtol=1e-4)
                torch.testing.assert_close(
                    base_out.cpu(), reference, atol=1e-3, rtol=1e-4
                )
                max_abs = max(max_abs, (out.cpu() - reference).abs().max().item())
            require_value_error(lambda: loader(a.float(), b, out=out))
            require_value_error(lambda: loader(a, b[:, :-1], out=out))
            bad = torch.empty(m, n * 2, device="cuda")[:, ::2]
            require_value_error(lambda: loader(a, b, out=bad))
            alias = (
                a.view(torch.float32).reshape(-1)[: m * n].view(m, n)
                if k >= 2 * n
                else b.view(torch.float32).reshape(-1)[: m * n].view(m, n)
            )
            require_value_error(lambda: loader(a, b, out=alias))
            other = torch.cuda.Stream()
            other.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(other):
                out.fill_(float("nan"))
                aot()
            torch.cuda.current_stream().wait_stream(other)
            torch.testing.assert_close(out.cpu(), reference, atol=1e-3, rtol=1e-4)
            graphs = {
                "cute_aot": capture(aot, nodes),
                "torch_cublas": capture(baseline, nodes),
            }
            timing = graph_times(graphs, nodes=nodes, rounds=rounds, replays=replays)
            speedup = (
                timing["torch_cublas"]["median_us"] / timing["cute_aot"]["median_us"]
            )
            row = {
                "name": item["name"],
                "shape_mnk": [m, n, k],
                "passed": True,
                "ordinary_input_sets": 3,
                "changed_input_graph_replays": 3,
                "invalid_input_checks": 4,
                "non_default_stream": True,
                "max_abs_error": max_abs,
                "timing": timing,
                "cublas_over_aot": speedup,
            }
            results.append(row)
            print(
                f'{item["name"]}: CuTe AOT {timing["cute_aot"]["median_us"]:.3f} us; cuBLAS {timing["torch_cublas"]["median_us"]:.3f} us; ratio {speedup:.3f}',
                flush=True,
            )
            del graphs, correctness_graph, baseline_graph
        finally:
            loader.close()
    require_value_error(
        lambda: RouterAOT(
            directory,
            shape_mnk=(3, 896, 7168),
            block_size=256,
            device=torch.device("cuda:0"),
        )
    )
    # Artifact corruption must fail before loading GPU code.
    first = manifest["kernels"][0]
    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        broken = Path(temporary)
        (broken / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (broken / (first["name"] + ".cubin")).write_bytes(b"corrupt")
        require_value_error(
            lambda: RouterAOT(
                broken,
                shape_mnk=tuple(first["shape_mnk"]),
                block_size=first["block"][0],
                device=torch.device("cuda:0"),
            )
        )
    confirmation = (
        confirm_tuning(directory, results, rounds=rounds, nodes=nodes, replays=replays)
        if confirm
        else []
    )
    report = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "torch_cuda": torch.version.cuda,
        "cutlass_python_available": False,
        "source_sha256": manifest["source_sha256"],
        "cutedsl_export_version": manifest["cutedsl"],
        "method": {
            "mode": "CUDA graph replay (non-eager)",
            "operations_per_graph": nodes,
            "timed_replays_per_round": replays,
            "paired_rounds": rounds,
            "order": "alternating",
            "warmup_graph_replays": 20,
            "timing": "CUDA events; microseconds per operation",
            "cache": "hot repeated weights, no L2 flush",
            "baseline": "torch.mm(BF16 A, BF16 B.T, out_dtype=torch.float32, out=preallocated)",
            "input_and_output_allocation": "outside capture/timing",
            "compilation_and_module_load": "outside capture/timing",
            "correctness": "CPU FP64 from BF16 inputs; atol=1e-3, rtol=1e-4",
            "bf16_reduced_precision_reduction": False,
        },
        "cases": results,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "loader_sha256": hashlib.sha256(
            Path(__file__).with_name("windows_router_aot.py").read_bytes()
        ).hexdigest(),
        "packaged_loader_sha256": hashlib.sha256(
            Path(inspect.getfile(RouterAOT)).read_bytes()
        ).hexdigest(),
        "tuning_confirmation": confirmation,
        "unsupported_shape_rejected": True,
        "corrupt_artifact_rejected": True,
        "scope": "Direct AOT operation microbenchmark; not an engine/model throughput result or an automatic-dispatch test. Registry integration is validated separately.",
    }
    (directory / "windows-results.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument(
        "--operations", "--nodes", dest="nodes", type=int, required=True
    )
    parser.add_argument("--replays", type=int, required=True)
    parser.add_argument("--confirm-tuning", choices=("yes", "no"), required=True)
    args = parser.parse_args()
    if min(args.rounds, args.nodes, args.replays) < 1:
        parser.error("Benchmark counts must be positive")
    run(
        args.directory,
        rounds=args.rounds,
        nodes=args.nodes,
        replays=args.replays,
        confirm=args.confirm_tuning == "yes",
    )
