"""Exploratory same-script microbenchmarks; no TokenSpeed runtime changes."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from tokenspeed_kernel.ops.activation import silu_and_mul
from tokenspeed_kernel.ops.activation.triton import silu_and_mul as triton_silu
from tokenspeed_kernel.ops.layernorm import rmsnorm
from tokenspeed_kernel.ops.layernorm.triton import rmsnorm as triton_norm


def measure(fn, iterations, graph_nodes):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    samples = []
    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    for _ in range(7):
        torch.cuda.synchronize()
        start.record()
        wall = time.perf_counter_ns()
        cpu = time.thread_time_ns()
        for _ in range(iterations):
            fn()
        enqueue = time.perf_counter_ns() - wall
        cpu = time.thread_time_ns() - cpu
        end.record()
        end.synchronize()
        samples.append({
            'wall_us_per_op': (time.perf_counter_ns() - wall) / iterations / 1000,
            'enqueue_wall_us_per_op': enqueue / iterations / 1000,
            'enqueue_thread_cpu_us_per_op': cpu / iterations / 1000,
            'stream_span_us_per_op': start.elapsed_time(end) * 1000 / iterations,
        })
    graph_samples = []
    graph_error = None
    try:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(graph_nodes):
                fn()
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        for _ in range(7):
            start.record()
            wall = time.perf_counter_ns()
            for _ in range(100):
                graph.replay()
            end.record()
            end.synchronize()
            graph_samples.append({
                'wall_us_per_op': (time.perf_counter_ns() - wall) / 100 / graph_nodes / 1000,
                'stream_span_us_per_op': start.elapsed_time(end) * 1000 / 100 / graph_nodes,
            })
    except Exception as exc:
        graph_error = str(exc)
    return {
        'iterations': iterations, 'rounds': 7, 'samples': samples,
        'median': {key: statistics.median(s[key] for s in samples) for key in samples[0]},
        'graph_nodes': graph_nodes, 'graph_replays_per_round': 100,
        'graph_samples': graph_samples, 'graph_error': graph_error,
        'graph_median': {key: statistics.median(s[key] for s in graph_samples) for key in graph_samples[0]} if graph_samples else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(20261028)
    torch.set_num_threads(1)
    deps = {}
    for name in ('torch', 'tokenspeed-kernel', 'triton-windows', 'tokenspeed-triton', 'flashinfer-python'):
        try:
            deps[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    report = {
        'captured_at': datetime.now(timezone.utc).isoformat(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'platform': platform.platform(), 'machine': platform.machine(),
        'torch_cuda': torch.version.cuda, 'dependencies': deps,
        'torch_config': torch.__config__.show(),
        'gpu': torch.cuda.get_device_name(0),
        'timing_note': 'Eager stream span includes GPU idle gaps caused by host submission. Graph timing amortizes graph launch across fixed nodes; it is not pure isolated kernel duration.',
        'cases': [], 'passed': False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)

    def run(name, fn, *, iterations=200, graph_nodes=32):
        value = measure(fn, iterations, graph_nodes)
        value['name'] = name
        report['cases'].append(value)
        args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(f'{name}: eager {value["median"]["wall_us_per_op"]:.3f} us; graph {value["graph_median"]}', flush=True)

    with torch.inference_mode():
        x = torch.randn(4096, device='cuda')
        out = torch.empty_like(x)
        run('torch_add_4096', lambda: torch.add(x, 1, out=out))
        x = torch.randn(16 * 1024 * 1024, device='cuda')
        out = torch.empty_like(x)
        run('device_copy_64MiB', lambda: out.copy_(x), iterations=30, graph_nodes=8)
        for batch in (1, 4):
            x = torch.randn(batch, 1024, dtype=torch.bfloat16, device='cuda')
            weight = torch.randn(1024, dtype=torch.bfloat16, device='cuda')
            out = torch.empty_like(x)
            reference = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6) * weight.float()).to(x.dtype)
            for name, function in (('facade', rmsnorm), ('forced_triton', triton_norm)):
                actual = function(x, weight, 1e-6, out=out)
                torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.02)
                run(f'rmsnorm_{name}_m{batch}_n1024', lambda function=function: function(x, weight, 1e-6, out=out))
            x = torch.randn(batch, 6144, dtype=torch.bfloat16, device='cuda')
            out = torch.empty(batch, 3072, dtype=x.dtype, device='cuda')
            a, b = x.float().chunk(2, dim=-1)
            reference = (torch.nn.functional.silu(a) * b).to(x.dtype)
            for name, function in (('facade', silu_and_mul), ('forced_triton', triton_silu)):
                actual = function(x, out=out)
                torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.02)
                run(f'silu_{name}_m{batch}_n6144', lambda function=function: function(x, out=out))
            for k, n in ((1024, 4096), (1024, 6144), (3072, 1024), (1024, 151936)):
                x = torch.randn(batch, k, dtype=torch.bfloat16, device='cuda')
                weight = torch.randn(n, k, dtype=x.dtype, device='cuda')
                out = torch.empty(batch, n, dtype=x.dtype, device='cuda')
                fn = lambda: torch.mm(x, weight.T, out=out)
                fn()
                expected = x.float() @ weight[:16].float().T
                torch.testing.assert_close(out[:, :16].float(), expected, rtol=0.03, atol=0.15)
                run(f'bf16_mm_m{batch}_k{k}_n{n}', fn, iterations=60 if n > 10000 else 200)
    report['passed'] = True
    args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
