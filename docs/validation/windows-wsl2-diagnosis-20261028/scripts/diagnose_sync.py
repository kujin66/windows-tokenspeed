"""Measure per-step synchronization separately from batched launch throughput."""
import argparse
import hashlib
import json
import platform
import queue
import statistics
import threading
import time
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report = dict(platform=platform.platform(), torch=torch.__version__, cuda=torch.version.cuda, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), cases=[], passed=False)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    def run(name, fn):
        for _ in range(20):
            fn()
        values = []
        for _ in range(7):
            start = time.perf_counter_ns()
            for _ in range(100):
                fn()
            values.append((time.perf_counter_ns() - start) / 100 / 1000)
        report['cases'].append(dict(name=name, iterations=100, rounds=7, samples_us=values, median_us=statistics.median(values)))
        print(f'{name}: {statistics.median(values):.3f} us', flush=True)
    x = torch.ones(4096, device='cuda')
    out = torch.empty_like(x)
    host = torch.empty(4096, pin_memory=True)
    event = torch.cuda.Event()
    torch.cuda.synchronize()
    run('idle_device_synchronize', torch.cuda.synchronize)
    def empty_event():
        event.record()
        event.synchronize()
    run('empty_event_record_synchronize', empty_event)
    def add_event():
        torch.add(x, 1, out=out)
        empty_event()
    run('add_event_synchronize', add_event)
    def add_copy_event():
        torch.add(x, 1, out=out)
        host.copy_(out, non_blocking=True)
        empty_event()
    run('add_pinned_d2h_event_synchronize', add_copy_event)
    torch.testing.assert_close(host, torch.full_like(host, 2))
    commands, responses = queue.Queue(), queue.Queue()
    def echo():
        while True:
            command = commands.get()
            if command is None:
                return
            responses.put(command)
    worker = threading.Thread(target=echo)
    worker.start()
    def ping():
        commands.put(1)
        assert responses.get(timeout=10) == 1
    try:
        run('python_queue_thread_roundtrip', ping)
    finally:
        commands.put(None)
        worker.join(timeout=10)
    def cuda_worker():
        torch.cuda.set_device(0)
        worker_event = torch.cuda.Event()
        while True:
            if commands.get() is None:
                return
            torch.add(x, 1, out=out)
            host.copy_(out, non_blocking=True)
            worker_event.record()
            responses.put(worker_event)
    worker = threading.Thread(target=cuda_worker)
    worker.start()
    def ping_cuda():
        commands.put(1)
        responses.get(timeout=10).synchronize()
    try:
        run('thread_add_pinned_d2h_event_synchronize', ping_cuda)
    finally:
        commands.put(None)
        worker.join(timeout=10)
    assert not worker.is_alive()
    torch.testing.assert_close(host, torch.full_like(host, 2))
    report['passed'] = True
    args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
