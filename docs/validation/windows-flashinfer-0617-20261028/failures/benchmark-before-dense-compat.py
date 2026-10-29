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

"""Time warm inference and capture a separate scheduler-side trace."""
import argparse
import hashlib
import importlib.metadata
import itertools
import os
import sys
import json
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from tokenspeed.runtime.entrypoints.engine import Engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--execution', choices=('eager', 'graph'), required=True)
    parser.add_argument('--profile', choices=('CPU,GPU', 'none'), required=True)
    parser.add_argument('--attention-backend', choices=('triton', 'flashinfer'), required=True)
    parser.add_argument('--input-lengths', type=int, nargs='+', required=True)
    args = parser.parse_args()
    engine_config = dict(
        seed=20261028, dtype='bfloat16', world_size=1, max_model_len=704,
        max_num_seqs=4, max_total_tokens=2816, gpu_memory_utilization=0.25,
        attention_backend=args.attention_backend, enforce_eager=args.execution == 'eager',
        disable_prefill_graph=True, grammar_backend='none', sampling_backend='greedy',
        disable_kvstore=True, retraction_snapshot_ratio=0, enable_prefix_caching=False,
        log_level='warning',
    )
    if args.execution == 'graph':
        engine_config.update(cudagraph_capture_sizes=[1, 4], max_cudagraph_capture_size=4)
    sampling = dict(temperature=0, max_new_tokens=128, ignore_eos=True)
    import flashinfer
    import torch
    import tokenspeed_kernel.ops.layernorm as norm
    import tokenspeed_kernel.ops.activation as activation
    model = args.model.resolve()
    model_files = [model / 'config.json', *sorted(model.glob('*.safetensors'))]
    assert len(model_files) >= 2
    model_hashes = {}
    for path in model_files:
        with path.open('rb') as stream:
            model_hashes[path.name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    operation_sources = {}
    for name, module in [('layernorm', norm), ('activation', activation)]:
        path = Path(module.__file__).resolve()
        operation_sources[name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if sys.platform == 'win32':
        repo = Path(__file__).resolve().parents[1]
        if os.environ['TOKENSPEED_COMPARISON_OPS'] == 'flashinfer':
            expected = repo / '.venv/flashinfer0617-kernel'
        else:
            expected = repo / '.venv/Lib/site-packages'
        assert all(Path(item['path']).is_relative_to(expected) for item in operation_sources.values())
        assert importlib.metadata.version('flashinfer-python') == '0.6.17+g2ea3c24b5dac'
    report = dict(
        captured_at=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        engine=engine_config, sampling=sampling, input_lengths=args.input_lengths,
        model_files=model_hashes, ops=os.environ.get('TOKENSPEED_COMPARISON_OPS', 'official'),
        operation_sources=operation_sources, flashinfer=dict(version=importlib.metadata.version('flashinfer-python'), module=flashinfer.__file__),
        torch=dict(version=torch.__version__, cuda=torch.version.cuda), gpu=torch.cuda.get_device_name(0),
        warmup=3, rounds=5, profile=args.profile, cases=[], passed=False,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    save()
    engine = None
    try:
        engine = Engine(model=str(model), **engine_config)
        for input_length, batch in itertools.product(args.input_lengths, (1, 4)):
            rng = random.Random(20261028 + input_length * 100 + batch)
            inputs = [[rng.randrange(1000, 32000) for _ in range(input_length)] for _ in range(batch)]
            case = dict(input_length=input_length, batch_size=batch, samples=[], warmup_ids=[], profile_error=None)
            report['cases'].append(case)
            def generate():
                result = engine.generate(input_ids=inputs, sampling_params=sampling)
                outputs = result if isinstance(result, list) else [result]
                ids = [row['output_ids'] for row in outputs]
                assert len(ids) == batch and all(len(row) == 128 for row in ids)
                assert all(row['meta_info']['completion_tokens'] == 128 for row in outputs)
                return ids
            for _ in range(3):
                case['warmup_ids'].append(generate())
            for _ in range(5):
                start = time.perf_counter()
                ids = generate()
                case['samples'].append(dict(seconds=time.perf_counter() - start, output_ids=ids))
            case['tokens_per_second'] = batch * 128 * 5 / sum(row['seconds'] for row in case['samples'])
            print(f'input={input_length} batch={batch}: {case["tokens_per_second"]:.2f} tok/s', flush=True)
            save()
            if args.profile == 'CPU,GPU':
                try:
                    result = engine.llm.run(engine.tokenizer_manager.start_profile(
                        output_dir=str((args.report.parent / f'trace-b{batch}').resolve()),
                        activities=['CPU', 'GPU'], with_stack=False, record_shapes=False,
                        profile_id=f'{args.execution}-b{batch}',
                    ))
                except Exception as exc:
                    case['profile_error'] = str(exc)
                    print(f'Profiler unavailable: {exc}', flush=True)
                else:
                    case['profile_output_ids'] = generate()
                    engine.stop_profile()
            save()
        report['passed'] = True
        save()
    except Exception as exc:
        report['error'] = str(exc)
        save()
        raise
    finally:
        if engine is not None:
            engine.shutdown()


if __name__ == '__main__':
    main()
