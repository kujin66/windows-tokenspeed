"""Time warm inference and capture a separate scheduler-side trace."""
import argparse
import hashlib
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
    args = parser.parse_args()
    engine_config = dict(
        seed=20261028, dtype='bfloat16', world_size=1, max_model_len=704,
        max_num_seqs=4, max_total_tokens=2816, gpu_memory_utilization=0.25,
        attention_backend='triton', enforce_eager=args.execution == 'eager',
        disable_prefill_graph=True, grammar_backend='none', sampling_backend='greedy',
        disable_kvstore=True, retraction_snapshot_ratio=0, enable_prefix_caching=False,
        log_level='warning',
    )
    if args.execution == 'graph':
        engine_config.update(cudagraph_capture_sizes=[1, 4], max_cudagraph_capture_size=4)
    sampling = dict(temperature=0, max_new_tokens=128, ignore_eos=True)
    report = dict(
        captured_at=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        engine=engine_config, sampling=sampling, input_length=128,
        warmup=3, rounds=5, profile=args.profile, cases=[], passed=False,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    save()
    engine = Engine(model=str(args.model.resolve()), **engine_config)
    try:
        for batch in (1, 4):
            rng = random.Random(20261028 + 128 * 100 + batch)
            inputs = [[rng.randrange(1000, 32000) for _ in range(128)] for _ in range(batch)]
            case = dict(batch_size=batch, samples=[], warmup_ids=[], profile_error=None)
            report['cases'].append(case)
            def generate():
                result = engine.generate(input_ids=inputs, sampling_params=sampling)
                outputs = result if isinstance(result, list) else [result]
                ids = [row['output_ids'] for row in outputs]
                assert len(ids) == batch and all(len(row) == 128 for row in ids)
                return ids
            for _ in range(3):
                case['warmup_ids'].append(generate())
            for _ in range(5):
                start = time.perf_counter()
                ids = generate()
                case['samples'].append(dict(seconds=time.perf_counter() - start, output_ids=ids))
            case['tokens_per_second'] = batch * 128 * 5 / sum(row['seconds'] for row in case['samples'])
            print(f'batch={batch}: {case["tokens_per_second"]:.2f} tok/s', flush=True)
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
        engine.shutdown()


if __name__ == '__main__':
    main()
