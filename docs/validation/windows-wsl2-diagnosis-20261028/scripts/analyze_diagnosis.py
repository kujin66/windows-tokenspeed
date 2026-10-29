import collections
import bisect
import gzip
import json
import statistics
from pathlib import Path

root = Path(__file__).resolve().parents[1]
port = root / '.port/perf-diagnosis'

def trace_summary(path):
    trace = json.loads(gzip.decompress(path.read_bytes()))
    events = trace['traceEvents']
    categories = collections.Counter()
    names = collections.defaultdict(lambda: collections.Counter())
    durations = collections.defaultdict(list)
    grouped = collections.defaultdict(list)
    calls = collections.defaultdict(collections.Counter)
    for event in events:
        if event.get('ph') != 'X':
            continue
        cat = event.get('cat', 'uncategorized')
        categories[cat] += 1
        names[cat][event['name']] += event.get('dur', 0)
        durations[cat].append(event.get('dur', 0))
        grouped[cat].append(event)
        calls[cat][event['name']] += 1
    kernels = grouped['kernel']
    begin = min((e['ts'] for e in kernels), default=0)
    end = max((e['ts'] + e['dur'] for e in kernels), default=0)
    intervals = sorted((e['ts'], e['ts'] + e['dur']) for e in kernels)
    union = 0
    last_end = float('-inf')
    for start, finish in intervals:
        union += max(0, finish - max(start, last_end))
        last_end = max(last_end, finish)
    phases = [e for e in grouped['user_annotation'] if "'mode':'DECODE'" in e['name']]
    api = sorted(grouped['cuda_runtime'] + grouped['cuda_driver'], key=lambda e: e['ts'])
    starts = [e['ts'] for e in api]
    decode = []
    for phase in phases:
        selected = api[bisect.bisect_left(starts, phase['ts']):bisect.bisect_right(starts, phase['ts'] + phase['dur'])]
        selected = [e for e in selected if e['tid'] == phase['tid'] and 'LaunchKernel' in e['name']]
        spans = sorted((e['ts'], e['ts'] + e['dur']) for e in selected)
        active = 0
        last = float('-inf')
        for start, finish in spans:
            active += max(0, finish - max(start, last))
            last = max(last, finish)
        decode.append(dict(forward_us=phase['dur'], launch_api_union_us=active, launch_callback_count=len(selected)))
    return {
        'file': path.relative_to(port).as_posix(), 'categories': dict(categories),
        'gpu_kernel_activities_available': bool(kernels),
        'kernel_span_us': end - begin if kernels else None, 'kernel_union_us': union if kernels else None,
        'kernel_busy_fraction': union / (end - begin) if end > begin else None,
        'category_summed_us': {cat: sum(values) for cat, values in durations.items()},
        'name_counts': {cat: dict(values) for cat, values in calls.items()},
        'decode': dict(count=len(decode), median={key: statistics.median(row[key] for row in decode) for key in decode[0]}, samples=decode) if decode else None,
        'top_names_by_summed_us': {cat: values.most_common(16) for cat, values in names.items()},
    }

traces = [trace_summary(path) for path in port.rglob('*.trace.json.gz')]
(port / 'trace-analysis.json').write_text(json.dumps(traces, indent=2) + '\n', encoding='utf-8')
for trace in traces:
    print(trace['file'], 'GPU kernel span/busy us', trace['kernel_span_us'], trace['kernel_union_us'])
    for category in ('kernel', 'cuda_runtime', 'cuda_driver', 'user_annotation'):
        print(category, trace['top_names_by_summed_us'].get(category, [])[:10])
windows = json.loads((port / 'windows-micro.json').read_text(encoding='utf-8'))
linux = json.loads((port / 'wsl2-micro.json').read_text(encoding='utf-8'))
assert windows['passed'] and linux['passed']
assert windows['script_sha256'] == linux['script_sha256']
assert [c['name'] for c in windows['cases']] == [c['name'] for c in linux['cases']]
for w, l in zip(windows['cases'], linux['cases'], strict=True):
    print(w['name'], 'eager us', round(w['median']['wall_us_per_op'], 2), round(l['median']['wall_us_per_op'], 2), 'graph us', round(w['graph_median']['wall_us_per_op'], 2), round(l['graph_median']['wall_us_per_op'], 2))
