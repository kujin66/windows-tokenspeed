import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root / '.port/flashinfer-comparison'
commands = []
for ops in ('flashinfer', 'triton'):
    attention = 'flashinfer' if ops == 'flashinfer' else 'triton'
    for execution in ('eager', 'graph'):
        report = out / f'windows-{ops}-{execution}.json'
        command = [sys.executable, str(root / 'tools/windows/run.py'), sys.executable,
                   str(root / '.port/launch_flashinfer0617.py'), '--ops', ops,
                   str(root / '.port/flashinfer_comparison_bench.py'),
                   '--model', str(root / '.port/models/Qwen3-0.6B'), '--report', str(report),
                   '--execution', execution, '--profile', 'none', '--attention-backend', attention,
                   '--input-lengths', '128', '512']
        commands.append(command)
        (out / 'windows-commands.json').write_text(json.dumps(commands, indent=2) + '\n', encoding='utf-8')
        with report.with_suffix('.log').open('wb') as log:
            result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise SystemExit(result.returncode)
        data = json.loads(report.read_text(encoding='utf-8'))
        assert data['passed']
        print(json.dumps(dict(completed=f'{ops}-{execution}', cases=[dict(input=c['input_length'], batch=c['batch_size'], tps=c['tokens_per_second']) for c in data['cases']])), flush=True)
