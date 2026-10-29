import hashlib
import json
import shutil
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(repo/'tools/perf'))
from compare_engine_matrix import CELLS,compare

run = repo/'.port/engine-matrix-retest-20261029'
native_outcomes = json.loads((run/'native-owner/windows-outcomes.json').read_text())
retry_outcome = json.loads((run/'wsl-vllm-retry-outcome.json').read_text())
linux_outcomes = json.loads((run/'wsl-outcomes.json').read_text())
if len(native_outcomes)!=3 or not all(item['passed'] for item in native_outcomes) or not retry_outcome['passed'] or not all(item['passed'] for item in linux_outcomes[:2]):
    raise RuntimeError('The selected six benchmark processes must finish successfully')
dest = repo/'docs/validation/engine-matrix-retest-20261029'
dest.mkdir(parents=True,exist_ok=True)
for os,engine,backend in CELLS:
    name = f'{os}-{engine}-{backend}-graph'
    source = run/'native-owner' if os=='windows' else run
    for extension in ('.json','.log'):
        selected_name = name+'-pinned' if os=='wsl2' and engine=='vllm' else name
        shutil.copyfile(source/(selected_name+extension),dest/(name+extension))
result = compare(dest)
(dest/'comparisons.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
preliminary = dest/'preliminary'
preliminary.mkdir(exist_ok=True)
for path in run.glob('windows-*-graph.*'):
    shutil.copyfile(path,preliminary/path.name)
shutil.copyfile(run/'windows-outcomes.json',preliminary/'controller-outcomes.json')
restricted_confirm = preliminary/'restricted-confirm'
restricted_confirm.mkdir(exist_ok=True)
for path in (run/'native-confirm').glob('*'):
    if path.is_file():
        shutil.copyfile(path,restricted_confirm/path.name)
for extension in ('.json','.log'):
    shutil.copyfile(run/('wsl2-vllm-auto-graph'+extension),preliminary/('wsl2-vllm-auto-graph-unpinned'+extension))
shutil.copyfile(run/'wsl-outcomes.json',preliminary/'wsl-controller-outcomes.json')
for name in ('wsl-commands.json','wsl-triton-overlay.json','wsl-triton-overlay.patch','wsl-vllm-freeze.txt','wsl-uva-probe.json','wsl-uva-probe.log','wsl-vllm-retry-command.json','wsl-vllm-retry-outcome.json'):
    shutil.copyfile(run/name,dest/name)
for name in ('windows-commands.json','windows-outcomes.json'):
    shutil.copyfile(run/'native-owner'/name,dest/('final-'+name))
for path in (run/'native-owner').glob('*-environment.json'):
    shutil.copyfile(path,dest/path.name)
sources = dest/'sources'
sources.mkdir(exist_ok=True)
for name in ('setup_vllm_wsl_retest.py','run_matrix_windows.py','run_matrix_windows_confirm.py','run_matrix_windows_owner.py','run_checked_benchmark.py','run_matrix_wsl.py','run_matrix_wsl.sh','launch_matrix_vllm.py','launch_flashinfer0617.py','wsl-env.sh','probe_wsl_vllm_uva.py','retry_wsl_vllm.py','write_matrix_overlay_patch.py'):
    shutil.copyfile(repo/'.port'/name,sources/name)
for name in ('benchmark_engine_matrix.py','compare_engine_matrix.py'):
    shutil.copyfile(repo/'tools/perf'/name,sources/name)
for name in ('fi_sm121_fa2.py','prepare_flashinfer_dense_compat.py'):
    shutil.copyfile(repo/'.port'/name,sources/name)
shutil.copyfile(repo/'tools/windows/run.py',sources/'windows_run.py')
shutil.copyfile(Path(__file__),sources/Path(__file__).name)
install = json.loads((run/'wsl-vllm-install.json').read_text())
selected = {x['metadata']['name']:dict(version=x['metadata']['version'],download=x['download_info']) for x in install['install'] if x['metadata']['name'].lower().replace('_','-') in ('vllm','torch','flashinfer-python','apache-tvm-ffi')}
(dest/'wsl-wheel-provenance.json').write_text(json.dumps(selected,indent=2)+'\n')

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(str(v) for v in row)+' |' for row in rows]])

guide = repo/'docs/guides/windows-wsl2-engine-backend-retest.md'
text = guide.read_text(encoding='utf-8')
names = [f'{os}-{engine}-{backend}' for os,engine,backend in CELLS]
text = text.replace('THROUGHPUT_TABLE',table(['Input / batch','Win vLLM','Win TS Triton','Win TS FI','WSL2 vLLM','WSL2 TS Triton','WSL2 TS FI'],[[f'{r["input_length"]} / {r["batch_size"]}',*[f'{r[n]:.2f}' for n in names]] for r in result['throughput']])+'\n\nAll throughput values are aggregate output tokens/s. FI means FlashInfer.')
ratios = result['ratios']
def ratio(length,batch,a,b):
    return next(r['ratio'] for r in ratios if r['input_length']==length and r['batch_size']==batch and r['numerator']==a and r['denominator']==b)
within = []
cross = []
for r in result['throughput']:
    length,batch = r['input_length'],r['batch_size']
    for os in ('windows','wsl2'):
        tri,fi,v = [f'{os}-{s}' for s in ('tokenspeed-triton','tokenspeed-flashinfer','vllm-auto')]
        within.append([os,f'{length} / {batch}',f'{ratio(length,batch,tri,v):.3f}x',f'{ratio(length,batch,fi,v):.3f}x',f'{ratio(length,batch,fi,tri):.3f}x'])
    cross.append([f'{length} / {batch}',*[f'{ratio(length,batch,"windows-"+s,"wsl2-"+s):.3f}x' for s in ('vllm-auto','tokenspeed-triton','tokenspeed-flashinfer')]])
text = text.replace('ENGINE_BACKEND_RATIOS',table(['Platform','Input / batch','TS Triton / vLLM','TS FI / vLLM','FI / Triton'],within))
text = text.replace('PLATFORM_RATIOS',table(['Input / batch','vLLM Win / WSL2','TS Triton Win / WSL2','TS FI Win / WSL2'],cross))
environment = {name:json.loads((dest/(name+'-graph.json')).read_text()) for name in names}
columns = ['windows-tokenspeed-triton','windows-vllm-auto','wsl2-tokenspeed-triton','wsl2-vllm-auto']
env_rows = [['Python',*[environment[n]['python'] for n in columns]],['Torch',*[environment[n]['packages']['torch'] for n in columns]],['Torch CUDA',*[environment[n]['torch_cuda'] for n in columns]],['FlashInfer',*[environment[n]['packages'].get('flashinfer-python','not installed') for n in columns]],['Triton',*[environment[n]['packages'].get('triton-windows',environment[n]['packages'].get('triton','not installed')) for n in columns]],['TokenSpeed Triton',*[environment[n]['packages'].get('tokenspeed-triton','not installed') for n in columns]],['vLLM',*[environment[n]['packages'].get('vllm','not installed') for n in columns]]]
text = text.replace('ENVIRONMENT_TABLE',table(['Package','Windows TS','Windows vLLM','WSL2 TS','WSL2 vLLM'],env_rows))
unstable = [f'{r["cell"]} input={r["input_length"]}, batch={r["batch_size"]}' for r in result['consistency'] if not r['repeat_consistent']]
matched = sum(r['reference_ids_match'] for r in ratios)
summary = f'{len(result["consistency"])} case/configuration combinations completed; {len(unstable)} had repeated greedy-ID variation. {matched} of {len(ratios)} listed pairwise comparisons had identical first-warmup reference IDs.'
if unstable:
    summary += '\n\nRepeated-output variation occurred in:\n\n'+'\n'.join('- '+s for s in unstable)
text = text.replace('CONSISTENCY_SUMMARY',summary)
text = text.replace('HISTORICAL_TABLE',table(['Stack','Input / batch','Old tokens/s','Fresh tokens/s','Fresh vs old'],[[r['cell'],f'{r["input_length"]} / {r["batch_size"]}',f'{r["old_tps"]:.2f}',f'{r["new_tps"]:.2f}',f'{100*(r["new_over_old"]-1):+.2f}%'] for r in result['historical']]))
guide.write_text(text,encoding='utf-8')
(dest/'README.md').write_text('# Fresh non-eager engine/backend evidence\n\nSee [the methods and complete tables](../../guides/windows-wsl2-engine-backend-retest.md).\n\nThe six top-level *-graph.json files are the selected fresh cohort. comparisons.json is derived only after matching model/script/GPU/workload/input identities and validating all output counts and measurement rounds. Raw reports retain every sampled token ID and elapsed time. Logs preserve startup configuration and failures. preliminary/ is excluded from all selected ratios: its native run overlapped dependency installation and includes a safetensors startup access violation. The final native cohort was rerun after installation and WSL testing. Source snapshots, command overrides, isolated vLLM dependency pins and official wheel provenance are included. No kernel-level AOT speedup is attributed to these dense-model results.\n',encoding='utf-8')
checksums = []
for p in sorted(dest.rglob('*')):
    if p.is_file() and p.name!='checksums.sha256':
        checksums.append(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(dest).as_posix())
(dest/'checksums.sha256').write_text('\n'.join(checksums)+'\n',encoding='utf-8')
print(json.dumps(result['throughput'],indent=2))
