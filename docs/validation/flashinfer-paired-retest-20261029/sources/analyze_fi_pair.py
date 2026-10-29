import hashlib
import json
import math
import shutil
import statistics
from pathlib import Path

root = Path(__file__).resolve().parents[1]
run = root / '.port/flashinfer-paired-retest-20261029'
baseline = root / 'docs/validation/engine-matrix-retest-20261029'
outcomes = json.loads((run/'outcomes.json').read_text(encoding='utf-8'))
expected_order=['windows-flashinfer-repeat1','wsl2-flashinfer-repeat1','wsl2-flashinfer-repeat2','windows-flashinfer-repeat2']
assert [item['name'] for item in outcomes]==expected_order
assert all(item['passed'] and item['returncode']==0 and not item['environment_changes'] for item in outcomes)
old={os:json.loads((baseline/f'{os}-tokenspeed-flashinfer-graph.json').read_text()) for os in ('windows','wsl2')}
reports={}
for os in ('windows','wsl2'):
    for repeat in (1,2):
        name=f'{os}-flashinfer-repeat{repeat}'
        data=json.loads((run/(name+'.json')).read_text())
        assert data['passed'] and data['backend']=='flashinfer' and data['engine_name']=='tokenspeed' and data['execution']=='graph'
        assert data['warmup']==3 and data['rounds']==5 and data['output_length']==128
        for field in ('benchmark_sha256','packages','sources','engine_config','sampling','graph_policy','model_files','gpu','seed','warmup','rounds','input_lengths','batch_sizes','output_length','os','python','torch_cuda','source_root_override'):
            assert data[field]==old[os][field],(name,field)
        assert not any(v is None for v in data['packages'].values())
        assert data['packages']['flashinfer-python']==('0.6.17+g2ea3c24b5dac' if os=='windows' else '0.7.2')
        assert not data['engine_config']['enforce_eager'] and data['engine_config']['cudagraph_capture_sizes']==[1,4]
        assert not data['aot_router']['active']
        expected={(128,1),(128,4),(512,1),(512,4)}
        cases={(c['input_length'],c['batch_size']):c for c in data['cases']}
        assert len(data['cases'])==len(cases)==4 and set(cases)==expected
        reference={(c['input_length'],c['batch_size']):c for c in old[os]['cases']}
        for key,case in cases.items():
            assert case['input_ids']==reference[key]['input_ids']
            assert len(case['warmup_ids'])==3 and len(case['samples'])==5
            for ids in case['warmup_ids']+[s['output_ids'] for s in case['samples']]:
                assert len(ids)==key[1] and all(len(row)==128 for row in ids)
            assert all(math.isfinite(s['seconds']) and s['seconds']>0 for s in case['samples'])
            rate=key[1]*128*5/sum(s['seconds'] for s in case['samples'])
            assert math.isclose(rate,case['tokens_per_second'],rel_tol=1e-12)
            ids=case['warmup_ids']+[s['output_ids'] for s in case['samples']]
            assert case['repeat_consistent']==all(row==ids[0] for row in ids)
        reports[(os,repeat)]=data
    if os=='windows':
        guard_old=json.loads((baseline/'windows-tokenspeed-flashinfer-graph-environment.json').read_text())
        for repeat in (1,2):
            assert json.loads((run/f'windows-flashinfer-repeat{repeat}-environment.json').read_text())==guard_old
for field in ('benchmark_sha256','model_files','gpu','seed','input_lengths','batch_sizes','output_length','rounds','warmup'):
    assert old['windows'][field]==old['wsl2'][field],field
sha=hashlib.sha256((root/'tools/perf/benchmark_engine_matrix.py').read_bytes()).hexdigest()
assert sha==old['windows']['benchmark_sha256']

def get_case(data,key):
    return next(c for c in data['cases'] if (c['input_length'],c['batch_size'])==key)

rows=[]
for key in ((128,1),(128,4),(512,1),(512,4)):
    row=dict(input_length=key[0],batch_size=key[1],platforms={})
    for os in ('windows','wsl2'):
        original=get_case(old[os],key)
        cases=[get_case(reports[(os,r)],key) for r in (1,2)]
        durations=[s['seconds'] for c in cases for s in c['samples']]
        row['platforms'][os]=dict(
            old_tps=original['tokens_per_second'],
            old_seconds=[s['seconds'] for s in original['samples']],
            repeat_tps=[c['tokens_per_second'] for c in cases],
            repeat_seconds=[[s['seconds'] for s in c['samples']] for c in cases],
            pooled_tps=key[1]*128*len(durations)/sum(durations),
            median_seconds=statistics.median(durations),
            min_seconds=min(durations),max_seconds=max(durations),
            repeat_consistent=[c['repeat_consistent'] for c in cases],
        )
    win,wsl=[row['platforms'][p] for p in ('windows','wsl2')]
    row['old_ratio']=win['old_tps']/wsl['old_tps']
    row['repeat_ratios']=[a/b for a,b in zip(win['repeat_tps'],wsl['repeat_tps'],strict=True)]
    row['pooled_ratio']=win['pooled_tps']/wsl['pooled_tps']
    row['reference_ids_match']=[get_case(reports[('windows',r)],key)['warmup_ids'][0]==get_case(reports[('wsl2',r)],key)['warmup_ids'][0] for r in (1,2)]
    rows.append(row)
result=dict(schema=1,order=expected_order,benchmark_sha256=sha,environment_checks_passed=True,
    method='Two independent process repetitions per platform, reverse order Windows/WSL2 then WSL2/Windows; original 3 warmups/5 samples and all four cases; pooled TPS=total measured output tokens/total measured seconds; original observations retained, no outliers removed',
    scope='Same per-platform dependencies and module hashes as the original cohort; Windows FlashInfer 0.6.17 and WSL2 FlashInfer 0.7.2 remain different stacks; no AOT, engine or dependency changes applied',rows=rows)
(run/'comparisons.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
for row in rows:
    print(row['input_length'],row['batch_size'],'Windows',row['platforms']['windows']['repeat_tps'],'WSL2',row['platforms']['wsl2']['repeat_tps'],'ratios',row['repeat_ratios'],'pooled',row['pooled_ratio'])

dest=root/'docs/validation/flashinfer-paired-retest-20261029'
dest.mkdir(exist_ok=True)
for file in run.iterdir():
    if file.is_file():shutil.copyfile(file,dest/file.name)
sources=dest/'sources'
sources.mkdir(exist_ok=True)
for source in (Path(__file__),root/'.port/retest_fi_pair.py',root/'.port/retest_fi_wsl.sh',root/'.port/run_checked_benchmark.py',root/'.port/launch_flashinfer0617.py',root/'.port/wsl-env.sh',root/'tools/perf/benchmark_engine_matrix.py',root/'tools/windows/run.py'):
    shutil.copyfile(source,sources/source.name)
(dest/'README.md').write_text('# Paired FlashInfer confirmation\n\nSee [the report](../../guides/windows-wsl2-engine-backend-retest.md) for the original cohort and this confirmation. Four sequential process runs retain every measured time and output ID. Both repeats pass process, package, source, model, input, graph, output-count and throughput validation against the original per-platform environment. The two Windows FlashInfer import/version guard reports also match the original exactly. comparisons.json is recomputed from all ten measured calls per case/platform, with no samples removed. commands.json and outcomes.json retain execution order, pre/post GPU snapshots and lightweight CPU samples; no GPU telemetry is sampled during generation. These observations cannot isolate OS or compiler effects or establish output equivalence.\n',encoding='utf-8')
checks=[]
for file in sorted(dest.rglob('*')):
    if file.is_file() and file.name!='checksums.sha256':checks.append(hashlib.sha256(file.read_bytes()).hexdigest()+'  '+file.relative_to(dest).as_posix())
(dest/'checksums.sha256').write_text('\n'.join(checks)+'\n',encoding='utf-8')
