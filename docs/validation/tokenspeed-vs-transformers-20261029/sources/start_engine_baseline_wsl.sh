#!/usr/bin/env bash
set -euo pipefail
repo=/mnt/c/Users/user/projects/official-repos/windows-tokenspeed
publication=/mnt/c/Users/user/projects/official-repos/windows-tokenspeed-pr
source "$repo/.port/wsl-env.sh"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export PYTHONPATH="$publication"
out="$repo/.port/engine-baseline-20261029"
cd "$publication"
python -m tools.perf.probe_sdpa_backends --report "$out/wsl2-sdpa-backends.json" > "$out/wsl2-sdpa-backends.log" 2>&1
python -m tools.perf.run_without_tokenspeed_suite \
  --model /opt/tokenspeed-work/models/Qwen3-0.6B \
  --report-dir "$out" --platform-label wsl2 --diagnostics cpu \
  --configurations generate-torch loop-dynamic-torch loop-static-eager-torch loop-static-graph-torch loop-static-eager-flashinfer loop-static-graph-flashinfer \
  > "$out/wsl2-hf-suite.log" 2>&1
echo 'Completed WSL2 no-TokenSpeed controls'
python -m tools.perf.run_bucketed_without_tokenspeed \
  --model /opt/tokenspeed-work/models/Qwen3-0.6B \
  --report-dir "$out" --platform-label wsl2 \
  --normalizations torch flashinfer --input-lengths 128 512 --batch-sizes 1 4 \
  --output-length 128 --warmup 3 --rounds 5 --math-reduction full \
  > "$out/wsl2-hf-bucketed-suite.log" 2>&1
echo 'Completed WSL2 bucketed graph controls'
python -m tools.perf.run_bucketed_without_tokenspeed \
  --model /opt/tokenspeed-work/models/Qwen3-0.6B \
  --report-dir "$out" --platform-label wsl2 \
  --normalizations flashinfer --input-lengths 128 512 --batch-sizes 1 4 \
  --output-length 128 --warmup 3 --rounds 5 --math-reduction reduced \
  > "$out/wsl2-hf-bucketed-reduced-suite.log" 2>&1
echo 'Completed WSL2 reduced-precision math graph controls'
export PYTHONPATH=/opt/tokenspeed-work/venv/flashinfer-comparison
cd "$repo"
for execution in eager graph; do
  report="$out/wsl2-tokenspeed-flashinfer-$execution.json"
  if [ -e "$report" ]; then echo "Preserve existing report: $report"; exit 1; fi
  python "$repo/.port/flashinfer_comparison_bench.py" \
    --model /opt/tokenspeed-work/models/Qwen3-0.6B --report "$report" \
    --execution "$execution" --profile none --attention-backend flashinfer \
    --input-lengths 128 512 > "${report%.json}.log" 2>&1
  echo "Completed WSL2 TokenSpeed $execution"
done
