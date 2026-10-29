#!/usr/bin/env bash
set -euo pipefail
repo=$(dirname "$(dirname "$(readlink -f "$0")")")
source "$repo/.port/wsl-env.sh"
export HF_HUB_OFFLINE=1
python "$repo/.port/prepare_wsl_fa2.py"
export PYTHONPATH=/opt/tokenspeed-work/venv/flashinfer-comparison
python "$repo/.port/probe_fa2_prefill.py" --report "$repo/.port/flashinfer-comparison/wsl2-fa2-prefill-probe.json" > "$repo/.port/flashinfer-comparison/wsl2-fa2-prefill-probe.log" 2>&1
for execution in eager graph; do
  python "$repo/.port/flashinfer_comparison_bench.py" \
    --model /opt/tokenspeed-work/models/Qwen3-0.6B \
    --report "$repo/.port/flashinfer-comparison/wsl2-flashinfer-$execution.json" \
    --execution "$execution" --profile none --attention-backend flashinfer \
    --input-lengths 128 512 \
    > "$repo/.port/flashinfer-comparison/wsl2-flashinfer-$execution.log" 2>&1
  echo "Completed WSL2 FlashInfer $execution"
done
