#!/usr/bin/env bash
set -euo pipefail
repo=/mnt/c/Users/user/projects/official-repos/windows-tokenspeed
publication=/mnt/c/Users/user/projects/official-repos/windows-tokenspeed-pr
source "$repo/.port/wsl-env.sh"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export PYTHONPATH="$publication"
cd "$publication"
python -m tools.perf.run_followup_baseline_checks --model /opt/tokenspeed-work/models/Qwen3-0.6B --report-dir "$repo/.port/engine-baseline-20261029" --platform-label wsl2
