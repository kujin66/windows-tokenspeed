#!/usr/bin/env bash
set -euo pipefail
repo=/mnt/c/Users/user/projects/official-repos/windows-tokenspeed
source "$repo/.port/wsl-env.sh"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_FLASHINFER_SAMPLER=0
export TOKENSPEED_COMPARISON_OPS=flashinfer
export PYTHONPATH=/opt/tokenspeed-work/venv/flashinfer-comparison
unset TOKENSPEED_CUTE_DSL_AOT_CONFIG TOKENSPEED_CUTE_DSL_COMPILER_PYTHON
cd "$repo"
exec /opt/tokenspeed-work/venv/bin/python "$repo/tools/perf/benchmark_engine_matrix.py" --engine tokenspeed --backend flashinfer --platform wsl2-arm64 --execution graph --model /opt/tokenspeed-work/models/Qwen3-0.6B --report "$1" --input-lengths 128 512 --batch-sizes 1 4 --output-length 128 --warmup 3 --rounds 5
