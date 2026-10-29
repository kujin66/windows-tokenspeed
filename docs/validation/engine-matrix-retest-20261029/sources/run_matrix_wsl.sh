#!/usr/bin/env bash
set -euo pipefail
repo=/mnt/c/Users/user/projects/official-repos/windows-tokenspeed
source "$repo/.port/wsl-env.sh"
exec python "$repo/.port/run_matrix_wsl.py"
