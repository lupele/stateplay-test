#!/usr/bin/env bash
set -euo pipefail
RL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${RL_ROOT}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RL_PYTHON="${STATEPLAY_WORLD_PYTHON:-${RL_ROOT}/outputs/rl_posttrain/_envs/world/bin/python}"
exec "${RL_PYTHON}" -m rl_posttrain.launch_local --config rl_posttrain/configs/local_b_only.yaml "$@"
