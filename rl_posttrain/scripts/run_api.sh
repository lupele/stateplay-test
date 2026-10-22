#!/usr/bin/env bash
set -euo pipefail
RL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${RL_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RL_PYTHON="${STATEPLAY_WORLD_PYTHON:-${RL_ROOT}/outputs/rl_posttrain/_envs/world/bin/python}"
IFS=',' read -r -a RL_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
exec "${RL_PYTHON}" -m torch.distributed.run --standalone --nproc_per_node "${#RL_GPUS[@]}" -m rl_posttrain.train_api --config rl_posttrain/configs/api_b_only.yaml "$@"
