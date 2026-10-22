#!/usr/bin/env bash
set -euo pipefail
RL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${RL_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RL_PYTHON="${STATEPLAY_REWARD_PYTHON:-${RL_ROOT}/outputs/rl_posttrain/_envs/reward/bin/python}"
IFS=',' read -r -a RL_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
RL_TRAIN_ARGS=(--config rl_posttrain/configs/rm_qwen35.yaml)
RL_EVAL_ARGS=(--config rl_posttrain/configs/rm_qwen35.yaml --check-reload)
while (( $# )); do
  case "$1" in
    --config|--run-id)
      (( $# >= 2 )) || { printf 'Missing value for %s\n' "$1" >&2; exit 2; }
      RL_TRAIN_ARGS+=("$1" "$2"); RL_EVAL_ARGS+=("$1" "$2"); shift 2 ;;
    --smoke) RL_TRAIN_ARGS+=("$1"); RL_EVAL_ARGS+=("$1"); shift ;;
    --resume|--max-optimizer-steps|--interrupt-after-step|--smoke-accumulation)
      (( $# >= 2 )) || { printf 'Missing value for %s\n' "$1" >&2; exit 2; }
      RL_TRAIN_ARGS+=("$1" "$2"); shift 2 ;;
    --audit-replay) RL_TRAIN_ARGS+=("$1"); shift ;;
    --human-review|--checkpoint)
      (( $# >= 2 )) || { printf 'Missing value for %s\n' "$1" >&2; exit 2; }
      RL_EVAL_ARGS+=("$1" "$2"); shift 2 ;;
    --check-reload) RL_EVAL_ARGS+=("$1"); shift ;;
    *) printf 'Unsupported Reward Model argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
"${RL_PYTHON}" -m torch.distributed.run --standalone --nproc_per_node "${#RL_GPUS[@]}" -m rl_posttrain.reward_model.train "${RL_TRAIN_ARGS[@]}"
exec "${RL_PYTHON}" -m rl_posttrain.reward_model.evaluate "${RL_EVAL_ARGS[@]}"
