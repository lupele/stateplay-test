#!/usr/bin/env bash
set -euo pipefail
RL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${RL_ROOT}"
if [[ -z "${STATEPLAY_GPT_API_KEY:-${OPENAI_API_KEY:-}}" ]]; then
  printf 'Configured GPT API credential is unavailable; no request was submitted.\n' >&2
  exit 2
fi
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export STATEPLAY_API_STREAM="${STATEPLAY_API_STREAM:-1}"
RL_FORMAL_CONFIG="${STATEPLAY_FORMAL_CONFIG:-${RL_ROOT}/rl_posttrain/configs/api_b_only_threaded.yaml}"
exec "${RL_ROOT}/outputs/rl_posttrain/_envs/world/bin/python" -u -m rl_posttrain.watchdog \
  --run-id study_v1 --config "${RL_FORMAL_CONFIG}" \
  --execute-plan --auto-extend --wait-for-human-review "$@"
