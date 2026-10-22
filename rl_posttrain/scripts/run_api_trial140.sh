#!/usr/bin/env bash
set -euo pipefail
RL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${RL_ROOT}"
RL_RUN_ID="${1:-study_v2}"
if (( $# > 1 )); then echo "Usage: $0 [run_id]" >&2; exit 2; fi
bash rl_posttrain/scripts/run_api.sh --run-id "${RL_RUN_ID}" --max-conditions 140 --api-group-limit 140
exec outputs/rl_posttrain/_envs/world/bin/python -m rl_posttrain.trial_report --run-id "${RL_RUN_ID}"
