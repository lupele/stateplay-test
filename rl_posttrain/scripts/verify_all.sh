#!/usr/bin/env bash
set -euo pipefail
RL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${RL_ROOT}"
exec "${RL_ROOT}/outputs/rl_posttrain/_envs/world/bin/python" -m rl_posttrain.orchestrate --development-smoke --api-group-limit 1 "$@"
