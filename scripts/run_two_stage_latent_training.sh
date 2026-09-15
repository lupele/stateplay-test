#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOKENIZER_OUTPUT="${TOKENIZER_OUTPUT:-${ROOT}/outputs/tokenizer}"
STAGE2_OUTPUT="${STAGE2_OUTPUT:-${ROOT}/outputs/latent_world}"
TOKENIZER_CKPT="${TOKENIZER_CKPT:-${TOKENIZER_OUTPUT}/transition_tokenizer.pt}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
STAGE1_LOG="${STAGE1_LOG:-${TOKENIZER_OUTPUT}/train_${RUN_ID}.log}"
STAGE2_LOG="${STAGE2_LOG:-${STAGE2_OUTPUT}/train_${RUN_ID}.log}"
ORCH_LOG="${ORCH_LOG:-${STAGE2_OUTPUT}/two_stage_${RUN_ID}.log}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
BATCH_SIZE="${BATCH_SIZE:-4}"
STAGE1_MAX_TRAIN_STEPS="${STAGE1_MAX_TRAIN_STEPS:-1500}"
STAGE1_SAVE_STEPS="${STAGE1_SAVE_STEPS:-500}"
STAGE2_MAX_TRAIN_STEPS="${STAGE2_MAX_TRAIN_STEPS:-10000}"
STAGE2_SAVE_STEPS="${STAGE2_SAVE_STEPS:-500}"

mkdir -p "${TOKENIZER_OUTPUT}" "${STAGE2_OUTPUT}"

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "${ORCH_LOG}"
}

if pgrep -af 'training/train.py' | grep -E -- '--train_mode (transition_tokenizer|latent_state_world_model)' >/dev/null; then
  log "ERROR: another StatePlay training process is already running"
  pgrep -af 'training/train.py' | tee -a "${ORCH_LOG}"
  exit 1
fi

log "two-stage latent training started"
log "run id: ${RUN_ID}"
log "gpus: ${CUDA_VISIBLE_DEVICES}; batch per rank: ${BATCH_SIZE}"
log "stage 1: max_steps=${STAGE1_MAX_TRAIN_STEPS}, save_steps=${STAGE1_SAVE_STEPS}, output=${TOKENIZER_OUTPUT}"
log "stage 1 log: ${STAGE1_LOG}"

cd "${ROOT}"
env \
  TRAIN_MODE=transition_tokenizer \
  STATEPLAY_OUTPUT="${TOKENIZER_OUTPUT}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  SAVE_STEPS="${STAGE1_SAVE_STEPS}" \
  MAX_TRAIN_STEPS="${STAGE1_MAX_TRAIN_STEPS}" \
  bash scripts/train.sh >> "${STAGE1_LOG}" 2>&1

log "stage 1 finished"

if [[ ! -s "${TOKENIZER_CKPT}" ]]; then
  log "ERROR: tokenizer checkpoint not found or empty: ${TOKENIZER_CKPT}"
  exit 1
fi

log "tokenizer checkpoint ready: ${TOKENIZER_CKPT}"
log "stage 2: max_steps=${STAGE2_MAX_TRAIN_STEPS}, save_steps=${STAGE2_SAVE_STEPS}, output=${STAGE2_OUTPUT}"
log "stage 2 log: ${STAGE2_LOG}"

env \
  TRAIN_MODE=latent_state_world_model \
  STATEPLAY_OUTPUT="${STAGE2_OUTPUT}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  SAVE_STEPS="${STAGE2_SAVE_STEPS}" \
  MAX_TRAIN_STEPS="${STAGE2_MAX_TRAIN_STEPS}" \
  bash scripts/train.sh --transition_tokenizer_ckpt "${TOKENIZER_CKPT}" >> "${STAGE2_LOG}" 2>&1

log "stage 2 finished"
log "two-stage latent training completed"
