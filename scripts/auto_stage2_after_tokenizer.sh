#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOKENIZER_OUTPUT="${TOKENIZER_OUTPUT:-${ROOT}/outputs/tokenizer}"
STAGE2_OUTPUT="${STAGE2_OUTPUT:-${ROOT}/outputs/latent_world}"
TOKENIZER_CKPT="${TOKENIZER_CKPT:-${TOKENIZER_OUTPUT}/transition_tokenizer.pt}"
STAGE1_LOG="${STAGE1_LOG:-${TOKENIZER_OUTPUT}/train.log}"
STAGE2_LOG="${STAGE2_LOG:-${STAGE2_OUTPUT}/train.log}"
WATCH_LOG="${WATCH_LOG:-${STAGE2_OUTPUT}/auto_stage2_after_tokenizer.log}"
POLL_SECONDS="${POLL_SECONDS:-60}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
BATCH_SIZE="${BATCH_SIZE:-4}"
STAGE2_SAVE_STEPS="${STAGE2_SAVE_STEPS:-500}"
STAGE2_MAX_TRAIN_STEPS="${STAGE2_MAX_TRAIN_STEPS:-10000}"

mkdir -p "${STAGE2_OUTPUT}"

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "${WATCH_LOG}"
}

stage1_running() {
  pgrep -af 'training/train.py' | grep -F -- '--train_mode transition_tokenizer' >/dev/null
}

stage2_running() {
  pgrep -af 'training/train.py' | grep -F -- '--train_mode latent_state_world_model' >/dev/null
}

log "watcher started"
log "tokenizer checkpoint: ${TOKENIZER_CKPT}"
log "stage 1 log: ${STAGE1_LOG}"
log "stage 2 output: ${STAGE2_OUTPUT}"
log "stage 2 save steps: ${STAGE2_SAVE_STEPS}"
log "stage 2 max train steps: ${STAGE2_MAX_TRAIN_STEPS}"

while stage1_running; do
  log "Stage 1 is still running; checking again in ${POLL_SECONDS}s"
  sleep "${POLL_SECONDS}"
done

log "Stage 1 process is no longer running"

if [[ ! -s "${TOKENIZER_CKPT}" ]]; then
  log "ERROR: tokenizer checkpoint not found or empty: ${TOKENIZER_CKPT}"
  log "Stage 2 was not started. Check Stage 1 log: ${STAGE1_LOG}"
  exit 1
fi

if stage2_running; then
  log "Stage 2 is already running; nothing to start"
  exit 0
fi

log "Starting Stage 2 latent_state_world_model"
log "Stage 2 log: ${STAGE2_LOG}"

cd "${ROOT}"
exec env \
  TRAIN_MODE=latent_state_world_model \
  STATEPLAY_OUTPUT="${STAGE2_OUTPUT}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  SAVE_STEPS="${STAGE2_SAVE_STEPS}" \
  MAX_TRAIN_STEPS="${STAGE2_MAX_TRAIN_STEPS}" \
  bash scripts/train.sh --transition_tokenizer_ckpt "${TOKENIZER_CKPT}" \
  >> "${STAGE2_LOG}" 2>&1
