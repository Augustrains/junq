#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-4}"
BASE_UDP_PORT="${BASE_UDP_PORT:-50050}"
WINDOWS_TASK_PREFIX="${WINDOWS_TASK_PREFIX:-AFSIM-Warlock-}"
WINDOWS_SSH_TARGET="${WINDOWS_SSH_TARGET:-yang@127.0.0.1}"
WINDOWS_SSH_PORT="${WINDOWS_SSH_PORT:-2222}"
WINDOWS_SSH_KEY="${WINDOWS_SSH_KEY:-${HOME}/.ssh/133_guzechen}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_ROOT}/checkpoints/happo_recon_attack_parallel}"
METRICS_FILE="${METRICS_FILE:-${CHECKPOINT_DIR}/metrics.jsonl}"
DEVICE="${DEVICE:-auto}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train/train_recon_attack_parallel.py}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 || ! "${PYTHON_BIN}" -c 'import torch' >/dev/null 2>&1; then
  SMACV2_PYTHON="${HOME}/miniconda3/envs/smacv2/bin/python"
  if [[ -x "${SMACV2_PYTHON}" ]] && "${SMACV2_PYTHON}" -c 'import torch' >/dev/null 2>&1; then
    PYTHON_BIN="${SMACV2_PYTHON}"
  else
    echo "[parallel_happo] no PyTorch Python found; activate a PyTorch Conda environment or set PYTHON_BIN" >&2
    exit 1
  fi
fi
TARGET_UPDATES="${TARGET_UPDATES:-100}"
TRAIN_EPISODES="${TRAIN_EPISODES:-100}"
RECON_MIN_SAMPLES="${RECON_MIN_SAMPLES:-2048}"
ATTACK_MIN_SAMPLES="${ATTACK_MIN_SAMPLES:-2048}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-5000}"
EVAL_MAX_EPISODE_STEPS="${EVAL_MAX_EPISODE_STEPS:-5000}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-64}"
BOTTOM_DECISIONS_PER_HOUR="${BOTTOM_DECISIONS_PER_HOUR:-50}"
SIMULATION_CLOCK_RATE="${SIMULATION_CLOCK_RATE:-20}"
DECISION_SECONDS="${DECISION_SECONDS:-0}"
ADAPTIVE_DECISION_TIMING="${ADAPTIVE_DECISION_TIMING:-0}"
NATIVE_DECISION_PAUSE="${NATIVE_DECISION_PAUSE:-0}"
NATIVE_DECISION_PAUSE_TIMEOUT="${NATIVE_DECISION_PAUSE_TIMEOUT:-45}"
PLATFORM_STATE_STALL_SECONDS="${PLATFORM_STATE_STALL_SECONDS:-30}"
BOTTOM_GLOBAL_REWARD_WEIGHT="${BOTTOM_GLOBAL_REWARD_WEIGHT:-0.1}"
BOTTOM_GLOBAL_REWARD_CLIP="${BOTTOM_GLOBAL_REWARD_CLIP:-10.0}"
RECON_ATTACK_DEADLINE_MISS_PENALTY="${RECON_ATTACK_DEADLINE_MISS_PENALTY:--50}"
RESUME="${RESUME:-}"
mkdir -p "${CHECKPOINT_DIR}"
ARGS=("${TRAIN_SCRIPT}" --workers "${PARALLEL_WORKERS}" --base-port "${BASE_UDP_PORT}" --device "${DEVICE}" --updates "${TARGET_UPDATES}" --episodes-per-worker "${TRAIN_EPISODES}" --max-episode-steps "${MAX_EPISODE_STEPS}" --eval-max-episode-steps "${EVAL_MAX_EPISODE_STEPS}" --recon-min-samples "${RECON_MIN_SAMPLES}" --attack-min-samples "${ATTACK_MIN_SAMPLES}" --update-epochs "${UPDATE_EPOCHS}" --minibatch-size "${MINIBATCH_SIZE}" --bottom-decisions-per-hour "${BOTTOM_DECISIONS_PER_HOUR}" --simulation-clock-rate "${SIMULATION_CLOCK_RATE}" --decision-seconds "${DECISION_SECONDS}" --native-decision-pause-timeout "${NATIVE_DECISION_PAUSE_TIMEOUT}" --platform-state-stall-seconds "${PLATFORM_STATE_STALL_SECONDS}" --bottom-global-reward-weight "${BOTTOM_GLOBAL_REWARD_WEIGHT}" --bottom-global-reward-clip "${BOTTOM_GLOBAL_REWARD_CLIP}" --deadline-miss-penalty "${RECON_ATTACK_DEADLINE_MISS_PENALTY}" --checkpoint-dir "${CHECKPOINT_DIR}" --metrics-file "${METRICS_FILE}" --warlock-ssh-target "${WINDOWS_SSH_TARGET}" --warlock-ssh-port "${WINDOWS_SSH_PORT}" --warlock-ssh-key "${WINDOWS_SSH_KEY}" --warlock-task-prefix "${WINDOWS_TASK_PREFIX}")
[[ -n "${RESUME}" ]] && ARGS+=(--resume "${RESUME}")
[[ "${ENABLE_NEGATIVE_REWARDS:-0}" == 1 ]] && ARGS+=(--enable-negative-rewards)
[[ "${ADAPTIVE_DECISION_TIMING}" == 1 ]] && ARGS+=(--adaptive-decision-timing)
[[ "${NATIVE_DECISION_PAUSE}" == 1 ]] && ARGS+=(--native-decision-pause)
[[ "${EVALUATION_ONLY:-0}" == 1 ]] && ARGS+=(--evaluation-only)
[[ "${SHARE_POLICY_BY_TYPE:-0}" == 1 ]] && ARGS+=(--share-policy-by-type --eval-episodes "${EVAL_EPISODES:-3}")
echo "[parallel_happo] workers=${PARALLEL_WORKERS} ports=${BASE_UDP_PORT}..$((BASE_UDP_PORT+PARALLEL_WORKERS-1)) shared_checkpoint=${CHECKPOINT_DIR} native_pause=${NATIVE_DECISION_PAUSE}"
echo "[parallel_happo] python=${PYTHON_BIN}"
exec "${PYTHON_BIN}" "${ARGS[@]}"