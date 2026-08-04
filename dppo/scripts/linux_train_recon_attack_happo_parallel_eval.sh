#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

PARALLEL_WORKERS="${PARALLEL_WORKERS:-4}"
BASE_UDP_PORT="${BASE_UDP_PORT:-50050}"
SIMULATION_CLOCK_RATE="${SIMULATION_CLOCK_RATE:-20}"
BOTTOM_DECISIONS_PER_HOUR="${BOTTOM_DECISIONS_PER_HOUR:-50}"
TARGET_UPDATES="${TARGET_UPDATES:-100}"
TRAIN_EPISODES="${TRAIN_EPISODES:-100}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-5000}"
EVAL_MAX_EPISODE_STEPS="${EVAL_MAX_EPISODE_STEPS:-5000}"
RECON_MIN_SAMPLES="${RECON_MIN_SAMPLES:-2048}"
ATTACK_MIN_SAMPLES="${ATTACK_MIN_SAMPLES:-2048}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-64}"
DECISION_SECONDS="${DECISION_SECONDS:-1.3}"
NATIVE_DECISION_PAUSE="${NATIVE_DECISION_PAUSE:-1}"
NATIVE_DECISION_PAUSE_TIMEOUT="${NATIVE_DECISION_PAUSE_TIMEOUT:-45}"
PLATFORM_STATE_STALL_SECONDS="${PLATFORM_STATE_STALL_SECONDS:-30}"
BOTTOM_GLOBAL_REWARD_WEIGHT="${BOTTOM_GLOBAL_REWARD_WEIGHT:-0.1}"
BOTTOM_GLOBAL_REWARD_CLIP="${BOTTOM_GLOBAL_REWARD_CLIP:-10.0}"
RECON_ATTACK_DEADLINE_MISS_PENALTY="${RECON_ATTACK_DEADLINE_MISS_PENALTY:--50}"
WINDOWS_TASK_PREFIX="${WINDOWS_TASK_PREFIX:-AFSIM-Warlock-}"
WINDOWS_SSH_TARGET="${WINDOWS_SSH_TARGET:-yang@127.0.0.1}"
WINDOWS_SSH_PORT="${WINDOWS_SSH_PORT:-2222}"
WINDOWS_SSH_KEY="${WINDOWS_SSH_KEY:-${HOME}/.ssh/133_guzechen}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_ROOT}/checkpoints/happo_recon_attack_parallel_eval}"
METRICS_FILE="${METRICS_FILE:-${CHECKPOINT_DIR}/metrics.jsonl}"
PLOT_FILE="${PLOT_FILE:-${CHECKPOINT_DIR}/evaluation_curve.png}"
PLOT_WINDOW="${PLOT_WINDOW:-5}"
EVAL_WORKER="${EVAL_WORKER:-0}"
DEVICE="${DEVICE:-auto}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESUME="${RESUME:-}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 || ! "${PYTHON_BIN}" -c 'import torch, matplotlib' >/dev/null 2>&1; then
  SMACV2_PYTHON="${HOME}/miniconda3/envs/smacv2/bin/python"
  if [[ -x "${SMACV2_PYTHON}" ]] && "${SMACV2_PYTHON}" -c 'import torch, matplotlib' >/dev/null 2>&1; then
    PYTHON_BIN="${SMACV2_PYTHON}"
  else
    echo "[parallel_eval] no Python with PyTorch and matplotlib found" >&2
    exit 1
  fi
fi

mkdir -p "${CHECKPOINT_DIR}"
ARGS=(
  train/train_recon_attack_parallel_eval.py
  --workers "${PARALLEL_WORKERS}"
  --base-port "${BASE_UDP_PORT}"
  --simulation-clock-rate "${SIMULATION_CLOCK_RATE}"
  --bottom-decisions-per-hour "${BOTTOM_DECISIONS_PER_HOUR}"
  --updates "${TARGET_UPDATES}"
  --episodes-per-worker "${TRAIN_EPISODES}"
  --max-episode-steps "${MAX_EPISODE_STEPS}"
  --eval-max-episode-steps "${EVAL_MAX_EPISODE_STEPS}"
  --recon-min-samples "${RECON_MIN_SAMPLES}"
  --attack-min-samples "${ATTACK_MIN_SAMPLES}"
  --update-epochs "${UPDATE_EPOCHS}"
  --minibatch-size "${MINIBATCH_SIZE}"
  --decision-seconds "${DECISION_SECONDS}"
  --native-decision-pause-timeout "${NATIVE_DECISION_PAUSE_TIMEOUT}"
  --platform-state-stall-seconds "${PLATFORM_STATE_STALL_SECONDS}"
  --bottom-global-reward-weight "${BOTTOM_GLOBAL_REWARD_WEIGHT}"
  --bottom-global-reward-clip "${BOTTOM_GLOBAL_REWARD_CLIP}"
  --deadline-miss-penalty "${RECON_ATTACK_DEADLINE_MISS_PENALTY}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --metrics-file "${METRICS_FILE}"
  --plot-file "${PLOT_FILE}"
  --plot-window "${PLOT_WINDOW}"
  --eval-worker "${EVAL_WORKER}"
  --device "${DEVICE}"
  --warlock-ssh-target "${WINDOWS_SSH_TARGET}"
  --warlock-ssh-port "${WINDOWS_SSH_PORT}"
  --warlock-ssh-key "${WINDOWS_SSH_KEY}"
  --warlock-task-prefix "${WINDOWS_TASK_PREFIX}"
)
[[ "${NATIVE_DECISION_PAUSE}" == "1" ]] && ARGS+=(--native-decision-pause)
[[ -n "${RESUME}" ]] && ARGS+=(--resume "${RESUME}")
[[ "${ENABLE_NEGATIVE_REWARDS:-0}" == 1 ]] && ARGS+=(--enable-negative-rewards)

echo "[parallel_eval] workers=${PARALLEL_WORKERS} ports=${BASE_UDP_PORT}..$((BASE_UDP_PORT + PARALLEL_WORKERS - 1)) clock_rate=${SIMULATION_CLOCK_RATE} native_pause=${NATIVE_DECISION_PAUSE}"
echo "[parallel_eval] update_samples recon=${RECON_MIN_SAMPLES} attack=${ATTACK_MIN_SAMPLES} eval_after_each_update=1"
echo "[parallel_eval] checkpoint=${CHECKPOINT_DIR} metrics=${METRICS_FILE} plot=${PLOT_FILE}"
echo "[parallel_eval] python=${PYTHON_BIN}"
exec "${PYTHON_BIN}" "${ARGS[@]}"
