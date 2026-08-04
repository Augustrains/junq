#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

TARGET_UPDATES="${TARGET_UPDATES:-100}"
TRAIN_EPISODES="${TRAIN_EPISODES:-100}"
EVAL_EPISODES="${EVAL_EPISODES:-1}"
PLATFORM_STATE_STALL_SECONDS="${PLATFORM_STATE_STALL_SECONDS:-30}"
RECON_MIN_SAMPLES="${RECON_MIN_SAMPLES:-2048}"
ATTACK_MIN_SAMPLES="${ATTACK_MIN_SAMPLES:-2048}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-64}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-5000}"
RECON_ATTACK_DEADLINE_MISS_PENALTY="${RECON_ATTACK_DEADLINE_MISS_PENALTY:--50}"
BOTTOM_GLOBAL_REWARD_WEIGHT="${BOTTOM_GLOBAL_REWARD_WEIGHT:-0.1}"
BOTTOM_GLOBAL_REWARD_CLIP="${BOTTOM_GLOBAL_REWARD_CLIP:-10.0}"
# 50 bottom decisions/simulation-hour = 72 simulation seconds per decision.
BOTTOM_DECISIONS_PER_HOUR="${BOTTOM_DECISIONS_PER_HOUR:-50}"
SIMULATION_CLOCK_RATE="${SIMULATION_CLOCK_RATE:-60}"
DECISION_SECONDS="${DECISION_SECONDS:-0}"
ENABLE_NEGATIVE_REWARDS="${ENABLE_NEGATIVE_REWARDS:-0}"
# v2 adds the attack follower rejoin action (41 discrete attack actions).
# Keep its checkpoints separate from v1's 40-action actor heads.
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_ROOT}/checkpoints/happo_recon_attack_v2}"
SNAPSHOT_POOL="${SNAPSHOT_POOL:-${PROJECT_ROOT}/stage_snapshots}"
SNAPSHOT_POOL_SIZE="${SNAPSHOT_POOL_SIZE:-50}"
RESUME="${RESUME:-}"
AUTO_RESUME="${AUTO_RESUME:-1}"
RESET_RESUME_BUFFER="${RESET_RESUME_BUFFER:-0}"
DEVICE="${DEVICE:-auto}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
TERMINAL_EXIT_CODE="${TERMINAL_EXIT_CODE:-75}"
PLOT_TRAINING_CURVE="${PLOT_TRAINING_CURVE:-1}"
PLOT_EVERY_EPISODES="${PLOT_EVERY_EPISODES:-1}"
PLOT_MOVING_AVERAGE_WINDOW="${PLOT_MOVING_AVERAGE_WINDOW:-10}"

WINDOWS_SSH_TARGET="${WINDOWS_SSH_TARGET:-yang@127.0.0.1}"
WINDOWS_SSH_PORT="${WINDOWS_SSH_PORT:-2222}"
WINDOWS_SSH_KEY="${WINDOWS_SSH_KEY:-${HOME}/.ssh/133_guzechen}"
WINDOWS_TASK_NAME="${WINDOWS_TASK_NAME:-AFSIM-Warlock}"
WINDOWS_START_DELAY="${WINDOWS_START_DELAY:-0}"
WINDOWS_RESTART_DELAY="${WINDOWS_RESTART_DELAY:-3}"

PARALLEL_WORKERS="${PARALLEL_WORKERS:-4}"
if (( PARALLEL_WORKERS > 1 )); then
  source "${SCRIPT_DIR}/linux_train_recon_attack_happo_parallel_eval.sh"
fi

export CUDA_VISIBLE_DEVICES RECON_MIN_SAMPLES ATTACK_MIN_SAMPLES MAX_EPISODE_STEPS SNAPSHOT_POOL SNAPSHOT_POOL_SIZE

SSH_ARGS=(-p "${WINDOWS_SSH_PORT}" -i "${WINDOWS_SSH_KEY}" -o BatchMode=yes -o ConnectTimeout=15)

windows_command() {
  ssh "${SSH_ARGS[@]}" "${WINDOWS_SSH_TARGET}" "$1"
}

stop_warlock() {
  echo "[recon_attack_stage] Windows stop Warlock/Wizard"
  windows_command "powershell -NoProfile -Command \"Stop-ScheduledTask -TaskName '${WINDOWS_TASK_NAME}' -ErrorAction SilentlyContinue; Get-Process warlock,wizard -ErrorAction SilentlyContinue | Stop-Process -Force\"" || true
}

CLEANUP_STARTED=0
cleanup() {
  local status=$?
  trap - EXIT INT TERM HUP QUIT PIPE
  if (( CLEANUP_STARTED == 0 )); then
    CLEANUP_STARTED=1
    echo "[recon_attack_stage] cleanup status=${status}"
    stop_warlock
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP
trap 'exit 131' QUIT
trap 'exit 141' PIPE

mkdir -p "${CHECKPOINT_DIR}" "${SNAPSHOT_POOL}"
CURRENT_RESUME="${RESUME}"
if [[ "${AUTO_RESUME}" == "1" && -z "${CURRENT_RESUME}" && -f "${CHECKPOINT_DIR}/latest.pt" ]]; then
  CURRENT_RESUME="${CHECKPOINT_DIR}/latest.pt"
fi

echo "[recon_attack_stage] train_episodes=${TRAIN_EPISODES} eval_episodes_per_update=${EVAL_EPISODES} negative_rewards=${ENABLE_NEGATIVE_REWARDS} bottom_decisions_per_hour=${BOTTOM_DECISIONS_PER_HOUR} clock_rate=${SIMULATION_CLOCK_RATE} decision_seconds_override=${DECISION_SECONDS}"
echo "[recon_attack_stage] sample_minimums recon=${RECON_MIN_SAMPLES} attack=${ATTACK_MIN_SAMPLES} update_trigger=sample_threshold"
echo "[recon_attack_stage] deadline_miss_penalty=${RECON_ATTACK_DEADLINE_MISS_PENALTY}"
echo "[recon_attack_stage] bottom_reward=task_local+${BOTTOM_GLOBAL_REWARD_WEIGHT}*clip(team,+/-${BOTTOM_GLOBAL_REWARD_CLIP})"
echo "[recon_attack_stage] update_epochs=${UPDATE_EPOCHS} minibatch_size=${MINIBATCH_SIZE}"
echo "[recon_attack_stage] checkpoint_dir=${CHECKPOINT_DIR} snapshot_pool=${SNAPSHOT_POOL}/landing"

if (( TRAIN_EPISODES <= 0 )); then
  echo "[recon_attack_stage] TRAIN_EPISODES must be positive" >&2
  exit 2
fi

ARGS=(
  train/train_recon_attack_stage.py
  --algorithm happo
  --curriculum-stage recon_attack
  --bind
  --local-address 0.0.0.0:50050
  --platform-timeout 120
  --platform-state-stall-seconds "${PLATFORM_STATE_STALL_SECONDS}"
  --device "${DEVICE}"
  --bottom-decisions-per-hour "${BOTTOM_DECISIONS_PER_HOUR}"
  --simulation-clock-rate "${SIMULATION_CLOCK_RATE}"
  --decision-seconds "${DECISION_SECONDS}"
  --episodes "${TRAIN_EPISODES}"
  --eval-episodes "${EVAL_EPISODES}"
  --warlock-ssh-target "${WINDOWS_SSH_TARGET}"
  --warlock-ssh-port "${WINDOWS_SSH_PORT}"
  --warlock-ssh-key "${WINDOWS_SSH_KEY}"
  --warlock-task-name "${WINDOWS_TASK_NAME}"
  --start-remote-warlock
  --warlock-start-delay "${WINDOWS_START_DELAY}"
  --warlock-stop-delay "${WINDOWS_RESTART_DELAY}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --snapshot-pool "${SNAPSHOT_POOL}"
  --snapshot-pool-size "${SNAPSHOT_POOL_SIZE}"
  --recon-min-samples "${RECON_MIN_SAMPLES}"
  --attack-min-samples "${ATTACK_MIN_SAMPLES}"
  --max-episode-steps "${MAX_EPISODE_STEPS}"
  --update-epochs "${UPDATE_EPOCHS}"
  --minibatch-size "${MINIBATCH_SIZE}"
  --deadline-miss-penalty "${RECON_ATTACK_DEADLINE_MISS_PENALTY}"
  --bottom-global-reward-weight "${BOTTOM_GLOBAL_REWARD_WEIGHT}"
  --bottom-global-reward-clip "${BOTTOM_GLOBAL_REWARD_CLIP}"
)
if [[ "${ENABLE_NEGATIVE_REWARDS}" == "1" ]]; then
  ARGS+=(--enable-negative-rewards)
fi

if [[ "${PLOT_TRAINING_CURVE}" == "1" ]]; then
  ARGS+=(
    --plot-file "${CHECKPOINT_DIR}/training_curve.png"
    --plot-every-episodes "${PLOT_EVERY_EPISODES}"
    --plot-moving-average-window "${PLOT_MOVING_AVERAGE_WINDOW}"
  )
fi
if [[ -n "${CURRENT_RESUME}" ]]; then
  ARGS+=(--resume "${CURRENT_RESUME}")
fi
if [[ "${RESET_RESUME_BUFFER}" == "1" ]]; then
  ARGS+=(--reset-resume-buffer)
fi

set +e
python "${ARGS[@]}"
status=$?
set -e
if [[ "${PLOT_TRAINING_CURVE}" == "1" && -f "${CHECKPOINT_DIR}/metrics.jsonl" ]]; then
  python train/plot_recon_attack_training.py     --metrics "${CHECKPOINT_DIR}/metrics.jsonl"     --output "${CHECKPOINT_DIR}/training_curve.png"     --window "${PLOT_MOVING_AVERAGE_WINDOW}" ||     echo "[recon_attack_stage] warning: failed to update training curve" >&2
fi
if [[ "${status}" -ne 0 ]]; then
  echo "[recon_attack_stage] persistent trainer failed with exit code ${status}" >&2
  exit "${status}"
fi
echo "[recon_attack_stage] completed ${TRAIN_EPISODES} games in one persistent Python process"
