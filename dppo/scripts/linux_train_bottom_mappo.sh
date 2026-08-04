#!/usr/bin/env bash
set -euo pipefail

# Linux-side bottom MAPPO training launcher.
# Windows runs AFSIM/Warlock; this Linux process only binds UDP, receives state,
# sends AssignTask messages, and trains the four bottom policies on GPU.
#
# Optional Windows lifecycle control:
#   Set WINDOWS_AUTO_WARLOCK=1 and WINDOWS_SSH_TARGET=user@windows_ip to make
#   this script start Warlock through Windows OpenSSH, then stop Warlock/Wizard
#   when training exits. Keep it disabled if you want to manage Warlock by hand.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

# Override these from shell if needed, e.g.:
#   UPDATES=200 ROLLOUT_STEPS=128 CUDA_VISIBLE_DEVICES=0 ./scripts/linux_train_bottom_mappo.sh
BIND_HOST="${BIND_HOST:-0.0.0.0}"
BIND_PORT="${BIND_PORT:-50050}"
UPDATES="${UPDATES:-200}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-128}"
HIDDEN_SIZE="${HIDDEN_SIZE:-128}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
ALGORITHM="${ALGORITHM:-happo}"
CURRICULUM_STAGE="${CURRICULUM_STAGE:-recon_only}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-256}"
LR="${LR:-3e-4}"
DEVICE="${DEVICE:-auto}"
# 50 bottom decisions/simulation-hour = 72 simulation seconds per decision.
BOTTOM_DECISIONS_PER_HOUR="${BOTTOM_DECISIONS_PER_HOUR:-50}"
SIMULATION_CLOCK_RATE="${SIMULATION_CLOCK_RATE:-40}"
DECISION_SECONDS="${DECISION_SECONDS:-0}"
ENABLE_NEGATIVE_REWARDS="${ENABLE_NEGATIVE_REWARDS:-0}"
PLATFORM_TIMEOUT="${PLATFORM_TIMEOUT:-120}"
STAGE_SNAPSHOT="${STAGE_SNAPSHOT:-}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_ROOT}/checkpoints/bottom_mappo_linux_gpu}"
RESUME="${RESUME:-}"
METRICS_FILE="${METRICS_FILE:-}"
AUTO_EPISODES="${AUTO_EPISODES:-0}"
TERMINAL_EXIT_CODE="${TERMINAL_EXIT_CODE:-75}"

WINDOWS_AUTO_WARLOCK="${WINDOWS_AUTO_WARLOCK:-0}"
WINDOWS_WARLOCK_STOP_ON_EXIT="${WINDOWS_WARLOCK_STOP_ON_EXIT:-1}"
WINDOWS_SSH_TARGET="${WINDOWS_SSH_TARGET:-}"
WINDOWS_SSH_USER="${WINDOWS_SSH_USER:-yang}"
WINDOWS_SSH_KEY="${WINDOWS_SSH_KEY:-}"
WINDOWS_SSH_PORT="${WINDOWS_SSH_PORT:-22}"
WINDOWS_SSH_CONNECT_TIMEOUT="${WINDOWS_SSH_CONNECT_TIMEOUT:-15}"
WINDOWS_WARLOCK_EXE="${WINDOWS_WARLOCK_EXE:-D:/junq/afsim_work/afsim-2.9.0-win64_bin/bin_release/warlock.exe}"
WINDOWS_SCENARIO_DIR="${WINDOWS_SCENARIO_DIR:-D:/junq/afsim_work/afsim-2.9.0-win64_bin/demos/air_to_air}"
WINDOWS_SCENARIO_FILE="${WINDOWS_SCENARIO_FILE:-scenarios/island_assault_min.txt}"
WINDOWS_WARLOCK_ARGS="${WINDOWS_WARLOCK_ARGS:--log-server-host localhost -log-server-port 18888}"
WINDOWS_WARLOCK_START_DELAY="${WINDOWS_WARLOCK_START_DELAY:-0}"
WINDOWS_WARLOCK_RESTART_DELAY="${WINDOWS_WARLOCK_RESTART_DELAY:-3}"
WINDOWS_WARLOCK_START_CMD="${WINDOWS_WARLOCK_START_CMD:-}"
WINDOWS_WARLOCK_STOP_CMD="${WINDOWS_WARLOCK_STOP_CMD:-powershell -NoProfile -ExecutionPolicy Bypass -Command 'Get-Process warlock,wizard -ErrorAction SilentlyContinue | Stop-Process -Force'}"
WINDOWS_WARLOCK_STARTED=0

_quote_ps() {
  printf "%s" "$1" | sed "s/'/''/g"
}

_default_windows_start_cmd() {
  local exe dir scenario args
  exe="$(_quote_ps "${WINDOWS_WARLOCK_EXE}")"
  dir="$(_quote_ps "${WINDOWS_SCENARIO_DIR}")"
  scenario="$(_quote_ps "${WINDOWS_SCENARIO_FILE}")"
  args="$(_quote_ps "${WINDOWS_WARLOCK_ARGS} ${WINDOWS_SCENARIO_FILE}")"
  printf "powershell -NoProfile -ExecutionPolicy Bypass -Command \"Start-Process -FilePath '%s' -ArgumentList '%s' -WorkingDirectory '%s'\"" "${exe}" "${args}" "${dir}"
}

_run_windows_cmd() {
  local label cmd
  label="$1"
  cmd="$2"
  if [[ -z "${WINDOWS_SSH_TARGET}" ]]; then
    echo "[linux_train_bottom_mappo] Windows ${label} skipped: WINDOWS_SSH_TARGET is empty" >&2
    return 2
  fi
  local ssh_args=(-p "${WINDOWS_SSH_PORT}" -o "ConnectTimeout=${WINDOWS_SSH_CONNECT_TIMEOUT}")
  if [[ -n "${WINDOWS_SSH_KEY}" ]]; then
    ssh_args+=(-i "${WINDOWS_SSH_KEY}")
  fi
  echo "[linux_train_bottom_mappo] Windows ${label}: ${WINDOWS_SSH_TARGET}:${WINDOWS_SSH_PORT}"
  ssh "${ssh_args[@]}" "${WINDOWS_SSH_TARGET}" "${cmd}"
}

_start_windows_warlock() {
  if [[ "${WINDOWS_AUTO_WARLOCK}" != "1" ]]; then
    return 0
  fi
  local cmd
  cmd="${WINDOWS_WARLOCK_START_CMD}"
  if [[ -z "${cmd}" ]]; then
    cmd="$(_default_windows_start_cmd)"
  fi
  _run_windows_cmd "start Warlock" "${cmd}"
  WINDOWS_WARLOCK_STARTED=1
  if [[ "${WINDOWS_WARLOCK_START_DELAY}" != "0" ]]; then
    sleep "${WINDOWS_WARLOCK_START_DELAY}"
  fi
}

_restart_windows_warlock() {
  echo "[linux_train_bottom_mappo] episode terminal: restarting Windows Warlock"
  _run_windows_cmd "stop Warlock/Wizard" "${WINDOWS_WARLOCK_STOP_CMD}"
  WINDOWS_WARLOCK_STARTED=0
  if [[ "${WINDOWS_WARLOCK_RESTART_DELAY}" != "0" ]]; then
    sleep "${WINDOWS_WARLOCK_RESTART_DELAY}"
  fi
  _start_windows_warlock
}
_cleanup_windows_warlock() {
  local status=$?
  if [[ "${WINDOWS_AUTO_WARLOCK}" == "1" && "${WINDOWS_WARLOCK_STOP_ON_EXIT}" == "1" && "${WINDOWS_WARLOCK_STARTED}" == "1" ]]; then
    _run_windows_cmd "stop Warlock/Wizard" "${WINDOWS_WARLOCK_STOP_CMD}" || true
  fi
  exit "${status}"
}

trap _cleanup_windows_warlock EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

ARGS=(
  "${PROJECT_ROOT}/train/train_bottom_mappo.py"
  --algorithm "${ALGORITHM}"
  --curriculum-stage "${CURRICULUM_STAGE}"
  --commander curriculum
  --bind
  --local-address "${BIND_HOST}:${BIND_PORT}"
  --platform-timeout "${PLATFORM_TIMEOUT}"
  --device "${DEVICE}"
  --decision-seconds "${DECISION_SECONDS}"
  --bottom-decisions-per-hour "${BOTTOM_DECISIONS_PER_HOUR}"
  --simulation-clock-rate "${SIMULATION_CLOCK_RATE}"
  --updates "${UPDATES}"
  --rollout-steps "${ROLLOUT_STEPS}"
  --hidden-size "${HIDDEN_SIZE}"
  --update-epochs "${UPDATE_EPOCHS}"
  --minibatch-size "${MINIBATCH_SIZE}"
  --lr "${LR}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
)
if [[ "${ENABLE_NEGATIVE_REWARDS}" == "1" ]]; then
  ARGS+=(--enable-negative-rewards)
fi


if [[ -n "${STAGE_SNAPSHOT}" ]]; then
  ARGS+=(--stage-snapshot "${STAGE_SNAPSHOT}")
fi

if [[ -n "${METRICS_FILE}" ]]; then
  ARGS+=(--metrics-file "${METRICS_FILE}")
fi

if [[ "${AUTO_EPISODES}" == "1" ]]; then
  ARGS+=(--target-update "${UPDATES}" --terminal-exit-code "${TERMINAL_EXIT_CODE}")
fi

mkdir -p "${CHECKPOINT_DIR}"

echo "[linux_train_bottom_mappo] project=${PROJECT_ROOT}"
echo "[linux_train_bottom_mappo] bind=${BIND_HOST}:${BIND_PORT} algorithm=${ALGORITHM} stage=${CURRICULUM_STAGE} device=${DEVICE} negative_rewards=${ENABLE_NEGATIVE_REWARDS} bottom_decisions_per_hour=${BOTTOM_DECISIONS_PER_HOUR} clock_rate=${SIMULATION_CLOCK_RATE} decision_seconds_override=${DECISION_SECONDS} updates=${UPDATES} rollout_steps=${ROLLOUT_STEPS}"
echo "[linux_train_bottom_mappo] checkpoint_dir=${CHECKPOINT_DIR}"
echo "[linux_train_bottom_mappo] make sure Windows AFSIM udpnet target is this Linux IP:${BIND_PORT}"
if [[ "${AUTO_EPISODES}" == "1" && -z "${WINDOWS_SSH_TARGET}" && -n "${SSH_CLIENT:-}" ]]; then
  WINDOWS_SSH_TARGET="${WINDOWS_SSH_USER}@${SSH_CLIENT%% *}"
  echo "[linux_train_bottom_mappo] inferred Windows SSH target: ${WINDOWS_SSH_TARGET}"
fi

if [[ "${WINDOWS_AUTO_WARLOCK}" == "1" ]]; then
  echo "[linux_train_bottom_mappo] windows_auto_warlock=1 target=${WINDOWS_SSH_TARGET} stop_on_exit=${WINDOWS_WARLOCK_STOP_ON_EXIT}"
fi

if [[ "${AUTO_EPISODES}" == "1" && "${WINDOWS_AUTO_WARLOCK}" != "1" ]]; then
  echo "[linux_train_bottom_mappo] AUTO_EPISODES=1 requires WINDOWS_AUTO_WARLOCK=1" >&2
  exit 2
fi

_start_windows_warlock

if [[ "${AUTO_EPISODES}" != "1" ]]; then
  RUN_ARGS=("${ARGS[@]}")
  if [[ -n "${RESUME}" ]]; then
    RUN_ARGS+=(--resume "${RESUME}")
  fi
  python "${RUN_ARGS[@]}"
  exit $?
fi

CURRENT_RESUME="${RESUME}"
while true; do
  RUN_ARGS=("${ARGS[@]}")
  if [[ -n "${CURRENT_RESUME}" ]]; then
    RUN_ARGS+=(--resume "${CURRENT_RESUME}")
  fi

  if python "${RUN_ARGS[@]}"; then
    echo "[linux_train_bottom_mappo] target update ${UPDATES} reached"
    break
  else
    status=$?
  fi

  if [[ "${status}" -ne "${TERMINAL_EXIT_CODE}" ]]; then
    echo "[linux_train_bottom_mappo] trainer failed with exit code ${status}" >&2
    exit "${status}"
  fi

  latest="${CHECKPOINT_DIR}/latest.pt"
  if [[ ! -f "${latest}" ]]; then
    echo "[linux_train_bottom_mappo] terminal checkpoint missing: ${latest}" >&2
    exit 3
  fi
  _restart_windows_warlock
  CURRENT_RESUME="${latest}"
done
