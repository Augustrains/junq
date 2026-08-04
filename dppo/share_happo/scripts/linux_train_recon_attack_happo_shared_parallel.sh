#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PARALLEL_WORKERS="${PARALLEL_WORKERS:-4}"
export SIMULATION_CLOCK_RATE="${SIMULATION_CLOCK_RATE:-20}"
export SHARE_POLICY_BY_TYPE=1
export TRAIN_SCRIPT="train/train_recon_attack_shared_parallel.py"
export EVAL_EPISODES="${EVAL_EPISODES:-3}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_ROOT}/checkpoints/happo_recon_attack_shared_parallel}"
export METRICS_FILE="${METRICS_FILE:-${CHECKPOINT_DIR}/metrics.jsonl}"
exec "${SCRIPT_DIR}/linux_train_recon_attack_happo_parallel.sh" "$@"