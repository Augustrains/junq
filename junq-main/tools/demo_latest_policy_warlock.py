"""Run one visible AFSIM/Warlock episode with the latest saved policy.

Inference only: this script never trains or saves model weights.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.afsim_env import AFSIMIslandEnv
from envs.rl_interface import AFSIMRLInterface
from train.checkpointing import load_training_checkpoint
from train.decision_timing import apply_bottom_decision_timing, resolve_bottom_decision_timing
from train.happo_trainer import HAPPOConfig, HAPPOTrainer
from train.recon_attack_stage import high_quality_landing_status
from train.train_recon_attack_parallel import start_worker, task_command


DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "linux_native_continue_windows_v2_high_value_radar_sam" / "latest.pt"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--port", type=int, default=50050)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--log-file", default="", help="Optional JSONL telemetry path.")
    parser.add_argument("--warlock-control", choices=("ssh", "local"), default="ssh")
    parser.add_argument("--warlock-ssh-target", default="yang@127.0.0.1")
    parser.add_argument("--warlock-ssh-port", type=int, default=2222)
    parser.add_argument("--warlock-ssh-key", default="~/.ssh/133_guzechen")
    parser.add_argument("--warlock-task-prefix", default="AFSIM-Warlock-")
    parser.add_argument("--warlock-start-delay", type=float, default=5.0)
    parser.add_argument("--warlock-stop-settle-seconds", type=float, default=5.0)
    parser.add_argument("--warlock-start-retries", type=int, default=3)
    parser.add_argument("--warlock-first-packet-timeout", type=float, default=30.0)
    parser.add_argument("--platform-timeout", type=float, default=300.0)
    parser.add_argument("--simulation-clock-rate", type=float, default=60.0)
    parser.add_argument("--bottom-decisions-per-hour", type=float, default=50.0)
    parser.add_argument("--native-decision-pause", action="store_true")
    parser.add_argument("--native-decision-pause-timeout", type=float, default=45.0)
    return parser.parse_args()


def compact_events(events, wanted_type):
    return [
        {key: event[key] for key in ("actor", "platform", "target", "role", "result") if key in event}
        for event in events if str(event.get("type", "")) == wanted_type
    ]


def build_trainer(api, device):
    specs = api.get_bottom_agent_specs()
    return HAPPOTrainer(
        {name: dict(specs[name]) for name in ("recon", "attack")},
        global_state_dim=int(specs["global"]["obs_dim"]),
        agent_types=("recon", "attack"), hidden_sizes=(128, 128),
        config=HAPPOConfig(gamma=0.99, gae_lambda=0.95, learning_rate=3e-4, update_epochs=4, minibatch_size=64),
        device=device, trainable_agent_types=("recon", "attack"),
    )


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError("checkpoint not found: {0}".format(checkpoint))
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    env = AFSIMIslandEnv(bind=True, auto_start_warlock=False, local_address=("0.0.0.0", args.port))
    apply_bottom_decision_timing(env, resolve_bottom_decision_timing(args.bottom_decisions_per_hour, args.simulation_clock_rate, 0.0))
    env.native_decision_pause_control = bool(args.native_decision_pause)
    env.native_decision_pause_timeout = float(args.native_decision_pause_timeout)
    api = AFSIMRLInterface(env, reward_profile="recon_attack_stage")
    trainer = build_trainer(api, device)
    checkpoint_state = load_training_checkpoint(str(checkpoint))
    trainer.load_state_dict(checkpoint_state.get("trainer", checkpoint_state))
    trainer.set_trainable_agent_types(())
    worker_args = SimpleNamespace(
        warlock_control=args.warlock_control, warlock_ssh_target=args.warlock_ssh_target,
        warlock_ssh_port=args.warlock_ssh_port, warlock_ssh_key=args.warlock_ssh_key,
        warlock_task_prefix=args.warlock_task_prefix, warlock_start_delay=args.warlock_start_delay,
        warlock_stop_settle_seconds=args.warlock_stop_settle_seconds,
        warlock_start_retries=args.warlock_start_retries,
        warlock_first_packet_timeout=args.warlock_first_packet_timeout,
        platform_timeout=args.platform_timeout, bottom_global_reward_weight=0.1,
        bottom_global_reward_clip=10.0, adaptive_decision_timing=False,
    )
    worker = {"id": 0, "port": args.port, "env": env, "api": api}
    log = Path(args.log_file).expanduser().open("w", encoding="utf-8") if args.log_file else None
    try:
        print("DEMO_CHECKPOINT", checkpoint, "update", checkpoint_state.get("update", "unknown"), flush=True)
        start_worker(worker_args, worker)
        collector = worker["collector"]
        total_reward = 0.0
        for step in range(1, args.max_steps + 1):
            rollout = collector.collect(trainer, n_steps=1, reset=False)
            summary = rollout.summary()
            step_info = rollout.step_infos[-1] if rollout.step_infos else {}
            events = list(step_info.get("events", []))
            attack_events = list(step_info.get("bottom_action_events", {}).get("attack", []))
            total_reward += float(summary.get("team_reward_sum", 0.0))
            status = high_quality_landing_status(env, 3, 3, 1)
            row = {
                "step": step, "sim_seconds": float(env._current_sim_time()),
                "team_reward": float(summary.get("team_reward_sum", 0.0)), "total_reward": total_reward,
                "attack_targets": compact_events(attack_events, "attack_target_selected"),
                "destroyed": compact_events(events, "target_destroyed"),
                "alive_blue_sams": int(status.get("alive_blue_sams", 0)),
                "destroyed_blue_air": int(status.get("destroyed_blue_air", 0)),
                "attack_alive": int(status.get("attack_alive", 0)), "recon_alive": int(status.get("recon_alive", 0)),
                "done": bool(summary.get("terminal", False)), "done_reason": str(summary.get("done_reason", "none")),
            }
            if log:
                log.write(json.dumps(row, ensure_ascii=False) + "\n"); log.flush()
            if step % max(1, args.print_every) == 0 or row["attack_targets"] or row["destroyed"] or row["done"]:
                print("DEMO_STEP", json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
            if row["done"] or status.get("landing_combat_conditions_met", False):
                break
        print("DEMO_FINISHED", json.dumps({"steps": step, "total_reward": total_reward, "status": status}, ensure_ascii=False, sort_keys=True), flush=True)
    finally:
        if log:
            log.close()
        try:
            task_command(worker_args, 0, "stop")
        finally:
            env.close()


if __name__ == "__main__":
    main()
