"""Parallel HAPPO training with a fresh evaluation after every update.

Four AFSIM actors share one learner and one global on-policy buffer. Completed
episodes are merged until both sample thresholds are met. After each learner
update, one fresh scenario is evaluated without adding data to the train
buffer. Evaluation reward and simulation end time are plotted against
cumulative training environment steps.
"""

from __future__ import annotations

import os
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.afsim_env import AFSIMIslandEnv
from envs.rl_interface import AFSIMRLInterface
from train.checkpointing import append_metrics_jsonl, load_training_checkpoint, save_training_checkpoint
from train.decision_timing import apply_bottom_decision_timing, resolve_bottom_decision_timing
from train.happo_trainer import HAPPOConfig, HAPPOTrainer
from train.recon_attack_stage import high_quality_landing_status, recon_attack_terminal_bonus
from train.stage_episode_buffer import TaskTrajectoryBuffer
from train.train_recon_attack_parallel import collect_parallel, start_worker, start_workers_parallel, task_command


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--base-port", type=int, default=50050)
    parser.add_argument("--config-path", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--episodes-per-worker", type=int, default=100)
    parser.add_argument("--max-episode-steps", type=int, default=5000)
    parser.add_argument("--recon-min-samples", type=int, default=2048)
    parser.add_argument("--attack-min-samples", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4, help="recon actor and critic learning rate")
    parser.add_argument(
        "--attack-lr", type=float, default=0.0,
        help="attack actor learning rate; 0 uses 0.1*--lr with --attack-init-checkpoint, otherwise --lr",
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--bottom-decisions-per-hour", type=float, default=50)
    parser.add_argument("--simulation-clock-rate", type=float, default=60)
    parser.add_argument("--decision-seconds", type=float, default=0)
    parser.add_argument("--adaptive-decision-timing", action="store_true")
    parser.add_argument("--native-decision-pause", action="store_true", help="wait for AFSIM DecisionReady at each native pause boundary")
    parser.add_argument("--native-decision-pause-timeout", type=float, default=45.0)
    parser.add_argument('--sim-time-window-control', action='store_true', help='use Python SimStop/SimRestart at simulation-time decision windows instead of native DecisionReady pauses')
    parser.add_argument("--platform-timeout", type=float, default=120)
    parser.add_argument("--platform-state-stall-seconds", type=float, default=30)
    parser.add_argument("--bottom-global-reward-weight", type=float, default=0.1)
    parser.add_argument("--bottom-global-reward-clip", type=float, default=10)
    parser.add_argument("--deadline-miss-penalty", type=float, default=-50)
    parser.add_argument("--min-recon-alive", type=int, default=3)
    parser.add_argument("--min-attack-alive", type=int, default=3)
    parser.add_argument("--min-loaded-transports", type=int, default=1)
    parser.add_argument(
        "--checkpoint-dir",
        default=str(ROOT / "checkpoints" / "happo_recon_attack_parallel_eval"),
    )
    parser.add_argument("--metrics-file", default="")
    parser.add_argument("--plot-file", default="")
    parser.add_argument("--plot-window", type=int, default=5)
    parser.add_argument("--resume", default="")
    parser.add_argument(
        "--attack-init-checkpoint", default="",
        help="start a new run by importing only attack Actor weights",
    )
    parser.add_argument(
        "--recon-init-checkpoint", default="",
        help="initialize a shared recon Actor from one recon policy in an old checkpoint",
    )
    parser.add_argument(
        "--recon-init-policy", default="red_recon_1",
        help="source recon policy name used by --recon-init-checkpoint",
    )
    parser.add_argument("--eval-worker", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--share-policy-by-type", action="store_true")
    parser.add_argument(
        "--shared-recon-actor", action="store_true",
        help="use one shared recon Actor; recon follower trajectories remain filtered",
    )
    parser.add_argument("--eval-max-episode-steps", type=int, default=5000)
    parser.add_argument(
        "--evaluation-timeout", type=float, default=900.0,
        help="maximum wall seconds for the complete post-update evaluation batch",
    )
    parser.add_argument(
        "--warlock-control",
        choices=("ssh", "local"),
        default="local" if sys.platform == "win32" else "ssh",
    )
    parser.add_argument("--warlock-ssh-target", default="")
    parser.add_argument("--warlock-ssh-port", type=int, default=22)
    parser.add_argument("--warlock-ssh-key", default="")
    parser.add_argument("--warlock-task-prefix", default="AFSIM-Warlock-")
    parser.add_argument("--warlock-start-delay", type=float, default=2)
    parser.add_argument("--enable-negative-rewards", action="store_true")
    return parser.parse_args()


def terminal_state(env, summary, args, step, maximum_steps):
    status = high_quality_landing_status(
        env, args.min_recon_alive, args.min_attack_alive, args.min_loaded_transports
    )
    if status["landing_combat_conditions_met"]:
        return status, "combat_landing_ready"
    if status["landing_time_override_met"]:
        return status, "landing_deadline_missed"
    if summary.get("terminal", False):
        return status, str(summary.get("done_reason", "environment_terminal"))
    if step >= maximum_steps:
        return status, "stage_time_limit"
    return status, ""


def terminal_bonus(env, status, reason, args):
    return recon_attack_terminal_bonus(
        status,
        reason,
        env._current_sim_time(),
        deadline_miss_penalty=args.deadline_miss_penalty,
        negative_rewards_enabled=env.negative_rewards_enabled,
    )


def stalled_stream_reason(status):
    """Classify a missing final UDP packet without hiding a real stream fault."""
    sim_time = float(status.get("landing_time_seconds", 0.0))
    deadline = float(status.get("landing_force_open_time_seconds", 10800.0))
    if bool(status.get("landing_time_override_met", False)) or sim_time >= deadline - 600.0:
        return "landing_deadline_missed"
    return "udp_state_stream_stalled"


def load_evaluations(metrics_file):
    path = Path(metrics_file)
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                row = json.loads(line)
                if row.get("record_type") == "policy_evaluation":
                    rows.append(row)
    return rows


def rolling_mean(values, window):
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    for index in range(values.size):
        first = max(0, index - max(1, int(window)) + 1)
        result[index] = np.mean(values[first:index + 1])
    return result


def update_evaluation_plot(metrics_file, plot_file, window=5):
    """Plot evaluation reward and simulation end time vs training steps."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_evaluations(metrics_file)
    if not rows:
        return None
    training_steps = np.asarray([int(row.get("training_steps", 0)) for row in rows])
    rewards = np.asarray([float(row.get("evaluation_total_reward", 0.0)) for row in rows])
    end_hours = np.asarray(
        [float(row.get("evaluation_sim_seconds", 0.0)) / 3600.0 for row in rows]
    )
    successes = np.asarray([bool(row.get("success", False)) for row in rows])

    fig, (reward_ax, time_ax) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    reward_ax.plot(
        training_steps, rewards, color="#2563eb", marker="o", alpha=0.55,
        linewidth=1.4, label="Evaluation reward",
    )
    reward_ax.plot(
        training_steps, rolling_mean(rewards, window), color="#0f3d91",
        linewidth=2.2, label="{0}-evaluation moving average".format(max(1, int(window))),
    )
    if np.any(successes):
        reward_ax.scatter(
            training_steps[successes], rewards[successes], color="#16a34a",
            s=38, zorder=4, label="Combat landing success",
        )
    reward_ax.set_ylabel("Evaluation total reward")
    reward_ax.set_title("Post-update policy evaluation")
    reward_ax.grid(True, alpha=0.22)
    reward_ax.legend(loc="best")

    time_ax.plot(
        training_steps, end_hours, color="#be123c", marker="s",
        linewidth=1.5, label="Evaluation end time",
    )
    time_ax.axhline(
        3.0, color="#dc2626", linestyle="--", linewidth=1.0, label="3-hour deadline"
    )
    time_ax.set_xlabel("Cumulative training environment steps")
    time_ax.set_ylabel("Evaluation simulation hours")
    time_ax.grid(True, alpha=0.22)
    time_ax.legend(loc="best")

    fig.tight_layout()
    output = Path(plot_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    fig.savefig(temporary, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    temporary.replace(output)
    return output


def alive_unit_summary(env):
    """Return registered alive/total unit counts grouped by side and role."""
    summary = {}
    for side in ("red", "blue"):
        units = [platform for platform in env.platforms.values() if platform.side == side]
        roles = sorted({platform.role for platform in units})
        by_role = {
            role: {
                "alive": sum(1 for platform in units if platform.role == role and platform.alive),
                "total": sum(1 for platform in units if platform.role == role),
            }
            for role in roles
        }
        summary[side] = {
            "alive": sum(1 for platform in units if platform.alive),
            "total": len(units),
            "by_role": by_role,
        }
    return summary


def run_evaluation(args, worker, trainer, update_id, training_steps, training_episodes, evaluation_episode=1):
    """Run one fresh scenario without appending anything to a train buffer."""
    env = worker["env"]

    # AFSIM simulation time returns to zero when a fresh Mission starts.
    # Clear the previous episode's DecisionReady timestamp before launching it,
    # otherwise a new 72-second boundary can be rejected as older than 10800.
    env.native_decision_ready = False
    env.native_decision_ready_seen = False
    env.native_decision_ready_time = float("-inf")

    start_worker(args, worker)
    collector = worker["collector"]
    team_reward = 0.0
    started_at = time.monotonic()
    reason = "stage_time_limit"
    evaluation_steps = 0
    status = high_quality_landing_status(
        env, args.min_recon_alive, args.min_attack_alive, args.min_loaded_transports
    )
    for step in range(1, int(args.eval_max_episode_steps) + 1):
        rollout = collector.collect(trainer, n_steps=1, reset=False)
        summary = rollout.summary()
        evaluation_steps += int(summary.get("steps", 0))
        team_reward += float(summary.get("team_reward_sum", 0.0))
        if env.platform_state_age_seconds() > args.platform_state_stall_seconds:
            status = high_quality_landing_status(
                env, args.min_recon_alive, args.min_attack_alive,
                args.min_loaded_transports,
            )
            current_reason = stalled_stream_reason(status)
        else:
            status, current_reason = terminal_state(
                env, summary, args, step, args.eval_max_episode_steps
            )
        if current_reason:
            reason = current_reason
            break
    bonus = terminal_bonus(env, status, reason, args)
    result = {
        "record_type": "policy_evaluation",
        "update": int(update_id),
        "training_steps": int(training_steps),
        "training_episodes": int(training_episodes),
        "evaluation_worker": int(worker["id"]),
        "evaluation_episode": int(evaluation_episode),
        "evaluation_steps": int(evaluation_steps),
        "done_reason": reason,
        "success": reason == "combat_landing_ready",
        "ended_at_three_hours": reason == "landing_deadline_missed",
        "landing_executable": bool(status.get("ready", False)),
        "landing_combat_conditions_met": bool(
            status.get("landing_combat_conditions_met", False)
        ),
        "landing_trigger_reason": str(status.get("landing_trigger_reason", "closed")),
        "landing_status": status,
        "surviving_units": alive_unit_summary(env),
        "evaluation_team_reward": float(team_reward),
        "evaluation_terminal_reward": float(bonus),
        "evaluation_total_reward": float(team_reward + bonus),
        "evaluation_sim_seconds": float(env._current_sim_time()),
        "evaluation_wall_seconds": max(0.0, time.monotonic() - started_at),
        "evaluation_finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    print("POLICY_EVALUATION", json.dumps(result, sort_keys=True), flush=True)
    return result


def import_attack_actor_state(trainer, trainer_state):
    """Import only attack Actor parameters from an official HAPPO state."""
    if trainer_state.get("algorithm") != "happo_official_on_policy":
        raise ValueError("attack initialization requires an official HAPPO checkpoint")
    saved = trainer_state.get("policies", {})
    expected = sorted(
        name for name, agent_type in trainer.entity_agent_type.items()
        if agent_type == "attack"
    )
    missing = [name for name in expected if name not in saved]
    if missing:
        raise ValueError("checkpoint missing attack Actor policies: {0}".format(missing))
    imported = []
    for name in expected:
        packet = saved[name]
        if str(packet.get("agent_type", "")) != "attack":
            raise ValueError(
                "checkpoint policy {0} is not labeled as attack".format(name)
            )
        trainer.entity_policies[name].actor.load_state_dict(
            packet["actor"], strict=True
        )
        imported.append(name)
    return imported


def import_shared_recon_actor_state(trainer, trainer_state, source_policy="red_recon_1"):
    """Initialize the single shared recon Actor from one old recon Actor."""
    if trainer_state.get("algorithm") != "happo_official_on_policy":
        raise ValueError("recon initialization requires an official HAPPO checkpoint")
    recon_targets = [
        name for name, agent_type in trainer.entity_agent_type.items()
        if agent_type == "recon"
    ]
    if len(recon_targets) != 1:
        raise ValueError("recon initialization requires exactly one shared recon Actor")
    saved = trainer_state.get("policies", {})
    if source_policy not in saved:
        raise ValueError("checkpoint missing recon Actor policy: {0}".format(source_policy))
    packet = saved[source_policy]
    if str(packet.get("agent_type", "")) != "recon":
        raise ValueError("checkpoint policy {0} is not labeled as recon".format(source_policy))
    target = recon_targets[0]
    trainer.entity_policies[target].actor.load_state_dict(packet["actor"], strict=True)
    return target


def set_attack_actor_learning_rate(trainer, learning_rate):
    learning_rate = float(learning_rate)
    if learning_rate <= 0.0:
        raise ValueError("attack actor learning rate must be positive")
    updated = []
    for name, agent_type in trainer.entity_agent_type.items():
        if agent_type != "attack":
            continue
        for group in trainer.entity_policies[name].actor_optimizer.param_groups:
            group["lr"] = learning_rate
        updated.append(name)
    return updated

def main():
    args = parse_args()
    if args.eval_episodes < 1:
        raise ValueError("--eval-episodes must be positive")
    if args.eval_episodes > args.workers:
        raise ValueError("--eval-episodes cannot exceed --workers for concurrent evaluation")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if not 0 <= args.eval_worker < args.workers:
        raise ValueError("--eval-worker must select an existing worker")
    if args.resume and (args.attack_init_checkpoint or args.recon_init_checkpoint):
        raise ValueError("--resume cannot be combined with Actor initialization checkpoints")
    if args.recon_init_checkpoint and not args.shared_recon_actor:
        raise ValueError("--recon-init-checkpoint requires --shared-recon-actor")
    if args.attack_lr < 0.0:
        raise ValueError("--attack-lr cannot be negative")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = args.metrics_file or str(checkpoint_dir / "metrics.jsonl")
    plot_file = args.plot_file or str(checkpoint_dir / "evaluation_curve.png")
    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    timing = resolve_bottom_decision_timing(
        args.bottom_decisions_per_hour, args.simulation_clock_rate, args.decision_seconds
    )

    workers = []
    for worker_id in range(args.workers):
        env = AFSIMIslandEnv(
            config_path=args.config_path or None,
            bind=True,
            auto_start_warlock=False,
            local_address=("0.0.0.0", args.base_port + worker_id),
        )
        env.set_negative_rewards_enabled(args.enable_negative_rewards)
        apply_bottom_decision_timing(env, timing)
        env.native_decision_pause_control = bool(args.native_decision_pause)
        env.native_decision_pause_timeout = float(args.native_decision_pause_timeout)
        if args.sim_time_window_control:
            env.sim_time_window_control = True
            env.lockstep_pause_control = True
        api = AFSIMRLInterface(env, reward_profile="recon_attack_stage")
        workers.append({
            "id": worker_id,
            "port": args.base_port + worker_id,
            "env": env,
            "api": api,
            "buffer": TaskTrajectoryBuffer(("recon", "attack")),
            "episode": 0,
        })

    specs = workers[0]["api"].get_bottom_agent_specs()
    bottom_specs = {name: dict(specs[name]) for name in ("recon", "attack")}
    if args.shared_recon_actor:
        # Recon followers do not issue independent movement commands. One shared
        # Actor lets any surviving follower inherit leadership with a trained policy.
        bottom_specs["recon"]["entity_names"] = ("recon_shared",)
        bottom_specs["recon"]["num_entities"] = 1
    trainer = HAPPOTrainer(
        bottom_specs,
        global_state_dim=int(specs["global"]["obs_dim"]),
        agent_types=("recon", "attack"),
        hidden_sizes=(args.hidden_size, args.hidden_size),
        config=HAPPOConfig(
            gamma=args.gamma, gae_lambda=args.gae_lambda,
            learning_rate=args.lr, update_epochs=args.update_epochs,
            minibatch_size=args.minibatch_size,
        ),
        device=device,
        trainable_agent_types=("recon", "attack"),
    )
    attack_lr = float(args.attack_lr) if args.attack_lr > 0.0 else (
        float(args.lr) * 0.1 if args.attack_init_checkpoint else float(args.lr)
    )
    attack_lr_entities = set_attack_actor_learning_rate(trainer, attack_lr)
    global_buffer = TaskTrajectoryBuffer(("recon", "attack"))
    update_id = 0
    training_steps = 0
    training_episodes = 0
    imported_attack_actors = []
    initialization = {
        "mode": "fresh",
        "attack_actor_checkpoint": "",
        "imported_attack_actors": [],
        "recon_actor": "fresh",
        "critics": "fresh",
        "optimizers": "fresh",
        "recon_lr": float(args.lr),
        "attack_lr": float(attack_lr),
    }
    if args.recon_init_checkpoint:
        state = load_training_checkpoint(args.recon_init_checkpoint)
        imported_recon_actor = import_shared_recon_actor_state(
            trainer,
            state.get("trainer", state),
            source_policy=args.recon_init_policy,
        )
        initialization.update({
            "mode": "actor_warm_start",
            "recon_actor": "warm_start",
            "recon_actor_checkpoint": str(args.recon_init_checkpoint),
            "recon_source_policy": str(args.recon_init_policy),
            "imported_recon_actor": imported_recon_actor,
        })
        print(
            "shared_recon_actor_warm_start",
            "checkpoint", args.recon_init_checkpoint,
            "source", args.recon_init_policy,
            "target", imported_recon_actor,
            "critics", "fresh",
            "optimizers", "fresh",
            flush=True,
        )
    if args.attack_init_checkpoint:
        state = load_training_checkpoint(args.attack_init_checkpoint)
        imported_attack_actors = import_attack_actor_state(
            trainer, state.get("trainer", state)
        )
        initialization.update({
            "mode": "attack_actor_warm_start",
            "attack_actor_checkpoint": str(args.attack_init_checkpoint),
            "imported_attack_actors": list(imported_attack_actors),
        })
        print(
            "attack_actor_warm_start",
            "checkpoint", args.attack_init_checkpoint,
            "policies", len(imported_attack_actors),
            "attack_lr", attack_lr,
            "recon_lr", args.lr,
            "critics", "fresh",
            "optimizers", "fresh",
            flush=True,
        )
    if args.resume:
        state = load_training_checkpoint(args.resume)
        trainer.load_state_dict(state.get("trainer", state))
        initialization = dict(state.get("initialization", initialization))
        global_buffer.load_state_dict(state.get("stage_buffer", {}))
        update_id = int(state.get("update", 0))
        training_steps = int(state.get("training_steps", 0))
        training_episodes = int(state.get("training_episodes", 0))

    minimum = {"recon": args.recon_min_samples, "attack": args.attack_min_samples}
    executor = ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="afsim_actor")
    print(
        "parallel_eval_training", "workers", args.workers,
        "ports", "{0}..{1}".format(args.base_port, args.base_port + args.workers - 1),
        "clock_rate", args.simulation_clock_rate, "native_pause", args.native_decision_pause, "minimum_samples", minimum,
        "evaluation_mode", "concurrent", "eval_episodes", args.eval_episodes,
        "plot", plot_file, flush=True,
    )
    print("shared_policy_object", id(trainer), "workers", args.workers, flush=True)

    try:
        start_workers_parallel(executor, args, workers)

        while update_id < args.updates and any(
            worker["episode"] < args.episodes_per_worker for worker in workers
        ):
            active = [
                worker for worker in workers
                if worker["episode"] < args.episodes_per_worker
            ]
            done = set()
            round_team_rewards = {worker["id"]: 0.0 for worker in active}
            round_steps = {worker["id"]: 0 for worker in active}

            for step in range(1, args.max_episode_steps + 1):
                sampling = collect_parallel(executor, active, done, trainer)
                for worker in active:
                    worker_id = worker["id"]
                    if worker_id in done:
                        continue
                    rollout = sampling[worker_id]
                    summary = rollout.summary()
                    worker["buffer"].append_rollout(rollout)
                    step_count = int(summary.get("steps", 0))
                    training_steps += step_count
                    round_steps[worker_id] += step_count
                    round_team_rewards[worker_id] += float(summary.get("team_reward_sum", 0.0))
                    if worker["env"].platform_state_age_seconds() > args.platform_state_stall_seconds:
                        status = high_quality_landing_status(
                            worker["env"], args.min_recon_alive,
                            args.min_attack_alive, args.min_loaded_transports,
                        )
                        reason = stalled_stream_reason(status)
                    else:
                        status, reason = terminal_state(
                            worker["env"], summary, args, step, args.max_episode_steps
                        )
                    if not reason:
                        continue
                    bonus = terminal_bonus(worker["env"], status, reason, args)
                    worker["buffer"].apply_episode_global_reward(bonus)
                    worker["buffer"].finish_episode(end_reason=reason)
                    moved = global_buffer.merge_completed_from(worker["buffer"], worker_id)
                    worker["episode"] += 1
                    training_episodes += 1
                    done.add(worker_id)
                    append_metrics_jsonl(metrics_file, {
                        "record_type": "training_episode",
                        "worker": worker_id,
                        "worker_episode": int(worker["episode"]),
                        "training_episodes": int(training_episodes),
                        "training_steps": int(training_steps),
                        "policy_update": int(update_id),
                        "steps": int(round_steps[worker_id]),
                        "done_reason": reason,
                        "team_reward": float(round_team_rewards[worker_id]),
                        "terminal_reward": float(bonus),
                        "total_reward": float(round_team_rewards[worker_id] + bonus),
                        "sim_seconds": float(worker["env"]._current_sim_time()),
                        "moved_samples": moved,
                    })
                    print(
                        "worker_trajectory_merged", worker_id,
                        "episode", worker["episode"], "reason", reason,
                        "steps", round_steps[worker_id], "moved", moved,
                        "global", global_buffer.assigned_counts(), flush=True,
                    )
                if len(done) == len(active):
                    break

            if global_buffer.ready(minimum):
                counts = global_buffer.assigned_counts()
                update_metrics = trainer.update_all(global_buffer.to_batches())
                update_id += 1
                global_buffer.clear_after_update()
                for worker in workers:
                    worker["buffer"].policy_version = update_id
                append_metrics_jsonl(metrics_file, {
                    "record_type": "policy_update",
                    "update": update_id,
                    "training_steps": training_steps,
                    "training_episodes": training_episodes,
                    "samples": counts,
                    "workers": args.workers,
                    "metrics": update_metrics,
                })
                print(
                    "shared_happo_update", update_id,
                    "training_steps", training_steps,
                    "samples", counts, flush=True,
                )

                # Save immediately after the optimizer update. Evaluation is
                # external I/O and must not be allowed to discard a valid update.
                pre_evaluation_payload = {
                    "update": update_id,
                    "training_steps": training_steps,
                    "training_episodes": training_episodes,
                    "trainer": trainer.state_dict(),
                    "stage_buffer": global_buffer.state_dict(include_active=False),
                    "samples": counts,
                    "workers": args.workers,
                    "last_evaluation": None,
                    "last_evaluation_summary": None,
                    "share_policy_by_type": args.share_policy_by_type,
                    "initialization": dict(initialization),
                    "metrics_file": metrics_file,
                    "plot_file": plot_file,
                }
                _, pre_evaluation_latest = save_training_checkpoint(
                    pre_evaluation_payload,
                    checkpoint_dir,
                    "bottom_happo_recon_attack_parallel_eval",
                    update_id,
                )
                print(
                    "parallel_eval_checkpoint_pre_evaluation",
                    pre_evaluation_latest,
                    flush=True,
                )

                evaluation_workers = [
                    workers[(args.eval_worker + offset) % args.workers]
                    for offset in range(int(args.eval_episodes))
                ]
                evaluation_worker_ids = {
                    int(worker["id"]) for worker in evaluation_workers
                }
                # Stop paused non-evaluation Mission workers. Their repeated
                # DecisionReady heartbeats can starve the fresh test worker.
                for worker in workers:
                    if int(worker["id"]) not in evaluation_worker_ids:
                        task_command(args, worker["id"], "stop")
                        print("policy_evaluation_stopped_idle_worker", worker["id"], flush=True)
                evaluation_futures = {}
                for evaluation_episode, worker in enumerate(
                    evaluation_workers, start=1
                ):
                    start_record = {
                        "record_type": "policy_evaluation_start",
                        "update": int(update_id),
                        "training_steps": int(training_steps),
                        "training_episodes": int(training_episodes),
                        "evaluation_worker": int(worker["id"]),
                        "evaluation_episode": int(evaluation_episode),
                        "started_at_utc": datetime.now(timezone.utc).isoformat(),
                    }
                    append_metrics_jsonl(metrics_file, start_record)
                    print(
                        "POLICY_EVALUATION_START",
                        json.dumps(start_record, sort_keys=True), flush=True,
                    )
                    future = executor.submit(
                        run_evaluation, args, worker, trainer,
                        update_id, training_steps, training_episodes,
                        evaluation_episode,
                    )
                    evaluation_futures[future] = start_record
                evaluations = []
                try:
                    for future in as_completed(
                        evaluation_futures,
                        timeout=max(1.0, float(args.evaluation_timeout)),
                    ):
                        evaluation = future.result()
                        evaluations.append(evaluation)
                        append_metrics_jsonl(metrics_file, evaluation)
                except Exception as error:
                    pending = [
                        evaluation_futures[future]
                        for future in evaluation_futures if not future.done()
                    ]
                    error_record = {
                        "record_type": "policy_evaluation_error",
                        "update": int(update_id),
                        "training_steps": int(training_steps),
                        "training_episodes": int(training_episodes),
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "pending_evaluations": pending,
                        "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                    }
                    append_metrics_jsonl(metrics_file, error_record)
                    print(
                        "POLICY_EVALUATION_ERROR",
                        json.dumps(error_record, sort_keys=True), flush=True,
                    )
                    for future in evaluation_futures:
                        future.cancel()
                    raise
                evaluations.sort(key=lambda row: int(row["evaluation_episode"]))
                evaluation_rewards = np.asarray(
                    [row["evaluation_total_reward"] for row in evaluations],
                    dtype=np.float64,
                )
                success_count = sum(bool(row["success"]) for row in evaluations)
                evaluation_summary = {
                    "record_type": "policy_evaluation_summary",
                    "update": int(update_id),
                    "training_steps": int(training_steps),
                    "training_episodes": int(training_episodes),
                    "evaluation_episodes": int(len(evaluations)),
                    "evaluation_mode": "concurrent",
                    "evaluation_workers": [int(row["evaluation_worker"]) for row in evaluations],
                    "mean_reward": float(np.mean(evaluation_rewards)),
                    "reward_std": float(np.std(evaluation_rewards)),
                    "min_reward": float(np.min(evaluation_rewards)),
                    "max_reward": float(np.max(evaluation_rewards)),
                    "success_count": int(success_count),
                    "success_rate": float(success_count / len(evaluations)),
                }
                append_metrics_jsonl(metrics_file, evaluation_summary)
                print("POLICY_EVALUATION_SUMMARY", json.dumps(evaluation_summary, sort_keys=True), flush=True)
                output = update_evaluation_plot(metrics_file, plot_file, args.plot_window)
                print("evaluation_curve_updated", output, flush=True)

                payload = {
                    "update": update_id,
                    "training_steps": training_steps,
                    "training_episodes": training_episodes,
                    "trainer": trainer.state_dict(),
                    "stage_buffer": global_buffer.state_dict(include_active=False),
                    "samples": counts,
                    "workers": args.workers,
                    "last_evaluation": evaluations[-1],
                    "last_evaluation_summary": evaluation_summary,
                    "share_policy_by_type": args.share_policy_by_type,
                    "initialization": dict(initialization),
                    "metrics_file": metrics_file,
                    "plot_file": plot_file,
                }
                _, latest = save_training_checkpoint(
                    payload, checkpoint_dir,
                    "bottom_happo_recon_attack_parallel_eval", update_id,
                )
                print("parallel_eval_checkpoint", latest, "plot", plot_file, flush=True)
            else:
                print(
                    "shared_buffer_waiting", global_buffer.assigned_counts(),
                    "required", minimum, flush=True,
                )

            if update_id >= args.updates:
                break
            restarting = [
                worker for worker in active
                if worker["episode"] < args.episodes_per_worker
            ]
            if restarting:
                start_workers_parallel(executor, args, restarting)

        return 0 if update_id >= args.updates else 2
    finally:
        executor.shutdown(wait=True)
        for worker in workers:
            try:
                task_command(args, worker["id"], "stop")
            except Exception:
                pass
            worker["env"].close()


if __name__ == "__main__":
    raise SystemExit(main())


