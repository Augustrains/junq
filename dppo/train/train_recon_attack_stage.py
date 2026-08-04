"""Train reconnaissance and attack policies until a good landing boundary."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.afsim_env import AFSIMIslandEnv
from envs.rl_interface import AFSIMRLInterface
from train.checkpointing import append_metrics_jsonl, load_training_checkpoint, save_training_checkpoint
from train.decision_timing import apply_bottom_decision_timing, resolve_bottom_decision_timing
from train.happo_trainer import HAPPOConfig, HAPPOTrainer
from train.rule_driven_rollout_collector import RuleDrivenRolloutCollector
from train.recon_attack_stage import canonicalize_landing_snapshot, high_quality_landing_status, recon_attack_terminal_bonus
from train.stage_episode_buffer import TaskTrajectoryBuffer
from train.stage_snapshots import StageSnapshotPool, capture_stage_snapshot


def parse_args():
    parser = argparse.ArgumentParser(description="Recon+attack HAPPO stage with landing-boundary snapshots.")
    parser.add_argument("--algorithm", choices=("happo",), default="happo")
    parser.add_argument("--curriculum-stage", choices=("recon_attack",), default="recon_attack")
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--target-update", type=int, default=0)
    parser.add_argument("--terminal-exit-code", type=int, default=75)
    parser.add_argument("--episodes", type=int, default=1, help="Number of AFSIM games to run in this persistent Python process.")
    parser.add_argument("--eval-episodes", type=int, default=int(os.getenv("EVAL_EPISODES", "1")), help="Fresh AFSIM evaluation games after each policy update; evaluation never enters the training buffer.")
    parser.add_argument("--warlock-ssh-target", default="")
    parser.add_argument("--warlock-ssh-port", type=int, default=22)
    parser.add_argument("--warlock-ssh-key", default="")
    parser.add_argument("--warlock-task-name", default="AFSIM-Warlock")
    parser.add_argument("--start-remote-warlock", action="store_true")
    parser.add_argument("--warlock-start-delay", type=float, default=0.0)
    parser.add_argument("--warlock-stop-delay", type=float, default=2.0)
    parser.add_argument("--rollout-steps", type=int, default=128, help="Compatibility option; this stage collects one decision at a time.")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--bind", action="store_true")
    parser.add_argument("--auto-start-warlock", action="store_true")
    parser.add_argument("--config-path", default="")
    parser.add_argument("--local-address", default="0.0.0.0:50050")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bottom-decisions-per-hour", type=float, default=50.0)
    parser.add_argument("--simulation-clock-rate", type=float, default=40.0, help="AFSIM simulation seconds per wall-clock second; all decision pacing is derived from this value.")
    parser.add_argument("--decision-seconds", type=float, default=0.0, help="Explicit wall-clock interval override.")
    parser.add_argument("--enable-negative-rewards", action="store_true", help="Enable negative reward terms; disabled by default.")
    parser.add_argument("--platform-timeout", type=float, default=120.0)
    parser.add_argument("--platform-state-stall-seconds", type=float, default=float(os.getenv("PLATFORM_STATE_STALL_SECONDS", "30")), help="Fail fast if no PlatformState/MoveUpdate heartbeat arrives for this many wall-clock seconds.")
    parser.add_argument("--checkpoint-dir", default=str(ROOT / "checkpoints" / "recon_attack_stage"))
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--resume", default="")
    parser.add_argument("--reset-resume-buffer", action="store_true", help="Load policy/optimizer state but discard unfinished trajectories saved in the checkpoint.")
    parser.add_argument("--metrics-file", default="")
    parser.add_argument("--plot-file", default="")
    parser.add_argument("--plot-every-episodes", type=int, default=0)
    parser.add_argument("--plot-moving-average-window", type=int, default=10)
    parser.add_argument("--curriculum-seed", type=int, default=0)
    parser.add_argument("--recon-min-samples", type=int, default=int(os.getenv("RECON_MIN_SAMPLES", "2048")))
    parser.add_argument("--attack-min-samples", type=int, default=int(os.getenv("ATTACK_MIN_SAMPLES", "2048")))
    parser.add_argument("--max-episode-steps", type=int, default=int(os.getenv("MAX_EPISODE_STEPS", "5000")))
    parser.add_argument("--deadline-miss-penalty", type=float, default=float(os.getenv("RECON_ATTACK_DEADLINE_MISS_PENALTY", "-50")))
    parser.add_argument("--bottom-global-reward-weight", type=float, default=float(os.getenv("BOTTOM_GLOBAL_REWARD_WEIGHT", "0.1")))
    parser.add_argument("--bottom-global-reward-clip", type=float, default=float(os.getenv("BOTTOM_GLOBAL_REWARD_CLIP", "10.0")))
    parser.add_argument("--snapshot-pool", default=os.getenv("SNAPSHOT_POOL", str(ROOT / "stage_snapshots")))
    parser.add_argument("--snapshot-pool-size", type=int, default=int(os.getenv("SNAPSHOT_POOL_SIZE", "50")))
    parser.add_argument("--snapshot-settle-seconds", type=float, default=float(os.getenv("SNAPSHOT_SETTLE_SECONDS", "10")))
    parser.add_argument("--min-recon-alive", type=int, default=3)
    parser.add_argument("--min-attack-alive", type=int, default=3)
    parser.add_argument("--min-loaded-transports", type=int, default=1)
    parser.add_argument(
        "--share-policy-by-type",
        action="store_true",
        help="Use one shared actor/critic and one pooled update batch per unit type.",
    )
    return parser.parse_args()


def _address(value):
    host, port = value.rsplit(":", 1)
    return host, int(port)


def _device(value):
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def _new_episode_diagnostics(specs, env):
    fire_action_ids = {
        int(action.get("id", index))
        for index, action in enumerate(specs["attack"].get("action_table", []))
        if str(action.get("name", "")).startswith("ATTACK_TARGET")
    }
    recon_return_action_ids = {
        int(action.get("id", index))
        for index, action in enumerate(specs["recon"].get("action_table", []))
        if str(action.get("name", "")) == "RETURN_HOME"
    }
    return {
        "action_counts": {"recon": {}, "attack": {}},
        "assigned_action_counts": {"recon": {}, "attack": {}},
        "fallback_to_hold": {"recon": 0, "attack": 0},
        "recon_return_action_ids": sorted(recon_return_action_ids),
        "recon_return_requested": 0,
        "recon_return_executed": 0,
        "recon_return_sent": 0,
        "recon_first_detection_events": 0,
        "attack_fire_action_ids": sorted(fire_action_ids),
        "attack_fire_mask_steps": 0,
        "attack_fire_mask_slots": 0,
        "attack_fire_requested": 0,
        "attack_fire_executed": 0,
        "attack_fire_sent": 0,
        "attack_result_counts": {},
        "damage_dealt_events": 0,
        "team_reward_term_counts": {},
        "team_reward_term_sums": {},
        "local_reward_term_counts": {},
        "local_reward_term_sums": {},
        "blue_air_destroyed_start": int(env._destroyed_count("blue", "attack_aircraft")),
        "blue_sam_destroyed_start": int(env._destroyed_count("blue", "sam")),
    }


def _increment(counter, key, value=1):
    counter[key] = counter.get(key, 0) + value


def _update_episode_diagnostics(diagnostics, rollout):
    fire_ids = set(diagnostics["attack_fire_action_ids"])
    for agent_type in ("recon", "attack"):
        for row in rollout.bottom[agent_type].rows:
            action_name = str(row.get("action_name", row.get("action", "UNKNOWN")))
            _increment(diagnostics["action_counts"][agent_type], action_name)
            if bool(row.get("assigned", False)):
                _increment(
                    diagnostics["assigned_action_counts"][agent_type], action_name
                )
            if bool(row.get("fallback_to_hold", False)):
                diagnostics["fallback_to_hold"][agent_type] += 1
            if not bool(row.get("assigned", False)):
                continue
            requested_id = int(row.get("requested_action_id", row.get("action", 0)))
            executed_id = int(row.get("executed_action_id", requested_id))
            if agent_type == "recon":
                return_ids = set(diagnostics["recon_return_action_ids"])
                if requested_id in return_ids:
                    diagnostics["recon_return_requested"] += 1
                if executed_id in return_ids:
                    diagnostics["recon_return_executed"] += 1
                    if bool(row.get("sent", False)):
                        diagnostics["recon_return_sent"] += 1
                continue
            mask = row.get("action_mask", [])
            available_fire = sum(
                1 for action_id in fire_ids
                if action_id < len(mask) and float(mask[action_id]) > 0.0
            )
            if available_fire:
                diagnostics["attack_fire_mask_steps"] += 1
                diagnostics["attack_fire_mask_slots"] += available_fire
            if requested_id in fire_ids:
                diagnostics["attack_fire_requested"] += 1
            if executed_id in fire_ids:
                diagnostics["attack_fire_executed"] += 1
                if bool(row.get("sent", False)):
                    diagnostics["attack_fire_sent"] += 1

    for step_info in rollout.step_infos:
        for event in step_info.get("events", []):
            event_type = str(event.get("type", ""))
            if event_type in ("new_detection", "first_detection"):
                diagnostics["recon_first_detection_events"] += 1
            elif event_type == "attack_result":
                _increment(
                    diagnostics["attack_result_counts"],
                    str(event.get("result", "UNKNOWN")),
                )
            elif event_type == "damage_dealt":
                diagnostics["damage_dealt_events"] += 1
        for detail in step_info.get("reward_details", []):
            term = str(detail.get("type", "UNKNOWN"))
            _increment(diagnostics["team_reward_term_counts"], term)
            _increment(
                diagnostics["team_reward_term_sums"],
                term,
                float(detail.get("value", 0.0)),
            )
        for reward_info in step_info.get("bottom_rewards", {}).values():
            for detail in reward_info.get("local_reward_details", []):
                term = str(detail.get("type", "UNKNOWN"))
                _increment(diagnostics["local_reward_term_counts"], term)
                _increment(
                    diagnostics["local_reward_term_sums"],
                    term,
                    float(detail.get("value", 0.0)),
                )


def _finish_episode_diagnostics(diagnostics, env):
    diagnostics["blue_air_destroyed_end"] = int(
        env._destroyed_count("blue", "attack_aircraft")
    )
    diagnostics["blue_sam_destroyed_end"] = int(
        env._destroyed_count("blue", "sam")
    )
    diagnostics["blue_air_kills_this_episode"] = max(
        0,
        diagnostics["blue_air_destroyed_end"]
        - diagnostics["blue_air_destroyed_start"],
    )
    diagnostics["blue_sam_kills_this_episode"] = max(
        0,
        diagnostics["blue_sam_destroyed_end"]
        - diagnostics["blue_sam_destroyed_start"],
    )
    term_counts = diagnostics["local_reward_term_counts"]
    diagnostics["recon_navigation_reward_steps"] = int(
        term_counts.get("outside_recon_area_distance_norm_delta", 0)
    )
    diagnostics["recon_coverage_reward_steps"] = int(
        term_counts.get("coverage_credit_ratio_increment", 0)
    )
    diagnostics["recon_return_reward_steps"] = int(
        term_counts.get("return_distance_to_carrier_norm_delta", 0)
    )
    diagnostics["recon_area_status"] = {
        env._recon_area_name(area): {
            "coverage_ratio": float(env._recon_area_coverage_ratio(area)),
            "age_norm": float(env._recon_area_age_norm(area)),
            "complete": bool(env._recon_area_complete(area)),
            "covered_cells": int(len(env.recon_area_coverage.get(
                env._recon_area_name(area), set()
            ))),
            "valid_cells": int(len(env._recon_valid_cells(area))),
        }
        for area in env.recon_areas
    }
    return diagnostics


def _update_training_curve(metrics_file, plot_file, moving_average_window):
    if not plot_file:
        return
    from train.plot_recon_attack_training import load_episode_metrics, load_policy_evaluation_metrics, plot_training_curve

    rows = load_episode_metrics(metrics_file)
    if not rows:
        return
    output = plot_training_curve(rows, plot_file, moving_average_window, load_policy_evaluation_metrics(metrics_file))
    print("training_curve_updated", output, "episodes", len(rows), flush=True)


def _run_remote_warlock_command(args, command):
    if not args.warlock_ssh_target:
        raise RuntimeError("--warlock-ssh-target is required when --episodes is greater than 1")
    ssh_args = ["ssh", "-p", str(args.warlock_ssh_port)]
    if args.warlock_ssh_key:
        ssh_args.extend(["-i", args.warlock_ssh_key])
    ssh_args.extend(["-o", "BatchMode=yes", "-o", "ConnectTimeout=15", args.warlock_ssh_target, command])
    subprocess.run(ssh_args, check=True)


def _stop_remote_warlock(args):
    quoted_task = str(args.warlock_task_name).replace("'", "''")
    stop_command = (
        "powershell -NoProfile -Command \"Stop-ScheduledTask -TaskName '{0}' -ErrorAction SilentlyContinue; "
        "Get-Process warlock,wizard -ErrorAction SilentlyContinue | Stop-Process -Force; exit 0\""
    ).format(quoted_task)
    _run_remote_warlock_command(args, stop_command)


def _restart_remote_warlock(env, args, initial=False):
    quoted_task = str(args.warlock_task_name).replace("'", "''")
    start_command = (
        "powershell -NoProfile -Command \"Start-ScheduledTask -TaskName '{0}'\""
    ).format(quoted_task)
    _stop_remote_warlock(args)
    time.sleep(max(0.0, float(args.warlock_stop_delay)))
    env._drain_messages(timeout=0.5)
    env.prepare_for_scenario_restart()
    _run_remote_warlock_command(args, start_command)
    time.sleep(max(0.0, float(args.warlock_start_delay)))
    ready = env.wait_for_platforms(list(env.platforms.keys()), timeout=float(args.platform_timeout))
    known = sum(platform.platform_id is not None for platform in env.platforms.values())
    label = "live_platforms_ready" if initial else "live_platforms_restarted"
    print(
        label,
        ready,
        "known_count", known,
        "sim_time", round(float(env._current_sim_time()), 3),
        flush=True,
    )
    if not ready:
        raise RuntimeError("AFSIM platforms did not re-register after Warlock restart")

def main():
    args = parse_args()
    device = _device(args.device)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = args.metrics_file or str(checkpoint_dir / "metrics.jsonl")
    minimums = {"recon": int(args.recon_min_samples), "attack": int(args.attack_min_samples)}
    env = AFSIMIslandEnv(
        config_path=args.config_path or None,
        bind=bool(args.bind),
        auto_start_warlock=bool(args.auto_start_warlock),
        local_address=_address(args.local_address),
    )
    env.set_negative_rewards_enabled(args.enable_negative_rewards)
    print("negative_rewards_enabled", env.negative_rewards_enabled)
    decision_timing = resolve_bottom_decision_timing(
        args.bottom_decisions_per_hour, args.simulation_clock_rate, args.decision_seconds)
    apply_bottom_decision_timing(env, decision_timing)
    print("bottom_decision_timing",
          "decisions_per_sim_hour", decision_timing["decisions_per_sim_hour"],
          "simulation_interval_seconds", decision_timing["simulation_interval_seconds"],
          "simulation_clock_rate", decision_timing["simulation_clock_rate"],
          "wall_interval_seconds", decision_timing["wall_interval_seconds"],
          "explicit_wall_override", decision_timing["explicit_wall_override"])
    try:
        api = AFSIMRLInterface(env, reward_profile="recon_attack_stage")
        specs = api.get_bottom_agent_specs()
        bottom_specs = {name: specs[name] for name in ("recon", "attack")}
        trainer = HAPPOTrainer(
            bottom_specs,
            global_state_dim=int(specs["global"]["obs_dim"]),
            agent_types=("recon", "attack"),
            hidden_sizes=(args.hidden_size, args.hidden_size),
            config=HAPPOConfig(
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                learning_rate=args.lr,
                update_epochs=args.update_epochs,
                minibatch_size=args.minibatch_size,
                share_policy_by_type=args.share_policy_by_type,
            ),
            device=device,
            trainable_agent_types=("recon", "attack"),
        )
        print("happo_implementation", trainer.implementation, trainer.upstream_repository, trainer.upstream_commit)
        buffer = TaskTrajectoryBuffer(("recon", "attack"))
        update_id = 0
        episode_id = 1
        behavior_checkpoint = args.resume
        if args.resume:
            checkpoint = load_training_checkpoint(args.resume)
            trainer.load_state_dict(checkpoint.get("trainer", checkpoint))
            if args.reset_resume_buffer:
                print("resume_buffer_reset", "discarded", checkpoint.get("stage_buffer_counts_before_update", {}))
            else:
                buffer.load_state_dict(checkpoint.get("stage_buffer", {}))
            update_id = int(checkpoint.get("update", 0))
            episode_id = int(checkpoint.get("episode", 0)) + 1
            buffer.episode_index = max(int(buffer.episode_index), int(episode_id))
            print(
                "resumed", args.resume, "update", update_id, "episode", episode_id,
                "samples", buffer.assigned_counts(),
            )
        target_update = int(args.target_update) if args.target_update > 0 else update_id + int(args.updates)
        print("training_device", device)
        print(
            "stage", "recon_attack", "minimum_steps", minimums,
            "update_trigger", "sample_threshold",
            "target_update", target_update,
        )
        print(
            "linux_control_ready",
            "udp_bound", bool(env.sock),
            "networks_initialized", True,
            "resume_loaded", bool(args.resume),
            flush=True,
        )
        if args.start_remote_warlock:
            _restart_remote_warlock(env, args, initial=True)
        elif args.bind:
            ready = env.wait_for_platforms(list(env.platforms.keys()), timeout=args.platform_timeout)
            print(
                "live_platforms_ready",
                ready,
                "known_count", sum(p.platform_id is not None for p in env.platforms.values()),
                "sim_time", round(float(env._current_sim_time()), 3),
                flush=True,
            )
            if not ready:
                return 1

        def update_if_ready(trigger, current_episode_id):
            nonlocal update_id, behavior_checkpoint
            if not buffer.ready(minimums):
                return None
            counts_before_update = buffer.assigned_counts()
            metrics = trainer.update_all(buffer.to_batches())
            update_id += 1
            buffer.clear_after_update()
            payload = {
                "update": update_id,
                "episode": int(current_episode_id),
                "episode_terminal": False,
                "done_reason": "policy_update",
                "curriculum_stage": "recon_attack",
                "trainable_agent_types": ["recon", "attack"],
                "trainer": trainer.state_dict(),
                # A restarted simulator cannot resume in-flight task groups.
                # Completed rows were consumed and active prefixes were cut at
                # this policy boundary, so omit active task metadata here.
                "stage_buffer": buffer.state_dict(include_active=False),
                "stage_buffer_counts_before_update": counts_before_update,
                "update_trigger": str(trigger),
            }
            _ckpt, latest = save_training_checkpoint(
                payload, checkpoint_dir, "bottom_happo_recon_attack", update_id
            )
            behavior_checkpoint = str(latest)
            append_metrics_jsonl(metrics_file, {
                "record_type": "policy_update",
                "update": update_id,
                "episode": int(current_episode_id),
                "trigger": str(trigger),
                "samples": counts_before_update,
                "metrics": metrics,
            })
            print(
                "happo_update", update_id,
                "trigger", trigger,
                "samples", counts_before_update,
                "metrics", metrics,
                flush=True,
            )
            return {
                "update": update_id,
                "trigger": str(trigger),
                "samples": counts_before_update,
                "metrics": metrics,
            }

        def evaluate_policy(current_episode_id, current_update):
            """Run one fresh post-update game without changing training data."""
            collector = RuleDrivenRolloutCollector(
                api,
                bottom_global_reward_weight=args.bottom_global_reward_weight,
                bottom_global_reward_clip=args.bottom_global_reward_clip,
                bottom_agent_types=("recon", "attack"),
            )
            api.reset_rule_driven()
            started_at = time.monotonic()
            team_reward = 0.0
            terminal_reason = "stage_time_limit"
            status = high_quality_landing_status(
                env, args.min_recon_alive, args.min_attack_alive, args.min_loaded_transports
            )
            for _step in range(1, int(args.max_episode_steps) + 1):
                rollout = collector.collect(trainer, n_steps=1, reset=False)
                state_age = env.platform_state_age_seconds()
                if state_age > float(args.platform_state_stall_seconds):
                    raise RuntimeError("AFSIM platform-state stream stalled during evaluation for {0:.1f}s".format(state_age))
                team_reward += sum(float(item.get("team_reward", 0.0)) for item in rollout.step_infos)
                summary = rollout.summary()
                status = high_quality_landing_status(
                    env, args.min_recon_alive, args.min_attack_alive, args.min_loaded_transports
                )
                if status["landing_combat_conditions_met"]:
                    terminal_reason = "combat_landing_ready"
                    break
                if status["landing_time_override_met"] and not status["landing_combat_conditions_met"]:
                    terminal_reason = "landing_deadline_missed"
                    break
                if summary.get("terminal", False):
                    terminal_reason = summary.get("done_reason", "environment_terminal")
                    break
            bonus = recon_attack_terminal_bonus(
                status, terminal_reason, env._current_sim_time(),
                deadline_miss_penalty=args.deadline_miss_penalty,
                negative_rewards_enabled=env.negative_rewards_enabled,
            )
            result = {
                "record_type": "policy_evaluation",
                "update": int(current_update),
                "episode": int(current_episode_id),
                "done_reason": terminal_reason,
                "success": terminal_reason == "combat_landing_ready",
                "evaluation_team_reward": float(team_reward),
                "evaluation_terminal_reward": float(bonus),
                "evaluation_total_reward": float(team_reward + bonus),
                "evaluation_sim_seconds": float(env._current_sim_time()),
                "evaluation_wall_seconds": max(0.0, time.monotonic() - started_at),
            }
            print("POLICY_EVALUATION", result, flush=True)
            return result
        episodes_to_run = max(1, int(args.episodes))
        for managed_episode_index in range(episodes_to_run):
            policy_set = trainer
            collector = RuleDrivenRolloutCollector(
                api,
                bottom_global_reward_weight=args.bottom_global_reward_weight,
                bottom_global_reward_clip=args.bottom_global_reward_clip,
                bottom_agent_types=("recon", "attack"),
            )
            api.reset_rule_driven()
            print(
                "episode_control_ready",
                "episode", episode_id,
                "sim_time", round(float(env._current_sim_time()), 3),
                flush=True,
            )
            boundary = None
            terminal_reason = ""
            episode_team_reward = 0.0
            episode_updates = []
            episode_diagnostics = _new_episode_diagnostics(specs, env)
            episode_wall_started_at = time.monotonic()
            for step in range(1, int(args.max_episode_steps) + 1):
                rollout = collector.collect(policy_set, n_steps=1, reset=False)
                state_age = env.platform_state_age_seconds()
                if state_age > float(args.platform_state_stall_seconds):
                    raise RuntimeError("AFSIM platform-state stream stalled for {0:.1f}s".format(state_age))
                _update_episode_diagnostics(episode_diagnostics, rollout)
                buffer.append_rollout(rollout)
                episode_team_reward += sum(float(info.get("team_reward", 0.0)) for info in rollout.step_infos)
                summary = rollout.summary()
                status = high_quality_landing_status(
                    env,
                    min_recon_alive=args.min_recon_alive,
                    min_attack_alive=args.min_attack_alive,
                    min_loaded_transports=args.min_loaded_transports,
                )
                combat_landing_ready = bool(status["landing_combat_conditions_met"])
                if combat_landing_ready:
                    boundary = status
                    terminal_reason = "combat_landing_ready"
                    break
                if status["landing_time_override_met"] and not status["landing_combat_conditions_met"]:
                    terminal_reason = "landing_deadline_missed"
                    break
                if summary.get("terminal", False):
                    terminal_reason = summary.get("done_reason", "environment_terminal")
                    break
                if step % 100 == 0:
                    print(
                        "episode_progress", episode_id, "step", step,
                        "samples", buffer.assigned_counts(),
                        "active_samples", buffer.active_counts(),
                        "updates_this_episode", len(episode_updates),
                        "landing", status,
                    )
            else:
                terminal_reason = "stage_time_limit"

            bonus = 0.0
            snapshot_path = ""
            settlement_reward = 0.0
            if boundary is not None:
                # Stop issuing actions, settle in-flight weapon events, and
                # backfill target credit to every retained contributor before
                # applying the single global stage-success outcome.
                env.last_reward_events = []
                env._drain_messages(timeout=max(0.0, float(args.snapshot_settle_seconds)))
                env.compute_reward()
                settlement_events = list(env.last_reward_events)
                settlement_details = list(env.last_reward_details)
                for agent_type in ("recon", "attack"):
                    settlement_state = api.get_persistent_agent_state(agent_type)
                    settlement_info = api.build_post_step_bottom_reward_info(
                        agent_type,
                        settlement_state,
                        settlement_state,
                        {},
                        0.0,
                        reward_events=settlement_events,
                        reward_details=settlement_details,
                    )
                    buffer.apply_delayed_target_rewards(
                        agent_type,
                        settlement_info.get("target_contribution_rewards", {}),
                        float(settlement_info.get("local_reward_weight", 0.0)),
                    )
                bonus = recon_attack_terminal_bonus(
                    boundary,
                    terminal_reason,
                    env._current_sim_time(),
                    deadline_miss_penalty=args.deadline_miss_penalty,
                    negative_rewards_enabled=env.negative_rewards_enabled,
                )
                canonicalize_landing_snapshot(env)
                snapshot = capture_stage_snapshot(
                    env,
                    "landing",
                    policy_checkpoint=behavior_checkpoint,
                    source_scenario=env.config.get("scenario", {}).get("scenario_file", "scenarios/island_assault_min.txt"),
                    require_quiescent=True,
                    tags=("recon_attack_success", "episode_{0}".format(episode_id), "behavior_update_{0}".format(update_id)),
                )
                snapshot_path = str(StageSnapshotPool(
                    args.snapshot_pool, "landing", max_size=args.snapshot_pool_size, seed=args.curriculum_seed
                ).add(snapshot))
                print("landing_snapshot_saved", snapshot_path, "quality", boundary, "terminal_bonus", round(bonus, 4))
            elif terminal_reason == "landing_deadline_missed":
                bonus = recon_attack_terminal_bonus(
                    status,
                    terminal_reason,
                    env._current_sim_time(),
                    deadline_miss_penalty=args.deadline_miss_penalty,
                    negative_rewards_enabled=env.negative_rewards_enabled,
                )
                print(
                    "landing_deadline_penalty",
                    "sim_time", round(float(env._current_sim_time()), 3),
                    "configured_penalty", float(args.deadline_miss_penalty),
                    "terminal_bonus", round(float(bonus), 4),
                    "quality", status,
                )
            buffer.apply_episode_global_reward(bonus)
            buffer.finish_episode(
                terminal_reward_bonus=0.0,
                end_reason=terminal_reason or "episode_terminal",
            )
            terminal_samples = buffer.assigned_counts()
            terminal_update = update_if_ready("episode_force_close", episode_id)
            if terminal_update is not None:
                episode_updates.append(terminal_update)
            episode_sim_seconds = float(env._current_sim_time())
            episode_wall_seconds = max(0.0, time.monotonic() - episode_wall_started_at)
            episode_total_reward = episode_team_reward + bonus
            episode_diagnostics = _finish_episode_diagnostics(
                episode_diagnostics, env
            )
            stage_boundary_reward = bonus

            remaining_samples = buffer.assigned_counts()
            updated = bool(episode_updates)
            metrics = (
                dict(episode_updates[-1].get("metrics", {}))
                if episode_updates else {}
            )
            if not updated:
                print(
                    "happo_waiting_for_tasks",
                    "samples", remaining_samples, "required", minimums,
                )

            evaluation_records = []
            if updated and int(args.eval_episodes) > 0:
                if not args.warlock_ssh_target:
                    raise RuntimeError("post-update evaluation requires --warlock-ssh-target for a fresh AFSIM scenario")
                for evaluation_index in range(1, int(args.eval_episodes) + 1):
                    _restart_remote_warlock(env, args)
                    evaluation = evaluate_policy(episode_id, update_id)
                    evaluation["evaluation_index"] = evaluation_index
                    evaluation_records.append(evaluation)
                    append_metrics_jsonl(metrics_file, evaluation)
            payload = {
                "update": update_id,
                "episode": episode_id,
                "episode_terminal": True,
                "done_reason": terminal_reason,
                "curriculum_stage": "recon_attack",
                "trainable_agent_types": ["recon", "attack"],
                "trainer": trainer.state_dict(),
                "stage_buffer": buffer.state_dict(),
                "stage_buffer_counts_before_update": terminal_samples,
                "stage_buffer_remaining_counts": remaining_samples,
                "episode_updates": episode_updates,
                "policy_evaluations": evaluation_records,
                "landing_success": boundary is not None,
                "landing_snapshot": snapshot_path,
                "landing_quality": boundary or {},
                "episode_summary": summary,
                "episode_sim_seconds": episode_sim_seconds,
                "episode_wall_seconds": episode_wall_seconds,
                "episode_diagnostics": episode_diagnostics,
                "episode_team_reward": episode_team_reward,
                "stage_boundary_reward": stage_boundary_reward,
                "settlement_reward": settlement_reward,
                "episode_total_reward": episode_total_reward,
            }
            ckpt, latest = save_training_checkpoint(payload, checkpoint_dir, "bottom_happo_recon_attack", update_id)
            append_metrics_jsonl(metrics_file, {
                "update": update_id,
                "episode": episode_id,
                "done_reason": terminal_reason,
                "updated": updated,
                "samples": terminal_samples,
                "remaining_samples": remaining_samples,
                "policy_updates": episode_updates,
                "policy_evaluations": evaluation_records,
                "metrics": metrics,
                "landing_success": boundary is not None,
                "landing_snapshot": snapshot_path,
                "landing_quality": boundary or {},
                "terminal_bonus": bonus,
                "episode_sim_seconds": episode_sim_seconds,
                "episode_wall_seconds": episode_wall_seconds,
                "episode_team_reward": episode_team_reward,
                "stage_boundary_reward": stage_boundary_reward,
                "settlement_reward": settlement_reward,
                "episode_total_reward": episode_total_reward,
                "final_score_raw": summary.get("final_score_raw", 0.0),
                "final_score_norm": summary.get("final_score_norm", 0.0),
                "final_score_unit_count": summary.get("final_score_unit_count", 0),
                "episode_diagnostics": episode_diagnostics,
                "final_score_sim_time": summary.get("final_score_sim_time", 0.0),
            })
            plot_every = max(0, int(args.plot_every_episodes))
            if args.plot_file and plot_every > 0 and (managed_episode_index + 1) % plot_every == 0:
                try:
                    _update_training_curve(
                        metrics_file,
                        args.plot_file,
                        args.plot_moving_average_window,
                    )
                except Exception as exc:
                    print("training_curve_update_failed", repr(exc), flush=True)
            print(
                "episode_diagnostics",
                "episode", episode_id,
                episode_diagnostics,
                flush=True,
            )
            print(
                "EPISODE_END",
                "episode", episode_id,
                "reason", terminal_reason,
                "sim_seconds", round(episode_sim_seconds, 3),
                "sim_hours", round(episode_sim_seconds / 3600.0, 4),
                "wall_seconds", round(episode_wall_seconds, 3),
                "team_reward", round(episode_team_reward, 4),
                "stage_boundary_reward", round(stage_boundary_reward, 4),
                "settlement_reward", round(settlement_reward, 4),
                "terminal_reward", round(bonus, 4),
                "total_reward", round(episode_total_reward, 4),
            )
            print(
                "episode_complete", episode_id, terminal_reason,
                "score_hp", summary.get("final_score_raw", 0.0),
                "score_norm", summary.get("final_score_norm", 0.0),
                "score_units", summary.get("final_score_unit_count", 0),
                "sim_time", episode_sim_seconds,
                "saved", ckpt, "latest", latest,
            )
            print(
                "GAME_COMPLETE",
                "game", managed_episode_index + 1, "of", episodes_to_run,
                "episode", episode_id,
                "success", boundary is not None,
                "snapshot", snapshot_path,
            )
            if managed_episode_index + 1 < episodes_to_run:
                _restart_remote_warlock(env, args)
                episode_id += 1
                continue
            return 0
    finally:
        if args.warlock_ssh_target:
            try:
                print("python_cleanup_windows_warlock", args.warlock_ssh_target, flush=True)
                _stop_remote_warlock(args)
            except Exception as exc:
                print("python_cleanup_windows_warlock_failed", repr(exc), flush=True)
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
