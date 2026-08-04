"""Rule-driven bottom-level MAPPO/HAPPO training entry point."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from envs.afsim_env import AFSIMIslandEnv
from envs.rl_interface import AFSIMRLInterface
from train.checkpointing import append_metrics_jsonl, load_training_checkpoint, save_training_checkpoint
from train.decision_timing import apply_bottom_decision_timing, resolve_bottom_decision_timing
from train.curriculum_stages import curriculum_stage_names, get_curriculum_stage
from train.happo_trainer import HAPPOTrainer, HAPPOConfig
from train.rule_driven_rollout_collector import RuleDrivenRolloutCollector
from train.stage_snapshot_scenarios import SnapshotRestoringInterface


def parse_args():
    parser = argparse.ArgumentParser(description="Train flat always-on bottom policies with MAPPO or HAPPO.")
    parser.add_argument("--algorithm", choices=["happo"], default="happo", help="Bottom MARL optimizer.")
    parser.add_argument("--curriculum-stage", choices=curriculum_stage_names(), default="recon_only", help="Staged-training policy/task set.")
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--target-update", type=int, default=0, help="Absolute final update id; supports restart/resume loops.")
    parser.add_argument("--terminal-exit-code", type=int, default=0, help="Process exit code used when an episode ends before target-update.")
    parser.add_argument("--rollout-steps", type=int, default=16, help="Environment steps collected per rollout chunk.")
    parser.add_argument("--update-after-steps", type=int, default=1024, help="Accumulate this many environment decision steps before one network update.")
    parser.add_argument("--evaluation-steps", type=int, default=0, help="Post-update evaluation steps; 0 uses the environment maximum horizon.")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--bind", action="store_true", help="Bind UDP socket for live AFSIM/Warlock interaction.")
    parser.add_argument("--auto-start-warlock", action="store_true")
    parser.add_argument("--config-path", default="", help="Optional env config JSON path.")
    parser.add_argument("--stage-snapshot", default="", help="Dynamic landing/ground stage snapshot JSON restored after reset.")
    parser.add_argument("--local-address", default="", help="Override UDP bind address as HOST:PORT, e.g. 0.0.0.0:50050.")
    parser.add_argument("--device", default="auto", help="Training device: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--bottom-decisions-per-hour", type=float, default=50.0, help="Bottom-network decisions per simulation hour.")
    parser.add_argument("--simulation-clock-rate", type=float, default=40.0, help="AFSIM simulation seconds per wall-clock second; all decision pacing is derived from this value.")
    parser.add_argument("--decision-seconds", type=float, default=0.0, help="Explicit wall-clock interval override; <=0 derives it from frequency and clock rate.")
    parser.add_argument("--enable-negative-rewards", action="store_true", help="Enable negative reward terms; disabled by default.")
    parser.add_argument("--platform-timeout", type=float, default=60.0, help="Seconds to wait for AFSIM platform registration in live mode.")
    parser.add_argument("--checkpoint-dir", default=str(Path(__file__).resolve().parents[1] / "checkpoints" / "bottom_mappo"))
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--resume", default="", help="Resume from a bottom MAPPO checkpoint.")
    parser.add_argument("--metrics-file", default="", help="JSONL metrics path. Defaults to checkpoint-dir/metrics.jsonl.")
    parser.add_argument("--curriculum-seed", type=int, default=0)
    return parser.parse_args()


def _parse_local_address(value):
    if not value:
        return None
    if ":" not in value:
        raise ValueError("--local-address must use HOST:PORT format")
    host, port = value.rsplit(":", 1)
    return host, int(port)


def _resolve_device(value):
    value = str(value or "auto").lower()
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    return value



class FixedBottomBatchBuffer:
    """Preallocated per-agent-type rollout storage for one policy update."""

    def __init__(self, capacity_steps, specs):
        self.capacity_steps = max(1, int(capacity_steps))
        self.specs = specs
        self._arrays = {}
        self._write_rows = {}
        self._entity_counts = {
            name: max(1, int(specs[name].get("num_entities", 1)))
            for name in specs
        }

    def append(self, bottom_batches):
        for agent_type, batch in bottom_batches.items():
            if not batch or "obs" not in batch:
                continue
            rows = int(len(batch["obs"]))
            if rows == 0:
                continue
            capacity_rows = self.capacity_steps * self._entity_counts.get(agent_type, 1)
            current = int(self._write_rows.get(agent_type, 0))
            if current + rows > capacity_rows:
                raise RuntimeError(
                    "fixed rollout buffer overflow for {0}: {1}+{2}>{3}".format(
                        agent_type, current, rows, capacity_rows
                    )
                )
            arrays = self._arrays.setdefault(agent_type, {})
            for key, values in batch.items():
                values = np.asarray(values)
                if key not in arrays:
                    arrays[key] = np.zeros(
                        (capacity_rows,) + tuple(values.shape[1:]),
                        dtype=values.dtype,
                    )
                target = arrays[key]
                if target.shape[1:] != values.shape[1:]:
                    raise ValueError(
                        "fixed rollout field shape changed for {0}.{1}: {2} vs {3}".format(
                            agent_type, key, target.shape[1:], values.shape[1:]
                        )
                    )
                target[current:current + rows] = values
            self._write_rows[agent_type] = current + rows

    def as_batches(self):
        return {
            agent_type: {
                key: values[:self._write_rows.get(agent_type, 0)]
                for key, values in arrays.items()
            }
            for agent_type, arrays in self._arrays.items()
        }

    def capacity_rows(self):
        return {
            agent_type: self.capacity_steps * count
            for agent_type, count in self._entity_counts.items()
        }

def _evaluation_report(summary):
    bottom = summary.get("bottom", {})
    return {
        "steps": int(summary.get("steps", 0)),
        "team_reward_sum": float(summary.get("team_reward_sum", 0.0)),
        "reward_sum_by_agent": {
            name: float(values.get("reward_sum", 0.0))
            for name, values in bottom.items()
        },
        "terminal": bool(summary.get("terminal", False)),
        "done_reason": summary.get("done_reason", "none"),
        "episode_result": summary.get("episode_result", ""),
        "final_score_sim_time": float(summary.get("final_score_sim_time", 0.0)),
    }

def main():
    args = parse_args()
    stage = get_curriculum_stage(args.curriculum_stage)
    if not stage.scenario_ready and not args.stage_snapshot:
        raise RuntimeError(
            "curriculum stage {0!r} requires a dynamic --stage-snapshot; static template {1} is not used".format(
                stage.name, stage.scenario_template
            )
        )
    device = _resolve_device(args.device)
    print("training_device", device)
    print(
        "curriculum_stage", stage.name,
        "trainable", list(stage.trainable_agent_types),
        "tasks", list(stage.allowed_task_kinds),
        "scenario_template", stage.scenario_template,
        "reward_profile", stage.reward_profile,
    )
    env = AFSIMIslandEnv(
        config_path=args.config_path or None,
        bind=bool(args.bind),
        auto_start_warlock=bool(args.auto_start_warlock),
        local_address=_parse_local_address(args.local_address),
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
    if args.bind:
        ready_names = ["red_recon_1", "red_attack_1", "red_transport_1"]
        ready = env.wait_for_platforms(ready_names, timeout=float(args.platform_timeout))
        known = {name: p.platform_id for name, p in env.platforms.items() if p.platform_id is not None}
        print("live_platforms_ready", ready, "known_count", len(known))
        missing = sorted(name for name, platform in env.platforms.items() if platform.platform_id is None)
        if missing:
            print("missing_platforms", missing)
        if not ready:
            print("known_platforms", known)
            raise SystemExit(1)
    api = AFSIMRLInterface(env)
    if args.stage_snapshot:
        api = SnapshotRestoringInterface(api, args.stage_snapshot)
        print("stage_snapshot", args.stage_snapshot)
    specs = api.get_bottom_agent_specs()
    global_state_dim = int(specs["global"]["obs_dim"])
    bottom_types = ("recon", "attack", "landing", "ground")
    bottom_specs = {name: specs[name] for name in bottom_types}
    config = HAPPOConfig(
        gamma=args.gamma, gae_lambda=args.gae_lambda, learning_rate=args.lr,
        update_epochs=args.update_epochs, minibatch_size=args.minibatch_size,
    )
    trainer = HAPPOTrainer(
        bottom_specs, global_state_dim=global_state_dim,
        hidden_sizes=(args.hidden_size, args.hidden_size), config=config, device=device,
        trainable_agent_types=stage.trainable_agent_types,
    )
    print("bottom_algorithm", args.algorithm)
    print("happo_implementation", trainer.implementation, trainer.upstream_repository, trainer.upstream_commit)
    start_update = 1
    checkpoint = None
    episode_id = 1
    if args.resume:
        checkpoint = load_training_checkpoint(args.resume)
        trainer.load_state_dict(checkpoint.get("trainer", checkpoint))
        previous_stage = str(checkpoint.get("curriculum_stage", "legacy_unspecified"))
        if previous_stage != stage.name:
            print("curriculum_stage_transition", previous_stage, "->", stage.name)
        start_update = int(checkpoint.get("update", 0)) + 1
        episode_id = int(checkpoint.get("episode", 1)) + (1 if checkpoint.get("episode_terminal", False) else 0)
        print("resumed_bottom", args.resume, "start_update", start_update, "episode", episode_id)

    policy_set = trainer
    collector = RuleDrivenRolloutCollector(api)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = args.metrics_file or str(checkpoint_dir / "metrics.jsonl")

    update_after_steps = max(1, int(args.update_after_steps))
    evaluation_steps = int(args.evaluation_steps) if int(args.evaluation_steps) > 0 else int(env.max_steps)
    first_rollout = True
    end_update = int(args.target_update) if int(args.target_update) > 0 else start_update + int(args.updates) - 1
    if start_update > end_update:
        print("training_target_already_reached", "start_update", start_update, "target_update", end_update)
        env.close()
        return 0
    print("update_trigger", "captured_environment_steps", update_after_steps,
          "rollout_chunk_steps", int(args.rollout_steps),
          "evaluation_steps", evaluation_steps)
    exit_code = 0
    for update_id in range(start_update, end_update + 1):
        fixed_buffer = FixedBottomBatchBuffer(update_after_steps, bottom_specs)
        captured_steps = 0
        capture_terminal = False
        capture_episodes = 0
        last_capture_summary = {"steps": 0}
        while captured_steps < update_after_steps:
            rollout = collector.collect(
                policy_set,
                n_steps=min(int(args.rollout_steps), update_after_steps - captured_steps),
                reset=first_rollout,
            )
            first_rollout = False
            summary = rollout.summary()
            last_capture_summary = summary
            chunk_steps = int(summary.get("steps", 0))
            if chunk_steps <= 0:
                raise RuntimeError("rollout collector returned zero environment steps")
            fixed_buffer.append(rollout.to_numpy().get("bottom", {}))
            captured_steps += chunk_steps
            if bool(summary.get("terminal", False)):
                capture_terminal = True
                capture_episodes += 1
                first_rollout = True
            print("capture", "update", update_id, "chunk_steps", chunk_steps,
                  "captured_steps", captured_steps, "terminal", bool(summary.get("terminal", False)))

        batches = fixed_buffer.as_batches()
        metrics = trainer.update_all(batches)
        loss_view = {k: round(v.get("loss", 0.0), 4) for k, v in metrics.items()}
        print("update", update_id, "captured_steps", captured_steps,
              "preallocated_rows", fixed_buffer.capacity_rows(),
              "metrics", loss_view, "episodes_crossed", capture_episodes)

        # Evaluation is deliberately outside the training batch: it uses the
        # updated policy, but its transitions are never passed to update_all.
        evaluation_rollout = collector.collect(
            policy_set, n_steps=evaluation_steps, reset=True
        )
        evaluation_summary = evaluation_rollout.summary()
        evaluation = _evaluation_report(evaluation_summary)
        print("evaluation", "update", update_id,
              "reward_sum_by_agent", evaluation["reward_sum_by_agent"],
              "team_reward_sum", round(evaluation["team_reward_sum"], 6),
              "steps", evaluation["steps"],
              "terminal", evaluation["terminal"],
              "done_reason", evaluation["done_reason"],
              "final_score_sim_time", evaluation["final_score_sim_time"])
        # The next training chunk must start from a fresh episode after eval.
        first_rollout = True
        append_metrics_jsonl(metrics_file, {
            "update": update_id,
            "episode": episode_id,
            "curriculum_stage": stage.name,
            "trainable_agent_types": list(stage.trainable_agent_types),
            "captured_environment_steps": captured_steps,
            "metrics": metrics,
            "evaluation": evaluation,
        })
        periodic_save = args.save_every > 0 and update_id % int(args.save_every) == 0
        if periodic_save or capture_terminal:
            payload = {
                "update": update_id,
                "episode": episode_id,
                "episode_terminal": capture_terminal,
                "curriculum_stage": stage.name,
                "stage_config": stage.as_dict(),
                "trainable_agent_types": list(stage.trainable_agent_types),
                "trainer": trainer.state_dict(),
                "specs": bottom_specs,
                "captured_environment_steps": captured_steps,
                "episode_summary": last_capture_summary,
                "evaluation": evaluation,
            }
            ckpt_path, latest_path = save_training_checkpoint(
                payload, checkpoint_dir, "bottom_happo", update_id
            )
            print("saved", str(ckpt_path), "latest", str(latest_path))
        episode_id += capture_episodes

    env.close()
    return exit_code

if __name__ == "__main__":
    raise SystemExit(main())
