"""Run one inference-only Warlock demonstration with the production HAPPO checkpoint."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.afsim_env import AFSIMIslandEnv
from envs.rl_interface import AFSIMRLInterface
from train.checkpointing import load_training_checkpoint
from train.decision_timing import (
    apply_bottom_decision_timing,
    resolve_bottom_decision_timing,
)
from train.happo_trainer import HAPPOConfig, HAPPOTrainer
from train.recon_attack_stage import (
    high_quality_landing_status,
    recon_attack_terminal_bonus,
)
from train.rule_driven_rollout_collector import RuleDrivenRolloutCollector
from train.train_recon_attack_parallel_eval import alive_unit_summary, stalled_stream_reason


DEFAULT_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "happo_reward_fixed_production"
    / "bottom_happo_recon_attack_parallel_eval_update_000004.pt"
)
DEFAULT_CONFIG = Path(__file__).resolve().parent / "afsim_demo_units.json"
DEFAULT_RESULT = Path(__file__).resolve().parent / "last_demo_result.json"


class InferencePolicy:
    """Expose deterministic or sampled inference without any optimizer updates."""

    def __init__(self, trainer: HAPPOTrainer, deterministic: bool):
        self.trainer = trainer
        self.deterministic = bool(deterministic)

    def act_bottom(self, agent_type, entity_name, agent_state, global_state=None):
        trainer = self.trainer
        policy_id = trainer._policy_id(str(agent_type), str(entity_name))
        policy = trainer.entity_policies[policy_id]
        policy.actor.eval()
        policy.critic.eval()
        obs = np.asarray(agent_state["obs"], dtype=np.float32).reshape(1, -1)
        continuous = trainer.entity_action_spaces[policy_id].__class__.__name__ == "Box"
        available = None if continuous else np.asarray(
            agent_state["action_mask"], dtype=np.float32
        ).reshape(1, -1)
        cent_obs = trainer._central_obs(obs[0], global_state).reshape(1, -1)
        masks = np.ones((1, 1), dtype=np.float32)
        with torch.no_grad():
            value, action, logprob, _, _ = policy.get_actions(
                cent_obs,
                obs,
                trainer._rnn_state(),
                trainer._rnn_state(),
                masks,
                available,
                deterministic=self.deterministic,
            )
        values = action.detach().cpu().numpy().reshape(-1)
        selected = values.astype(np.float32) if continuous else int(values[0])
        return {
            "action": selected,
            "logprob": float(logprob.detach().cpu().numpy().reshape(-1)[0]),
            "value": float(value.detach().cpu().numpy().reshape(-1)[0]),
        }

    def value_bottom(self, *args, **kwargs):
        return self.trainer.value_bottom(*args, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--port", type=int, default=50050)
    parser.add_argument(
        "--external-warlock",
        action="store_true",
        help="do not launch Warlock; connect to a scenario already opened by the user",
    )
    parser.add_argument("--platform-timeout", type=float, default=120.0)
    parser.add_argument("--platform-state-stall-seconds", type=float, default=30.0)
    parser.add_argument("--warlock-start-delay", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--bottom-decisions-per-hour", type=float, default=50.0)
    parser.add_argument("--simulation-clock-rate", type=float, default=60.0)
    parser.add_argument("--decision-seconds", type=float, default=0.0)
    parser.add_argument("--native-decision-pause", action="store_true")
    parser.add_argument("--native-decision-pause-timeout", type=float, default=45.0)
    parser.add_argument("--adaptive-decision-timing", action="store_true")
    parser.add_argument("--sample-actions", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--min-recon-alive", type=int, default=3)
    parser.add_argument("--min-attack-alive", type=int, default=3)
    parser.add_argument("--min-loaded-transports", type=int, default=1)
    parser.add_argument("--deadline-miss-penalty", type=float, default=-50.0)
    parser.add_argument("--enable-negative-rewards", action="store_true")
    parser.add_argument("--result-file", type=Path, default=DEFAULT_RESULT)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate checkpoint/model/environment compatibility without starting Warlock",
    )
    return parser.parse_args()


def checkpoint_config(packet):
    saved = dict(packet.get("config", {}))
    allowed = {item.name for item in fields(HAPPOConfig)}
    return HAPPOConfig(**{key: value for key, value in saved.items() if key in allowed})


def infer_hidden_sizes(trainer_state):
    policies = trainer_state.get("policies", {})
    if not policies:
        raise ValueError("checkpoint trainer has no policies")
    actor = next(iter(policies.values()))["actor"]
    key = next(
        (name for name in actor if name.endswith("base.mlp.fc1.0.weight")),
        None,
    )
    if key is None:
        raise ValueError("cannot infer hidden size from checkpoint actor")
    hidden = int(actor[key].shape[0])
    layer_numbers = {
        int(name.split("fc2.")[1].split(".")[0])
        for name in actor
        if "base.mlp.fc2." in name and name.endswith(".0.weight")
    }
    return (hidden,) * (1 + len(layer_numbers))


def build_trainer(api, state, device):
    trainer_state = state.get("trainer", state)
    agent_types = tuple(trainer_state.get("agent_types", ("recon", "attack")))
    specs = api.get_bottom_agent_specs()
    missing = [name for name in agent_types if name not in specs]
    if missing:
        raise ValueError("environment is missing checkpoint agent types: {0}".format(missing))
    expected_global = int(trainer_state["global_state_dim"])
    actual_global = int(specs["global"]["obs_dim"])
    if actual_global != expected_global:
        raise ValueError(
            "critic global-state dimension mismatch: checkpoint={0}, environment={1}".format(
                expected_global, actual_global
            )
        )
    expected_entities = {
        name: tuple(values)
        for name, values in trainer_state.get("entity_names", {}).items()
    }
    trainer_specs = {name: dict(specs[name]) for name in agent_types}
    for name in agent_types:
        actual = tuple(specs[name].get("entity_names", ()))
        expected = expected_entities.get(name, actual)
        if len(expected) == 1 and len(actual) > 1:
            # Shared-Actor checkpoints use one synthetic policy id while the
            # live environment still exposes every aircraft name.
            trainer_specs[name]["entity_names"] = expected
            trainer_specs[name]["num_entities"] = 1
        elif expected != actual:
            raise ValueError(
                "{0} entity order mismatch: checkpoint={1}, environment={2}".format(
                    name, expected, actual
                )
            )
    trainer = HAPPOTrainer(
        trainer_specs,
        global_state_dim=actual_global,
        agent_types=agent_types,
        hidden_sizes=infer_hidden_sizes(trainer_state),
        config=checkpoint_config(trainer_state),
        device=device,
        trainable_agent_types=(),
    )
    trainer.load_state_dict(trainer_state)
    for policy in trainer.entity_policies.values():
        policy.actor.eval()
        policy.critic.eval()
    return trainer


def choose_device(requested):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def save_result(path, result):
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def compact_actions(rollout):
    result = {}
    for agent_type, stream in rollout.bottom.items():
        if agent_type not in ("recon", "attack"):
            continue
        result[agent_type] = {
            row["entity_name"]: (
                [round(float(value), 3) for value in np.asarray(row["action"]).reshape(-1)]
                if agent_type == "recon"
                else str(row["action_name"])
            )
            for row in stream.rows
        }
    return result


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError("checkpoint not found: {0}".format(checkpoint))

    state = load_training_checkpoint(checkpoint)
    device = choose_device(args.device)
    env = AFSIMIslandEnv(
        config_path=str(args.config_path.resolve()),
        bind=not args.check_only,
        auto_start_warlock=False,
        local_address=("0.0.0.0", args.port),
    )
    try:
        env.set_negative_rewards_enabled(args.enable_negative_rewards)
        timing = resolve_bottom_decision_timing(
            args.bottom_decisions_per_hour,
            args.simulation_clock_rate,
            args.decision_seconds,
        )
        apply_bottom_decision_timing(env, timing)
        env.native_decision_pause_control = bool(args.native_decision_pause)
        env.native_decision_pause_timeout = float(args.native_decision_pause_timeout)
        api = AFSIMRLInterface(env, reward_profile="recon_attack_stage")
        trainer = build_trainer(api, state, device)
        policy = InferencePolicy(trainer, deterministic=not args.sample_actions)
        print(
            "CHECKPOINT_READY",
            "path", checkpoint,
            "update", state.get("update", "?"),
            "device", device,
            "policies", len(trainer.entity_policies),
            "deterministic", not args.sample_actions,
            flush=True,
        )
        if args.check_only:
            return 0

        if not args.external_warlock:
            process = env.start_warlock()
            print("WARLOCK_STARTED", "pid", process.pid, flush=True)
        else:
            print("WARLOCK_EXTERNAL", "waiting on UDP port", args.port, flush=True)
        time.sleep(max(0.0, args.warlock_start_delay))
        ready = env.wait_for_platforms(list(env.platforms), timeout=args.platform_timeout)
        registered = sum(p.platform_id is not None for p in env.platforms.values())
        print(
            "PLATFORMS_READY", ready,
            "registered", registered,
            "expected", len(env.platforms),
            flush=True,
        )
        if not ready:
            raise RuntimeError("Warlock platforms did not register before timeout")

        api.reset_rule_driven()
        collector = RuleDrivenRolloutCollector(
            api,
            bottom_agent_types=("recon", "attack"),
            adaptive_decision_timing=args.adaptive_decision_timing,
        )
        started_at = time.monotonic()
        total_reward = 0.0
        reason = "stage_time_limit"
        status = high_quality_landing_status(
            env, args.min_recon_alive, args.min_attack_alive, args.min_loaded_transports
        )
        completed_steps = 0
        for step in range(1, args.max_steps + 1):
            rollout = collector.collect(policy, n_steps=1, reset=False)
            summary = rollout.summary()
            completed_steps += int(summary.get("steps", 0))
            total_reward += float(summary.get("team_reward_sum", 0.0))
            status = high_quality_landing_status(
                env, args.min_recon_alive, args.min_attack_alive,
                args.min_loaded_transports,
            )
            current_reason = ""
            if env.platform_state_age_seconds() > args.platform_state_stall_seconds:
                current_reason = stalled_stream_reason(status)
            elif status["landing_combat_conditions_met"]:
                current_reason = "combat_landing_ready"
            elif status["landing_time_override_met"]:
                current_reason = "landing_deadline_missed"
            elif summary.get("terminal", False):
                current_reason = str(summary.get("done_reason", "environment_terminal"))
            elif step >= args.max_steps:
                current_reason = "stage_time_limit"
            if step == 1 or step % max(1, args.log_every) == 0 or current_reason:
                print(
                    "DEMO_STEP",
                    json.dumps(
                        {
                            "step": step,
                            "sim_seconds": round(float(env._current_sim_time()), 3),
                            "team_reward": round(float(summary.get("team_reward_sum", 0.0)), 4),
                            "known_targets": len(env.detected_targets),
                            "actions": compact_actions(rollout),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if current_reason:
                reason = current_reason
                break

        terminal_reward = recon_attack_terminal_bonus(
            status,
            reason,
            env._current_sim_time(),
            deadline_miss_penalty=args.deadline_miss_penalty,
            negative_rewards_enabled=env.negative_rewards_enabled,
        )
        result = {
            "record_type": "warlock_happo_demonstration",
            "checkpoint": str(checkpoint),
            "checkpoint_update": int(state.get("update", 0)),
            "deterministic": not args.sample_actions,
            "steps": completed_steps,
            "done_reason": reason,
            "success": reason == "combat_landing_ready",
            "team_reward": float(total_reward),
            "terminal_reward": float(terminal_reward),
            "total_reward": float(total_reward + terminal_reward),
            "simulation_seconds": float(env._current_sim_time()),
            "wall_seconds": max(0.0, time.monotonic() - started_at),
            "landing_status": status,
            "surviving_units": alive_unit_summary(env),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        output = save_result(args.result_file, result)
        print("DEMO_RESULT", json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        print("RESULT_FILE", output, flush=True)
        return 0 if reason != "udp_state_stream_stalled" else 2
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
