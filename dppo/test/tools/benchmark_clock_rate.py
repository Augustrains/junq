"""Validate whether a live AFSIM/Warlock scenario sustains a requested clock rate.

The generated scenario explicitly enables realtime mode and sets ``clock_rate``.
The udpnet plugin's ``WallTime`` field is the observer callback ``simtime``,
so this script measures global AFSIM simulation time without sending actions.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from envs.afsim_env import AFSIMIslandEnv
from train.decision_timing import apply_bottom_decision_timing, resolve_bottom_decision_timing


def percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction)))
    return float(ordered[index])


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark a live AFSIM clock rate using actual simulation timestamps.")
    parser.add_argument("--simulation-clock-rate", type=float, required=True, help="Must equal the clock_rate in the running AFSIM scenario.")
    parser.add_argument("--bottom-decisions-per-hour", type=float, default=50.0)
    parser.add_argument("--steps", type=int, default=50, help="Decision windows to observe; 50 tests one simulation hour.")
    parser.add_argument("--config-path", default="")
    parser.add_argument("--local-address", default="")
    parser.add_argument("--auto-start-warlock", action="store_true", help="Generate a clock-rate-specific scenario copy and start Warlock with it.")
    parser.add_argument("--headless-mission", action="store_true", help="Use mission.exe instead of the Warlock GUI for the generated scenario.")
    parser.add_argument("--scenario-file", default="", help="Source scenario file. Defaults to scenario_file in the environment config.")
    parser.add_argument("--keep-generated-scenario", action="store_true", help="Keep the generated scenario after the benchmark exits.")
    parser.add_argument("--worker-id", type=int, default=0, help="Unique parallel-worker identifier.")
    parser.add_argument("--udp-target-address", default="127.0.0.1", help="Address the generated scenario sends UDP state to.")
    parser.add_argument("--udp-state-update-interval", type=float, default=0.0, help="Throttle per-platform MoveUpdate output in simulation seconds; 0 keeps every mover callback.")
    parser.add_argument("--platform-timeout", type=float, default=60.0)
    parser.add_argument("--min-effective-decisions", type=float, default=47.5)
    parser.add_argument("--max-effective-decisions", type=float, default=52.5)
    parser.add_argument("--max-p95-step-sim-seconds", type=float, default=80.0)
    parser.add_argument("--report-file", default="")
    return parser.parse_args()


def parse_address(value):
    if not value:
        return None
    host, port = value.rsplit(":", 1)
    return host, int(port)


def prepare_clock_rate_scenario(env, requested_file, clock_rate, udp_port, udp_address, worker_id, state_update_interval=0.0):
    """Create a sibling scenario copy with exactly one effective clock_rate."""
    scenario_cfg = env.config.get("scenario", {})
    scenario_dir = Path(str(scenario_cfg.get("scenario_dir", "")))
    configured_file = str(scenario_cfg.get("scenario_file", ""))
    source = Path(requested_file) if requested_file else scenario_dir / configured_file
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError("scenario file not found: {0}".format(source))

    text = source.read_text(encoding="utf-8")
    # One unique UDP port isolates one Python worker and one Warlock process.
    # Do not emit optional plugin commands: deployed grammar only needs port/address.
    udp_lines = ["udpnet", "   port {0}".format(int(udp_port)), "   address {0}".format(str(udp_address))]
    if float(state_update_interval) > 0.0:
        udp_lines.append("   state_update_interval {0:.12g} sec".format(float(state_update_interval)))
    udp_lines.append("end_udpnet")
    udp_block = "\\n".join(udp_lines)
    text, udp_blocks = re.subn(r"(?ms)^udpnet\s*\n.*?^end_udpnet", udp_block, text, count=1)
    if udp_blocks != 1:
        raise ValueError("source scenario must contain one active udpnet block")
    clock_line = "clock_rate {0:.12g}".format(float(clock_rate))
    rewritten, count = re.subn(r"(?m)^(\s*clock_rate\s+)\S+.*$", clock_line, text)
    if count == 0:
        rewritten = text.rstrip() + "\n" + clock_line + "\n"
    elif count > 1:
        seen, kept = False, []
        for line in rewritten.splitlines():
            if re.match(r"^\s*clock_rate\s+", line):
                if seen:
                    continue
                seen = True
            kept.append(line)
        rewritten = "\n".join(kept) + "\n"

    rate_label = ("{0:.12g}".format(float(clock_rate))).replace(".", "_")
    generated = source.with_name("{0}.worker_{1}.clockrate_{2}{3}".format(source.stem, int(worker_id), rate_label, source.suffix))
    # Preserve the source timing mode; append only the requested clock_rate.
    rewritten = rewritten.rstrip() + "\n\n# Generated benchmark timing override.\n" + clock_line + "\n"
    generated.write_text(rewritten, encoding="utf-8")
    # Preserve the original working directory so relative includes and assets
    # resolve exactly as they did for the source scenario.
    try:
        scenario_cfg["scenario_file"] = str(generated.relative_to(scenario_dir))
    except ValueError:
        scenario_cfg["scenario_dir"] = str(generated.parent)
        scenario_cfg["scenario_file"] = generated.name
    return generated

def main():
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    timing = resolve_bottom_decision_timing(
        args.bottom_decisions_per_hour, args.simulation_clock_rate
    )
    env = AFSIMIslandEnv(
        config_path=args.config_path or None,
        bind=True,
        auto_start_warlock=False,
        local_address=parse_address(args.local_address),
    )
    apply_bottom_decision_timing(env, timing)
    generated_scenario = None
    if args.auto_start_warlock:
        if args.headless_mission:
            scenario_cfg = env.config.get("scenario", {})
            configured_runner = Path(str(scenario_cfg.get("warlock_path", "")))
            scenario_cfg["warlock_path"] = str(configured_runner.with_name("mission.exe"))
            scenario_cfg["warlock_args"] = []
        generated_scenario = prepare_clock_rate_scenario(
            env, args.scenario_file, timing["simulation_clock_rate"],
            env.local_address[1], args.udp_target_address, args.worker_id,
            args.udp_state_update_interval,
        )
        print("generated_scenario=" + str(generated_scenario), flush=True)
        env.start_warlock()
    try:
        required = ["red_recon_1", "red_attack_1", "red_transport_1"]
        ready = env.wait_for_platforms(required, timeout=args.platform_timeout)
        if not ready:
            raise RuntimeError("required live platforms did not register: {0}".format(required))

        message_counts = Counter()
        original_handle_message = env._handle_message
        def counted_handle_message(message):
            message_counts[str(message.get("MsgType", ""))] += 1
            return original_handle_message(message)
        env._handle_message = counted_handle_message

        # Let the first live state report establish the simulation-time origin.
        env._drain_messages(timeout=max(0.5, env.decision_seconds))
        start_sim = env._current_sim_time()
        start_wall = time.monotonic()
        previous_sim = start_sim
        deltas_sim, deltas_wall, platform_progress = [], [], []
        for step in range(1, args.steps + 1):
            previous_updates = {name: float(p.last_update) for name, p in env.platforms.items()}
            wall_before = time.monotonic()
            env._drain_messages(timeout=env.decision_seconds)
            wall_after = time.monotonic()
            current_sim = env._current_sim_time()
            sim_delta = current_sim - previous_sim
            changed = sum(
                1 for name, p in env.platforms.items()
                if float(p.last_update) > previous_updates.get(name, 0.0)
            )
            deltas_sim.append(sim_delta)
            deltas_wall.append(wall_after - wall_before)
            platform_progress.append(changed)
            previous_sim = current_sim
            print(
                "step={0} sim_delta={1:.3f}s wall_delta={2:.4f}s updated_platforms={3}".format(
                    step, sim_delta, wall_after - wall_before, changed
                ),
                flush=True,
            )

        sim_elapsed = previous_sim - start_sim
        wall_elapsed = time.monotonic() - start_wall
        target_sim_elapsed = float(args.steps) * timing["simulation_interval_seconds"]
        effective_decisions = float(args.steps) * 3600.0 / max(sim_elapsed, 1e-9)
        effective_rate = sim_elapsed / max(wall_elapsed, 1e-9)
        report = {
            "requested_clock_rate": timing["simulation_clock_rate"],
            "requested_decisions_per_sim_hour": timing["decisions_per_sim_hour"],
            "requested_step_sim_seconds": timing["simulation_interval_seconds"],
            "requested_step_wall_seconds": timing["wall_interval_seconds"],
            "steps": args.steps,
            "sim_elapsed_seconds": sim_elapsed,
            "target_sim_elapsed_seconds": target_sim_elapsed,
            "wall_elapsed_seconds": wall_elapsed,
            "effective_clock_rate": effective_rate,
            "effective_decisions_per_sim_hour": effective_decisions,
            "mean_step_sim_seconds": statistics.mean(deltas_sim),
            "p95_step_sim_seconds": percentile(deltas_sim, 0.95),
            "max_step_sim_seconds": max(deltas_sim),
            "mean_step_wall_seconds": statistics.mean(deltas_wall),
            "steps_with_sim_progress": sum(1 for value in deltas_sim if value > 0.0),
            "steps_with_platform_updates": sum(1 for value in platform_progress if value > 0),
            "udp_state_update_interval": float(args.udp_state_update_interval),
            "udp_messages_total": int(sum(message_counts.values())),
            "udp_messages_per_wall_second": float(sum(message_counts.values())) / max(wall_elapsed, 1e-9),
            "udp_message_types": dict(message_counts),
        }
        report["passed"] = (
            args.min_effective_decisions <= effective_decisions <= args.max_effective_decisions
            and report["p95_step_sim_seconds"] <= args.max_p95_step_sim_seconds
            and report["steps_with_sim_progress"] == args.steps
            and report["steps_with_platform_updates"] == args.steps
        )
        print("CLOCK_RATE_BENCHMARK=" + json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
        if args.report_file:
            report_path = Path(args.report_file)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0 if report["passed"] else 2
    finally:
        env.close()
        if generated_scenario is not None and not args.keep_generated_scenario:
            try:
                generated_scenario.unlink()
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())