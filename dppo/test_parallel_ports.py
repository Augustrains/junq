"""Verify two Warlock instances can run in parallel on different UDP ports.

Usage:
    python test_parallel_ports.py [--base-port 50050] [--n-envs 2]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from envs.afsim_env import AFSIMIslandEnv
from envs.rl_interface import AFSIMRLInterface
from train.happo_trainer import HAPPOConfig, HAPPOTrainer
from train.parallel_rollout_collector import _make_env
from train.rule_driven_rollout_collector import RuleDrivenRolloutCollector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-port", type=int, default=50050)
    parser.add_argument("--test-ports", type=int, nargs="+", default=[50051, 50052],
                        help="Ports to test (skips primary port).")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--platform-timeout", type=float, default=60.0)
    parser.add_argument("--config-path", default="")
    args = parser.parse_args()

    config_path = args.config_path or str(ROOT / "envs" / "afsim_units.json")

    # Create a fake primary env just for config reference (no Warlock, no bind).
    base_env = AFSIMIslandEnv(
        config_path=config_path, bind=False, auto_start_warlock=False,
    )
    try:
        # Only test the ADDITIONAL ports, not the currently-used one.
        test_ports = list(args.test_ports)
        print("PORT_TEST testing_ports={0} (primary_port_not_touched)".format(test_ports))

        envs = []
        for port in test_ports:
            scenario_file = str(
                base_env.config.get("scenario", {}).get("scenario_file", "")
            ).replace(".txt", "_p{0}.txt".format(
                port - args.base_port if port > args.base_port else "_" + str(port)
            ))
            env = _make_env(
                base_env, port, scenario_file, auto_start_warlock=True,
            )
            envs.append(env)
            print("PORT_TEST env_created port={0} sock={1} scenario={2}".format(
                port, bool(env.sock), scenario_file))

        # Wait for platforms.
        all_ready = True
        for env in envs:
            ready = env.wait_for_platforms(
                ["red_attack_1"], timeout=args.platform_timeout
            )
            print("PORT_TEST port={0} platforms_ready={1}".format(
                env.local_address[1], ready))
            if not ready:
                all_ready = False

        if not all_ready:
            print("PORT_TEST FAIL not_all_envs_ready")
            return 2

        # Create APIs and collectors for each env.
        apis = [AFSIMRLInterface(env) for env in envs]
        collectors = [RuleDrivenRolloutCollector(api) for api in apis]

        # Create a dummy trainer.
        specs = apis[0].get_bottom_agent_specs()
        bottom_specs = {n: specs[n] for n in ("recon", "attack", "landing", "ground")}
        trainer = HAPPOTrainer(
            bottom_specs,
            global_state_dim=int(specs["global"]["obs_dim"]),
            hidden_sizes=(32, 32),
            config=HAPPOConfig(),
        )

        print("PORT_TEST trainer_created running_steps={0}".format(args.steps))

        for step in range(args.steps):
            t0 = time.monotonic()
            rollouts = []
            with ThreadPoolExecutor(max_workers=len(collectors)) as executor:
                futures = [
                    executor.submit(
                        c.collect, trainer, n_steps=1, reset=(step == 0)
                    )
                    for c in collectors
                ]
                for future in as_completed(futures):
                    rollouts.append(future.result())
            elapsed = time.monotonic() - t0
            print(json.dumps({
                "step": step,
                "n_rollouts": len(rollouts),
                "elapsed_s": round(elapsed, 3),
            }, ensure_ascii=False))
            for i, r in enumerate(rollouts):
                rsum = r.summary()
                print("  env[{0}] port={1} reward_sum={2} steps={3}".format(
                    i, envs[i].local_address[1],
                    round(rsum.get("team_reward_sum", 0.0), 4),
                    rsum.get("steps", 0)))

        print("PORT_TEST OK parallel_collection_works n_ports={0}".format(len(test_ports)))
    finally:
        for env in envs:
            try:
                env.close()
            except Exception:
                pass
        base_env.close()


if __name__ == "__main__":
    raise SystemExit(main())
