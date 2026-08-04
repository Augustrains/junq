"""Parallel multi-env rollout collection with thread-safe buffer."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Tuple

import numpy as np

from envs.afsim_env import AFSIMIslandEnv
from envs.rl_interface import AFSIMRLInterface
from train.rule_driven_rollout_collector import BottomOnlyRollout, RuleDrivenRolloutCollector

BOTTOM_AGENT_TYPES = ("recon", "attack", "landing", "ground")


def _make_env(
    base_env: AFSIMIslandEnv,
    port: int,
    scenario_file: str,
    auto_start_warlock: bool = True,
) -> AFSIMIslandEnv:
    """Create an env clone bound to a different UDP port."""
    env = AFSIMIslandEnv(
        config_path=str(getattr(base_env, "config_path", "") or ""),
        bind=True,
        auto_start_warlock=False,
        local_address=("0.0.0.0", port),
    )
    env.config.setdefault("scenario", {})["scenario_file"] = scenario_file
    env.config["scenario"]["warlock_log_path"] = str(
        env.config.get("scenario", {}).get("warlock_log_path", "warlock_runtime.log")
    ).replace(".log", "_p{0}.log".format(port))
    env.decision_seconds = float(base_env.decision_seconds)
    if auto_start_warlock:
        env.start_warlock()
    return env


class ParallelCollector:
    """Run N RuleDrivenRolloutCollectors in threads against N Warlock instances.

    All collectors share the same policy_set (HAPPOTrainer) and feed into the
    same thread-safe buffer.
    """

    def __init__(
        self,
        api: AFSIMRLInterface,
        n_envs: int = 2,
        base_port: int = 50050,
        scenario_dir: str = "",
        auto_start_warlock: bool = True,
        **collector_kwargs,
    ):
        self.n_envs = max(1, int(n_envs))
        self.base_port = int(base_port)
        self.auto_start_warlock = bool(auto_start_warlock)
        self.collector_kwargs = dict(collector_kwargs)

        scenario_dir = scenario_dir or str(
            api.env.config.get("scenario", {}).get("scenario_dir", "")
        )
        base_scenario = str(
            api.env.config.get("scenario", {}).get(
                "scenario_file", "scenarios/island_assault_min.txt"
            )
        )
        self.scenario_files: List[str] = []
        for i in range(self.n_envs):
            port = self.base_port + i
            if i == 0:
                self.scenario_files.append(base_scenario)
            else:
                p = str(base_scenario).replace(".txt", "_p{0}.txt".format(i))
                self.scenario_files.append(p)

        self.primary_api = api
        self.primary_env = api.env
        self.apis: List[AFSIMRLInterface] = []
        self.collectors: List[RuleDrivenRolloutCollector] = []

        for i in range(self.n_envs):
            port = self.base_port + i
            scenario_file = self.scenario_files[i]
            if i == 0:
                env: AFSIMIslandEnv = self.primary_env
                api_instance: AFSIMRLInterface = self.primary_api
            else:
                env = _make_env(
                    self.primary_env, port, scenario_file,
                    auto_start_warlock=self.auto_start_warlock,
                )
                api_instance = AFSIMRLInterface(env)

            collector = RuleDrivenRolloutCollector(api_instance, **self.collector_kwargs)
            self.apis.append(api_instance)
            self.collectors.append(collector)

        self._lock = threading.Lock()
        self._collected_rollouts: List[BottomOnlyRollout] = []
        self._collected_team_rewards: List[float] = []
        self._collected_dones: List[bool] = []
        self._done = False
        self._first_step = True
        self._step_id = 0

    def close(self) -> None:
        for api in self.apis:
            if api is not self.primary_api:
                try:
                    api.env.close()
                except Exception:
                    pass

    def _collect_one(
        self, index: int, policy_set, n_steps: int, reset: bool
    ) -> Tuple[int, Optional[BottomOnlyRollout], float, bool]:
        """Run one collector step in a worker thread."""
        collector = self.collectors[index]
        try:
            rollout = collector.collect(policy_set, n_steps=n_steps, reset=reset)
            summary = rollout.summary()
            team_reward = float(summary.get("team_reward_sum", 0.0))
            done = bool(summary.get("terminal", False))
            return index, rollout, team_reward, done
        except Exception:
            return index, None, 0.0, False

    def collect_all(
        self, policy_set, n_steps: int = 1
    ) -> List[BottomOnlyRollout]:
        """Collect one step from all envs in parallel. Thread-safe."""
        rollouts: List[Optional[BottomOnlyRollout]] = [None] * self.n_envs
        team_rewards: List[float] = [0.0] * self.n_envs
        dones: List[bool] = [False] * self.n_envs

        reset = self._first_step or self._done
        self._first_step = False

        with ThreadPoolExecutor(max_workers=self.n_envs) as executor:
            futures = {
                executor.submit(
                    self._collect_one, i, policy_set, n_steps, reset
                ): i
                for i in range(self.n_envs)
            }
            for future in as_completed(futures):
                i, rollout, team_reward, done = future.result()
                rollouts[i] = rollout
                team_rewards[i] = team_reward
                dones[i] = done

        results: List[BottomOnlyRollout] = [
            r for r in rollouts if r is not None
        ]
        if not results:
            raise RuntimeError("all parallel collectors failed")

        with self._lock:
            self._collected_rollouts.extend(results)
            self._collected_team_rewards.extend(team_rewards)
            self._collected_dones.extend(dones)
            self._done = any(dones)
            self._step_id += 1

        return results

    def drain_collected(self) -> Tuple[List[BottomOnlyRollout], bool]:
        """Atomically drain all collected rollouts and reset."""
        with self._lock:
            rollouts = list(self._collected_rollouts)
            done = bool(self._done)
            self._collected_rollouts.clear()
            self._collected_dones.clear()
            if done:
                self._done = False
                self._first_step = True
            return rollouts, done

    def aggregate_summary(self) -> dict:
        """Aggregate team rewards across all envs."""
        with self._lock:
            rewards = list(self._collected_team_rewards)
        return {
            "steps": self._step_id,
            "team_reward_sum": float(np.sum(rewards)) if rewards else 0.0,
            "team_reward_mean": float(np.mean(rewards)) if rewards else 0.0,
            "n_envs": self.n_envs,
        }
