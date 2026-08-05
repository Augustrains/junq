"""Run the production demo with scripted recon and network attack aircraft."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import demo_happo_warlock as _base_demo


SAFE_DISTANCE_M = 60_000.0
MAX_INGRESS_SECONDS = 1_800.0
COOLDOWN_SECONDS = 120.0


class ManualStopAFSIMIslandEnv(_base_demo.AFSIMIslandEnv):
    """Keep this demonstration alive until the operator stops it."""

    def is_done(self):
        return False


class ScriptedReconPolicy:
    """Delegate attack inference to HAPPO and script only recon movement."""

    def __init__(self, network_policy, env):
        self.network_policy = network_policy
        self.env = env
        self.team_state = {}
        self.last_blue_aam = self._blue_aam_snapshot()
        self.last_ammo_check_time = float("-inf")

    def _red_home(self):
        configured = self.env.config.get("red", {}).get("carrier", [])
        for name in configured:
            platform = self.env.platforms.get(name)
            if platform is not None:
                return platform
        for platform in self.env.platforms.values():
            if platform.side == "red" and platform.role == "carrier":
                return platform
        raise RuntimeError("red carrier was not found in the demo scene")

    def _armed_blue_aircraft(self):
        return [
            platform
            for platform in self.env.platforms.values()
            if platform.side == "blue"
            and platform.role == "attack_aircraft"
            and platform.alive
            and platform.platform_id is not None
            and str(platform.task_status).upper() in ("BLUE_CAP_PATROL", "BLUE_INTERCEPT")
            and int(self.env.attack_ammo.get(platform.name, {}).get("fox3", 0)) > 0
        ]

    def _blue_aam_snapshot(self):
        # Include destroyed aircraft so a kill is not mistaken for ammunition
        # expenditure. A real launch is the observed 1 -> 0 transition.
        return {
            platform.name: max(
                0, int(self.env.attack_ammo.get(platform.name, {}).get("fox3", 0))
            )
            for platform in self.env.platforms.values()
            if platform.side == "blue"
            and platform.role == "attack_aircraft"
        }

    def _distance(self, source, target):
        distance, _bearing = self.env._distance_and_bearing(
            source.lat, source.lon, target.lat, target.lon
        )
        return float(distance)

    def _nearest_armed_blue(self, aircraft):
        candidates = self._armed_blue_aircraft()
        if not candidates:
            return None, float("inf")
        target = min(candidates, key=lambda item: self._distance(aircraft, item))
        return target, self._distance(aircraft, target)

    def _toward(self, aircraft, target):
        north_m, east_m = self.env._relative_north_east(
            aircraft.lat, aircraft.lon, target.lat, target.lon
        )
        scale = max(abs(float(north_m)), abs(float(east_m)), 1.0)
        return np.asarray(
            [float(east_m) / scale, float(north_m) / scale, 0.0],
            dtype=np.float32,
        )

    @staticmethod
    def _is_leader(agent_state):
        return float(agent_state.get("obs_by_name", {}).get("is_leader", 0.0)) >= 0.5

    def _state_for(self, group_id, now):
        return self.team_state.setdefault(
            group_id,
            {"phase": "ADVANCE", "phase_started": now, "cycle": 1},
        )

    def _set_phase(self, group_id, phase, now, reason, aircraft, distance=None):
        state = self._state_for(group_id, now)
        if state["phase"] == phase:
            return
        previous = state["phase"]
        state["phase"] = phase
        state["phase_started"] = now
        if phase == "ADVANCE":
            state["cycle"] += 1
        distance_text = "" if distance is None else " distance_km={0:.3f}".format(distance / 1000.0)
        print(
            "SCRIPTED_RECON_PHASE team={0} leader={1} from={2} to={3} "
            "cycle={4} reason={5} sim_seconds={6:.3f}{7}".format(
                group_id, aircraft.name, previous, phase, state["cycle"],
                reason, now, distance_text,
            ),
            flush=True,
        )

    def _observe_blue_fire(self, now):
        if now == self.last_ammo_check_time:
            return
        current = self._blue_aam_snapshot()
        spent = sum(
            max(0, int(previous) - int(current.get(name, previous)))
            for name, previous in self.last_blue_aam.items()
        )
        if spent > 0:
            print(
                "SCRIPTED_RECON_BLUE_AAM_SPENT count={0} remaining={1} sim_seconds={2:.3f}".format(
                    spent, sum(current.values()), now
                ),
                flush=True,
            )
            for state in self.team_state.values():
                if state["phase"] == "ADVANCE":
                    state["force_retreat"] = True
        self.last_blue_aam = current
        self.last_ammo_check_time = now

    def _scripted_recon_action(self, entity_name, agent_state):
        aircraft = self.env.platforms.get(entity_name)
        if aircraft is None or not aircraft.alive:
            return np.zeros(3, dtype=np.float32)

        now = float(self.env._current_sim_time())
        self._observe_blue_fire(now)
        group_id = str(agent_state.get("group_id") or entity_name)
        state = self._state_for(group_id, now)
        target, target_distance = self._nearest_armed_blue(aircraft)

        # Followers acknowledge the action; the controller keeps formation.
        if self._is_leader(agent_state):
            elapsed = max(0.0, now - float(state["phase_started"]))
            if state.pop("force_retreat", False):
                self._set_phase(group_id, "RETREAT", now, "blue_aam_spent", aircraft, target_distance)
            elif state["phase"] == "ADVANCE" and target is None:
                self._set_phase(group_id, "RETREAT", now, "no_armed_blue_air", aircraft)
            elif state["phase"] == "ADVANCE" and elapsed >= MAX_INGRESS_SECONDS:
                self._set_phase(group_id, "RETREAT", now, "safety_timeout_without_fire", aircraft, target_distance)
            elif state["phase"] == "RETREAT" and (target is None or target_distance >= SAFE_DISTANCE_M):
                self._set_phase(group_id, "COOLDOWN", now, "outside_enemy_attack_range", aircraft, target_distance)
            elif state["phase"] == "COOLDOWN" and elapsed >= COOLDOWN_SECONDS and target is not None:
                self._set_phase(group_id, "ADVANCE", now, "cooldown_complete", aircraft)

        if state["phase"] == "ADVANCE" and target is not None:
            return self._toward(aircraft, target)
        return self._toward(aircraft, self._red_home())

    def act_bottom(self, agent_type, entity_name, agent_state, global_state=None):
        if str(agent_type) != "recon":
            return self.network_policy.act_bottom(
                agent_type, entity_name, agent_state, global_state=global_state
            )
        return {
            "action": self._scripted_recon_action(entity_name, agent_state),
            "logprob": 0.0,
            "value": 0.0,
        }

    def value_bottom(self, agent_type, entity_name, agent_state, global_state=None):
        if str(agent_type) == "recon":
            return 0.0
        return self.network_policy.value_bottom(
            agent_type, entity_name, agent_state, global_state=global_state
        )


class ScriptedReconCollector(_base_demo.RuleDrivenRolloutCollector):
    def __init__(self, api, **kwargs):
        super().__init__(api, **kwargs)
        self._hybrid_policy = None

    def collect(self, policy, n_steps, reset=True, decision_wall_budget=None):
        if self._hybrid_policy is None or reset:
            self._hybrid_policy = ScriptedReconPolicy(policy, self.api.env)
        return super().collect(
            self._hybrid_policy, n_steps, reset=reset,
            decision_wall_budget=decision_wall_budget,
        )


_base_demo.RuleDrivenRolloutCollector = ScriptedReconCollector
_base_demo.AFSIMIslandEnv = ManualStopAFSIMIslandEnv


def main():
    return _base_demo.main()


if __name__ == "__main__":
    raise SystemExit(main())
