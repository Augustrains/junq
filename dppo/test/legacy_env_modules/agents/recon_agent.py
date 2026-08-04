import random
from typing import Dict, Optional


class ReconAgent(object):
    """Small wrapper for the three-dimensional continuous recon actuator."""

    def __init__(self, env, rng: Optional[random.Random] = None, policy: str = "random"):
        self.env = env
        self.rng = rng or random.Random()
        self.policy = policy

    def step(self, group_id: str) -> Dict[str, object]:
        state = self.env.get_recon_task_state(group_id)
        if not state:
            return {
                "group_id": group_id,
                "success": False,
                "error": "recon_group_not_found",
                "actions": {},
            }

        decisions = {}
        success = True
        for aircraft_name, aircraft_state in state.get("aircraft", {}).items():
            action = self.select_action(aircraft_state)
            ok = self.env.apply_recon_aircraft_continuous_action(
                group_id, aircraft_name, action
            )
            decisions[aircraft_name] = {
                "action": list(action),
                "action_name": "CONTINUOUS_MOVE",
                "sent": ok,
            }
            success = success and ok

        return {
            "group_id": group_id,
            "success": success,
            "actions": decisions,
            "state": state,
        }

    def select_action(self, aircraft_state: Dict[str, object]):
        if not any(float(value) > 0.0 for value in aircraft_state.get("action_mask", [])):
            return [0.0, 0.0, 0.0]
        if self.policy != "random":
            return [0.0, 1.0, 0.0]
        return [self.rng.uniform(-1.0, 1.0) for _ in range(3)]
