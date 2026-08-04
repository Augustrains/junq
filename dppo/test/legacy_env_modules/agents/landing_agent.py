import random
from typing import Dict, Optional


class LandingAgent(object):
    """Landing sub-agent wrapper.

    The high-level commander selects a landing zone. This agent reads per-ship
    landing state, selects one atomic action for each assigned transport, and
    sends the actions through the env.
    """

    def __init__(self, env, rng: Optional[random.Random] = None, policy: str = "random"):
        self.env = env
        self.rng = rng or random.Random()
        self.policy = policy

    def step(self, group_id: str) -> Dict[str, object]:
        state = self.env.get_landing_task_state(group_id)
        if not state:
            return {
                "group_id": group_id,
                "success": False,
                "error": "landing_group_not_found",
                "actions": {},
            }

        decisions = {}
        success = True
        for ship_name, ship_state in state.get("ships", {}).items():
            action = self.select_action(ship_state)
            ok = self.env.apply_landing_ship_continuous_action(group_id, ship_name, action)
            action_name = "CONTINUOUS_MOVE"
            decisions[ship_name] = {
                "action": list(action),
                "action_name": action_name,
                "sent": ok,
            }
            success = success and ok

        return {
            "group_id": group_id,
            "success": success,
            "actions": decisions,
            "state": state,
        }

    def select_action(self, ship_state: Dict[str, object]):
        if not any(float(value) > 0.0 for value in ship_state.get("action_mask", [])):
            return [0.0, 0.0, 0.0]
        if self.policy != "random":
            return [0.0, 1.0, 0.0]
        return [self.rng.uniform(-1.0, 1.0) for _ in range(3)]

    @staticmethod
    def _action_name(state: Dict[str, object], action_id: int) -> str:
        for action in state.get("action_table", []):
            if int(action.get("id", -1)) == int(action_id):
                return action.get("name", "")
        return "UNKNOWN"
