import random
from typing import Dict, Optional


class GroundAgent(object):
    """Ground sub-agent wrapper for landed red ground forces."""

    def __init__(self, env, rng: Optional[random.Random] = None, policy: str = "random"):
        self.env = env
        self.rng = rng or random.Random()
        self.policy = policy

    def step(self, group_id: str) -> Dict[str, object]:
        state = self.env.get_ground_task_state(group_id)
        if not state:
            return {"group_id": group_id, "success": False, "error": "ground_group_not_found", "actions": {}}
        decisions = {}
        success = True
        for unit_name, unit_state in state.get("units", {}).items():
            action_id = self.select_action(unit_state)
            ok = self.env.apply_ground_unit_action(group_id, unit_name, action_id)
            decisions[unit_name] = {"action_id": action_id, "action_name": self._action_name(state, action_id), "sent": ok}
            success = success and ok
        return {"group_id": group_id, "success": success, "actions": decisions, "state": state}

    def select_action(self, unit_state: Dict[str, object]) -> int:
        mask = list(unit_state.get("action_mask", []))
        available = [index for index, value in enumerate(mask) if float(value) > 0.0]
        if not available:
            return 0
        if self.policy != "random":
            return available[0]
        return self.rng.choice(available)

    @staticmethod
    def _action_name(state: Dict[str, object], action_id: int) -> str:
        for action in state.get("action_table", []):
            if int(action.get("id", -1)) == int(action_id):
                return action.get("name", "")
        return "UNKNOWN"
