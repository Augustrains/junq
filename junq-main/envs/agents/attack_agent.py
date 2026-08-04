import random
from typing import Dict, Optional


class AttackAgent(object):
    """Attack sub-agent wrapper.

    The high-level commander selects a mission area and a 2/3-aircraft package. This
    agent reads per-aircraft attack state, selects a tactical action for each
    assigned aircraft, and sends those actions through the env.
    """

    def __init__(self, env, rng: Optional[random.Random] = None, policy: str = "random"):
        self.env = env
        self.rng = rng or random.Random()
        self.policy = policy

    def step(self, group_id: str) -> Dict[str, object]:
        state = self.env.get_attack_task_state(group_id)
        if not state:
            return {
                "group_id": group_id,
                "success": False,
                "error": "attack_group_not_found",
                "actions": {},
            }

        decisions = {}
        success = True
        for aircraft_name, aircraft_state in state.get("aircraft", {}).items():
            action_id = self.select_action(aircraft_state)
            ok = self.env.apply_attack_aircraft_action(group_id, aircraft_name, action_id)
            action_name = self._action_name(state, action_id)
            decisions[aircraft_name] = {
                "action_id": action_id,
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

    def select_action(self, aircraft_state: Dict[str, object]) -> int:
        mask = list(aircraft_state.get("action_mask", []))
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
