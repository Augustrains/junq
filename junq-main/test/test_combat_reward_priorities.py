import os
import unittest

from envs.reward_manager import RewardManager


class CombatRewardPriorityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.manager = RewardManager(os.path.join(root, "envs", "reward_rules.json"))

    def event(self, event_type, role, amount=1):
        return {
            "type": event_type,
            "side": "blue",
            "role": role,
            "amount": amount,
        }

    def test_fixed_damage_reward_per_hp_has_requested_priority(self):
        values = {
            role: self.manager._combat_event_value(self.event("damage_dealt", role))
            for role in ("attack_aircraft", "radar", "sam", "ground_force")
        }
        self.assertEqual(values["attack_aircraft"], 20.0)
        self.assertEqual(values["radar"], 12.0)
        self.assertEqual(values["sam"], 8.0)
        self.assertEqual(values["ground_force"], 0.0)
        self.assertGreater(values["attack_aircraft"], values["radar"])
        self.assertGreater(values["radar"], values["sam"])

    def test_destruction_bonus_has_requested_priority(self):
        values = {
            role: self.manager._combat_event_value(self.event("target_destroyed", role))
            for role in ("attack_aircraft", "radar", "sam", "ground_force")
        }
        self.assertEqual(values["attack_aircraft"], 60.0)
        self.assertEqual(values["radar"], 40.0)
        self.assertEqual(values["sam"], 25.0)
        self.assertEqual(values["ground_force"], 0.0)
        self.assertGreater(values["attack_aircraft"], values["radar"])
        self.assertGreater(values["radar"], values["sam"])


if __name__ == "__main__":
    unittest.main()