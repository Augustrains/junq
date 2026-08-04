"""Regression tests for leader and wingman RETURN_HOME behavior.

Run from the repository root with:
    python -m unittest test.test_attack_independent_return -v
"""

import unittest

from envs.afsim_env import AFSIMIslandEnv, PlatformState
from envs.controllers.attack_controller import AttackController


class AttackIndependentReturnTest(unittest.TestCase):
    def setUp(self):
        actions = {
            "actions": [
                {"id": 0, "name": "HOLD", "afsim_task": "ATTACK_HOLD"},
                {"id": 1, "name": "RETURN_HOME", "afsim_task": "RETREAT"},
                {
                    "id": 40,
                    "name": "RETURN_TO_LEADER",
                    "afsim_task": "ATTACK_REJOIN_FORMATION",
                },
            ]
        }
        red = {
            "attack_aircraft": ["leader", "wingman"],
            "commandable_attack_aircraft": ["leader", "wingman"],
            "fixed_attack_teams": [["leader", "wingman"]],
        }
        controller = AttackController(red, actions)
        leader = PlatformState(
            name="leader", role="attack_aircraft", side="red",
            platform_id=1, alive=True, lat=20.0, lon=120.0,
        )
        wingman = PlatformState(
            name="wingman", role="attack_aircraft", side="red",
            platform_id=2, alive=True, lat=20.0, lon=120.1,
        )
        group = controller._new_group("attack_team_1", "", {}, [leader, wingman])
        controller.active_groups[group.group_id] = group

        env = AFSIMIslandEnv.__new__(AFSIMIslandEnv)
        env.attack_controller = controller
        env.platforms = {leader.name: leader, wingman.name: wingman}
        env.attack_ammo = {
            leader.name: {"fox3": 1, "agm": 1},
            wingman.name: {"fox3": 0, "agm": 0},
        }
        env.pending_attack_returns = {}
        env.pending_attack_fire_commands = {}
        env.last_reward_events = []
        env._attack_action_target_slots = lambda: []
        env._attack_action_allowed = lambda group_id, aircraft_name, action_id: True
        env._current_sim_time = lambda: 123.0
        env._send = lambda message: self.sent.append(dict(message))

        self.env = env
        self.group = group
        self.leader = leader
        self.wingman = wingman
        self.sent = []

    def test_return_home_unlocks_only_after_weapon_expenditure(self):
        leader_mask = self.env._build_attack_action_mask(self.leader, self.group, {})
        wingman_mask = self.env._build_attack_action_mask(self.wingman, self.group, {})

        self.assertEqual(leader_mask[1], 1.0)
        self.assertEqual(leader_mask[0], 0.0)
        self.assertEqual(wingman_mask[1], 1.0)
        self.assertEqual(wingman_mask[0], 0.0)
        self.assertEqual(sum(wingman_mask), 1.0)

    def test_hold_is_masked_while_one_weapon_type_remains(self):
        self.env.attack_ammo["wingman"] = {"fox3": 0, "agm": 1}

        mask = self.env._build_attack_action_mask(self.wingman, self.group, {})

        self.assertEqual(mask[0], 0.0)
        self.assertEqual(mask[1], 1.0)

    def test_wingman_return_home_only_returns_that_wingman(self):
        ok = self.env.apply_attack_aircraft_action(
            self.group.group_id, self.wingman.name, 1
        )

        self.assertTrue(ok)
        self.assertEqual([message["PlatformName"] for message in self.sent], ["wingman"])
        self.assertNotIn("leader", self.env.pending_attack_returns)
        self.assertEqual(
            self.env.pending_attack_returns["wingman"]["phase"], "returning"
        )

    def test_leader_return_home_still_returns_entire_formation(self):
        self.env.attack_ammo["leader"]["fox3"] = 0
        ok = self.env.apply_attack_aircraft_action(
            self.group.group_id, self.leader.name, 1
        )

        self.assertTrue(ok)
        self.assertEqual(
            [message["PlatformName"] for message in self.sent],
            ["leader", "wingman"],
        )
        self.assertEqual(set(self.env.pending_attack_returns), {"leader", "wingman"})



    def test_ground_attack_unlock_requires_all_four_prerequisites_destroyed(self):
        prerequisite_names = (
            "blue_radar_1", "blue_sam_1", "blue_sam_2", "blue_sam_4"
        )
        for name in prerequisite_names:
            self.env.platforms[name] = PlatformState(
                name=name, role="radar" if "radar" in name else "sam",
                side="blue", alive=True,
            )

        self.assertFalse(self.env._ground_attack_unlocked())
        for name in prerequisite_names:
            self.env.platforms[name].alive = False
        self.assertTrue(self.env._ground_attack_unlocked())


if __name__ == "__main__":
    unittest.main()