"""Scenario test: attack first, then lock every aircraft to RETURN_HOME.

Run from the repository root with:
    python -m unittest test.test_attack_return_action_lock -v
"""

import unittest

from envs.afsim_env import AFSIMIslandEnv


class AttackReturnActionLockTest(unittest.TestCase):
    def setUp(self):
        self.env = AFSIMIslandEnv(bind=False)
        for index, platform in enumerate(self.env.platforms.values(), start=1):
            platform.platform_id = index

        target = self.env.platforms["blue_attack_1"]
        target.lat, target.lon, target.alt = 25.0, 121.0, 5000.0
        target.alive = True
        target.detected = True
        self.env.detected_targets[target.name] = {
            "Name": target.name,
            "Type": "BLUE_ATTACK_AIRCRAFT",
            "Lat": target.lat,
            "Lon": target.lon,
            "Alt": target.alt,
            "known": True,
            "alive": True,
        }
        # Put the fixed attack team inside AAM range so the first target
        # action is a real fire attempt and unlocks RETURN_HOME.
        for name in self.env.config["red"]["fixed_attack_teams"][0]:
            member = self.env.platforms[name]
            member.lat, member.lon, member.alt = 25.0, 120.9, 5000.0
            member.at_home = False

        self.group = self.env.start_attack_group(target.name, fixed_team_index=0)
        self.assertIsNotNone(self.group)
        self.aircraft = self.env.attack_controller.ensure_group_leader(self.group)
        self.assertIsNotNone(self.aircraft)
        state = self.env.get_attack_task_state(self.group.group_id)
        self.action_ids = {
            row["name"]: row["id"] for row in state["action_table"]
        }
        self.sent_messages = []
        self.env._send = lambda message: self.sent_messages.append(dict(message))

    def tearDown(self):
        self.env.close()

    def test_attack_then_return_masks_and_rejects_every_other_action(self):
        attack_id = next(
            action_id for name, action_id in self.action_ids.items()
            if name.startswith("ATTACK_TARGET_")
        )
        self.assertTrue(self.env.apply_attack_aircraft_action(
            self.group.group_id, self.aircraft.name, attack_id
        ))
        self.assertIn(
            self.sent_messages[-1]["Task"],
            ("FIRE_AAM", "FIRE_AGM"),
        )
        # Model the end of the short fire-command acceptance window. During
        # that window all actions are intentionally blocked except HOLD.
        self.env.pending_attack_fire_commands.pop(self.aircraft.name, None)

        return_id = self.action_ids["RETURN_HOME"]
        self.assertTrue(self.env.apply_attack_aircraft_action(
            self.group.group_id, self.aircraft.name, return_id
        ))
        self.assertEqual(self.sent_messages[-1]["Task"], "RETREAT")
        for member in self.group.platforms:
            self.assertEqual(
                self.env.pending_attack_returns[member.name]["phase"], "returning"
            )

        state = self.env.get_attack_task_state(self.group.group_id)
        message_count = len(self.sent_messages)
        for member in self.group.platforms:
            mask = state["aircraft"][member.name]["action_mask"]
            enabled = [index for index, value in enumerate(mask) if value > 0.0]
            self.assertEqual(enabled, [return_id])

            for action_id in sorted(self.env.attack_controller.action_specs):
                if action_id == return_id:
                    continue
                self.assertFalse(self.env.apply_attack_aircraft_action(
                    self.group.group_id, member.name, action_id
                ))

            # RETURN_HOME now means continue the existing RETREAT. No new
            # ATTACK_HOLD or other command may be emitted.
            self.assertTrue(self.env.apply_attack_aircraft_action(
                self.group.group_id, member.name, return_id
            ))
            self.assertEqual(member.task, "RETREAT")

        self.assertEqual(len(self.sent_messages), message_count)


if __name__ == "__main__":
    unittest.main()