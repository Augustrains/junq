"""Tests for attack-aircraft return-to-carrier local rewards."""

import unittest
from types import SimpleNamespace

from envs.afsim_env import AFSIMIslandEnv
from envs.rl_interface import AFSIMRLInterface


class AttackReturnRewardTest(unittest.TestCase):
    def setUp(self):
        rules = {
            "attack_return_rewards": {
                "enabled": True,
                "progress_per_10km": 0.5,
                "max_progress_m_per_step": 50000.0,
                "arrival_at_carrier": 6.0,
                "rearm_completed_base": 8.0,
                "rearm_completed_per_missile": 4.0,
            }
        }
        env = SimpleNamespace(
            bounds={"lat_min": 0.0, "lat_max": 10.0, "lon_min": 0.0, "lon_max": 10.0},
            reward_manager=SimpleNamespace(rules=rules),
            last_reward_events=[],
            negative_rewards_enabled=True,
            attack_state_config={"normalization": {"max_distance_m": 400000.0}},
        )
        env.reward_team_id_for_entity = lambda agent_type, name: ""
        env.get_reward_target_for_team = lambda agent_type, team_id: None
        env.get_target_reward_contributors = lambda target: {"all": []}
        env.reward_movement_scale = lambda agent_type: 1.0
        env._distance_and_bearing = AFSIMIslandEnv._distance_and_bearing
        self.interface = AFSIMRLInterface(env, reward_profile="recon_attack_stage")

    @staticmethod
    def state(obs):
        return {
            "scope": "persistent_agents",
            "agents": {"attack_1": {"group_id": "attack_team_1", "obs_by_name": obs}},
        }

    @staticmethod
    def obs(lon_norm, returning=0.0, rearming=0.0, aam=0.0, agm=1.0):
        return {
            "lat_norm": 0.0,
            "lon_norm": lon_norm,
            "friendly_carrier_lat_norm": 0.0,
            "friendly_carrier_lon_norm": 0.0,
            "returning_to_carrier": returning,
            "rearming": rearming,
            "aam_count_norm": aam,
            "agm_count_norm": agm,
        }

    def rewards(self, before, after, events=None):
        return self.interface._compute_recon_attack_local_rewards(
            "attack", self.state(before), self.state(after),
            {"attack_1": {"action_id": 0}}, reward_events=events or [],
        )

    def test_no_return_progress_reward_before_returning(self):
        rewards, _ = self.rewards(self.obs(0.10), self.obs(0.05))
        self.assertEqual(rewards["attack_1"], 0.0)

    def test_return_progress_reward_is_positive_and_clipped(self):
        rewards, details = self.rewards(
            self.obs(0.10, returning=1.0),
            self.obs(0.01, returning=1.0),
        )
        self.assertAlmostEqual(rewards["attack_1"], 2.5)
        self.assertEqual(details[-1]["type"], "attack_return_carrier_progress")

    def test_arrival_reward(self):
        rewards, details = self.rewards(
            self.obs(0.01, returning=1.0),
            self.obs(0.01, rearming=1.0),
        )
        self.assertEqual(rewards["attack_1"], 6.0)
        self.assertEqual(details[-1]["type"], "attack_return_arrived_at_carrier")

    def test_rearm_reward_scales_with_loaded_missiles(self):
        event = {
            "type": "attack_rearmed", "platform": "attack_1",
            "aam_loaded": 1, "agm_loaded": 1,
        }
        rewards, details = self.rewards(
            self.obs(0.0, rearming=1.0, aam=0.0, agm=0.0),
            self.obs(0.0, aam=1.0, agm=1.0),
            [event],
        )
        self.assertEqual(rewards["attack_1"], 16.0)
        self.assertEqual(details[-1]["type"], "attack_return_rearm_completed")


if __name__ == "__main__":
    unittest.main()
