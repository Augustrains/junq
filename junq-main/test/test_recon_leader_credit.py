"""Regression tests for recon leader credit and shared-Actor batching."""

import unittest
from types import SimpleNamespace

import numpy as np

from envs.afsim_env import AFSIMIslandEnv
from envs.rl_interface import AFSIMRLInterface
from train.official_happo_adapter import HAPPOTrainer
from train.rule_driven_rollout_collector import RuleDrivenRolloutCollector


class ReconLeaderCreditTest(unittest.TestCase):
    def setUp(self):
        leader = SimpleNamespace(name="red_recon_2")
        controller = SimpleNamespace(
            active_groups={"recon_team_1": object()},
            ensure_group_leader=lambda group: leader,
        )
        env = SimpleNamespace(
            bounds={"lat_min": 0.0, "lat_max": 10.0, "lon_min": 0.0, "lon_max": 10.0},
            last_reward_events=[],
            negative_rewards_enabled=True,
            recon_state_config={"normalization": {"max_distance_m": 2000000.0}},
            recon_controller=controller,
        )
        env.reward_team_id_for_entity = lambda agent_type, name: "fixed_recon_1"
        env.get_reward_target_for_team = lambda agent_type, team_id: {
            "name": "blue_target", "lat": 0.0, "lon": 0.0, "effective_priority": 1.0
        }
        env.get_target_reward_contributors = lambda target: {"all": []}
        env.reward_movement_scale = lambda agent_type: 10.0
        env._distance_and_bearing = AFSIMIslandEnv._distance_and_bearing
        self.interface = AFSIMRLInterface(env, reward_profile="recon_attack_stage")

    @staticmethod
    def state(lon_norm):
        return {
            "scope": "persistent_agents",
            "agents": {
                name: {
                    "group_id": "recon_team_1",
                    "obs_by_name": {"lat_norm": 0.0, "lon_norm": lon_norm},
                }
                for name in ("red_recon_1", "red_recon_2", "red_recon_3")
            },
        }

    def test_team_progress_is_credited_once_to_current_leader(self):
        rewards, details = self.interface._compute_recon_attack_local_rewards(
            "recon", self.state(0.10), self.state(0.05), {}, reward_events=[]
        )
        self.assertEqual(rewards["red_recon_1"], 0.0)
        self.assertGreater(rewards["red_recon_2"], 0.0)
        self.assertEqual(rewards["red_recon_3"], 0.0)
        movement = [d for d in details if d["type"] == "group_move_toward_recon_target"]
        self.assertEqual(len(movement), 1)
        self.assertEqual(movement[0]["reward_recipient"], "red_recon_2")

    def test_only_recon_followers_are_filtered(self):
        assigned = RuleDrivenRolloutCollector._actor_trajectory_assigned
        self.assertTrue(assigned("recon", {"obs_by_name": {"is_leader": 1.0}}))
        self.assertFalse(assigned("recon", {"obs_by_name": {"is_leader": 0.0}}))
        self.assertTrue(assigned("attack", {"obs_by_name": {"is_leader": 0.0}}))

    def test_shared_recon_policy_receives_all_leader_rows(self):
        trainer = HAPPOTrainer.__new__(HAPPOTrainer)
        trainer.entity_names = {"recon": ("recon_shared",)}
        batch = {
            "entity_name": np.asarray(["red_recon_1", "red_recon_4"]),
            "action": np.asarray([[0.0], [1.0]], dtype=np.float32),
        }
        batches = list(trainer._entity_batches("recon", batch))
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0][0], "recon_shared")
        self.assertEqual(len(batches[0][1]["action"]), 2)


if __name__ == "__main__":
    unittest.main()