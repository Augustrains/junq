"""Tests for attack-only HAPPO Actor warm starts."""

import unittest
from types import SimpleNamespace

from train.train_recon_attack_parallel_eval import (
    import_attack_actor_state,
    set_attack_actor_learning_rate,
)


class FakeActor:
    def __init__(self):
        self.loaded = None

    def load_state_dict(self, state, strict=True):
        self.loaded = (dict(state), strict)


class AttackActorWarmStartTest(unittest.TestCase):
    def setUp(self):
        self.attack_actor = FakeActor()
        self.recon_actor = FakeActor()
        self.attack_optimizer = SimpleNamespace(param_groups=[{"lr": 3e-4}])
        self.recon_optimizer = SimpleNamespace(param_groups=[{"lr": 3e-4}])
        self.trainer = SimpleNamespace(
            entity_agent_type={"attack_1": "attack", "recon_1": "recon"},
            entity_policies={
                "attack_1": SimpleNamespace(
                    actor=self.attack_actor, actor_optimizer=self.attack_optimizer
                ),
                "recon_1": SimpleNamespace(
                    actor=self.recon_actor, actor_optimizer=self.recon_optimizer
                ),
            },
        )

    def test_imports_only_attack_actor(self):
        state = {
            "algorithm": "happo_official_on_policy",
            "policies": {
                "attack_1": {"agent_type": "attack", "actor": {"weight": 7}},
                "recon_1": {"agent_type": "recon", "actor": {"weight": 9}},
            },
        }
        imported = import_attack_actor_state(self.trainer, state)
        self.assertEqual(imported, ["attack_1"])
        self.assertEqual(self.attack_actor.loaded, ({"weight": 7}, True))
        self.assertIsNone(self.recon_actor.loaded)

    def test_sets_only_attack_actor_learning_rate(self):
        updated = set_attack_actor_learning_rate(self.trainer, 3e-5)
        self.assertEqual(updated, ["attack_1"])
        self.assertEqual(self.attack_optimizer.param_groups[0]["lr"], 3e-5)
        self.assertEqual(self.recon_optimizer.param_groups[0]["lr"], 3e-4)

    def test_missing_attack_policy_is_rejected(self):
        state = {"algorithm": "happo_official_on_policy", "policies": {}}
        with self.assertRaisesRegex(ValueError, "missing attack Actor"):
            import_attack_actor_state(self.trainer, state)


if __name__ == "__main__":
    unittest.main()
