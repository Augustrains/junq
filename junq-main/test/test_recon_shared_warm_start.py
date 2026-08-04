"""Tests for initializing a shared recon Actor from red_recon_1."""

import unittest
from types import SimpleNamespace

from train.train_recon_attack_parallel_eval import import_shared_recon_actor_state


class FakeActor:
    def __init__(self):
        self.loaded = None

    def load_state_dict(self, state, strict=True):
        self.loaded = (dict(state), strict)


class SharedReconWarmStartTest(unittest.TestCase):
    def test_imports_red_recon_1_actor_into_shared_policy(self):
        actor = FakeActor()
        trainer = SimpleNamespace(
            entity_agent_type={"recon_shared": "recon", "red_attack_1": "attack"},
            entity_policies={"recon_shared": SimpleNamespace(actor=actor)},
        )
        state = {
            "algorithm": "happo_official_on_policy",
            "policies": {
                "red_recon_1": {"agent_type": "recon", "actor": {"weight": 11}}
            },
        }
        target = import_shared_recon_actor_state(trainer, state)
        self.assertEqual(target, "recon_shared")
        self.assertEqual(actor.loaded, ({"weight": 11}, True))

    def test_rejects_non_shared_recon_target(self):
        trainer = SimpleNamespace(
            entity_agent_type={"red_recon_1": "recon", "red_recon_2": "recon"},
            entity_policies={},
        )
        state = {"algorithm": "happo_official_on_policy", "policies": {}}
        with self.assertRaisesRegex(ValueError, "exactly one shared recon Actor"):
            import_shared_recon_actor_state(trainer, state)


if __name__ == "__main__":
    unittest.main()