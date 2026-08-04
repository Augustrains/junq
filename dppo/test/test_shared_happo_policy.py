import numpy as np
from train.happo_trainer import HAPPOConfig, HAPPOTrainer


def _specs():
    return {
        "recon": {"obs_dim": 3, "action_dim": 2, "action_type": "discrete", "entity_names": ["recon_1", "recon_2"]},
        "attack": {"obs_dim": 3, "action_dim": 2, "action_type": "discrete", "entity_names": ["attack_1", "attack_2", "attack_3"]},
    }


def _trainer():
    return HAPPOTrainer(_specs(), 4, ("recon", "attack"), (8, 8), HAPPOConfig(share_policy_by_type=True, update_epochs=1))


def test_shared_policy_uses_one_network_per_type():
    trainer = _trainer()
    assert set(trainer.entity_policies) == {"recon", "attack"}
    assert trainer._policy_id("recon", "recon_1") == "recon"
    assert trainer._policy_id("attack", "attack_3") == "attack"


def test_shared_policy_pools_all_same_type_rows():
    trainer = _trainer()
    batch = {"action": np.asarray([0, 1, 0]), "entity_name": np.asarray(["recon_1", "recon_2", "recon_1"])}
    pooled = list(trainer._entity_batches("recon", batch))
    assert len(pooled) == 1
    assert pooled[0][0] == "recon"
    assert len(pooled[0][1]["action"]) == 3