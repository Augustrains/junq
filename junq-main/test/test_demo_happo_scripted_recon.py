from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np


SHOW_DIR = Path(__file__).resolve().parents[1] / "show"
sys.path.insert(0, str(SHOW_DIR))

from demo_happo_scripted_recon_warlock import ScriptedReconPolicy


class FakeNetworkPolicy:
    def __init__(self):
        self.calls = []

    def act_bottom(self, agent_type, entity_name, agent_state, global_state=None):
        self.calls.append((agent_type, entity_name))
        return {"action": 7, "logprob": -0.25, "value": 1.5}

    def value_bottom(self, *args, **kwargs):
        return 2.0


class FakeEnv:
    def __init__(self):
        self.now = 0.0
        self.config = {"red": {"carrier": ["red_carrier"]}}
        self.platforms = {
            "red_carrier": SimpleNamespace(
                name="red_carrier", side="red", role="carrier",
                alive=True, platform_id=1, lat=0.0, lon=-10_000.0,
            ),
            "red_recon_1": SimpleNamespace(
                name="red_recon_1", side="red", role="recon_aircraft",
                alive=True, platform_id=2, lat=0.0, lon=0.0,
            ),
            "blue_attack_1": SimpleNamespace(
                name="blue_attack_1", side="blue", role="attack_aircraft",
                alive=True, platform_id=3, lat=0.0, lon=100_000.0,
                task_status="BLUE_CAP_PATROL",
            ),
        }
        self.attack_ammo = {"blue_attack_1": {"fox3": 1, "agm": 1}}

    def _current_sim_time(self):
        return self.now

    @staticmethod
    def _distance_and_bearing(source_lat, source_lon, target_lat, target_lon):
        return abs(target_lon - source_lon), 0.0

    @staticmethod
    def _relative_north_east(source_lat, source_lon, target_lat, target_lon):
        return target_lat - source_lat, target_lon - source_lon


def test_recon_is_scripted_and_attack_still_uses_network():
    env = FakeEnv()
    network = FakeNetworkPolicy()
    policy = ScriptedReconPolicy(network, env)
    leader_state = {"group_id": "recon_team_1", "obs_by_name": {"is_leader": 1.0}}

    advance = policy.act_bottom("recon", "red_recon_1", leader_state)["action"]
    assert isinstance(advance, np.ndarray)
    assert advance[0] > 0.0
    assert network.calls == []

    env.now = 72.0
    env.attack_ammo["blue_attack_1"]["fox3"] = 0
    retreat = policy.act_bottom("recon", "red_recon_1", leader_state)["action"]
    assert retreat[0] < 0.0

    attack = policy.act_bottom("attack", "red_attack_1", {}, global_state=np.zeros(1))
    assert attack == {"action": 7, "logprob": -0.25, "value": 1.5}
    assert network.calls == [("attack", "red_attack_1")]
