"""Smoke test for continuous transport movement and automatic shore landing.

Run from the repository root:
    python train/transport_continuous_shore_auto_landing_smoke_test.py

This test is offline: it does not start Warlock or require UDP traffic.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.afsim_env import AFSIMIslandEnv
from envs.rl_interface import AFSIMRLInterface


SEA_POINT = (23.700000, 120.000000)
LAND_POINT = (23.700000, 120.300000)


def main():
    env = AFSIMIslandEnv()
    try:
        api = AFSIMRLInterface(env)
        landing_spec = api.get_agent_specs()["landing"]
        assert landing_spec["action_type"] == "continuous"
        assert landing_spec["action_dim"] == 3
        assert tuple(landing_spec["action_shape"]) == (3,)

        teams = env.initialize_bottom_teams()
        group_id = teams["landing"][0]
        group = env.landing_controller.active_groups[group_id]
        ship = group.platforms[0]
        ship_name = ship.name

        before = env.get_landing_task_state(group_id)
        assert len(before["ships"][ship_name]["action_mask"]) == 3

        after, _, _, info = api.step_task_agent(
            "landing", group_id, {ship_name: [1.0, 0.0, 0.0]}, advance_sim=False
        )
        assert info["actions"][ship_name]["sent"]
        assert len(after["ships"][ship_name]["action_mask"]) == 3
        assert ship.task == "LANDING_MOVE_POINT"

        berth_lat, berth_lon, shoreline = env._clip_transport_move_to_shore(
            *SEA_POINT, *LAND_POINT
        )
        assert shoreline is not None
        assert not env.is_on_annotated_island(berth_lat, berth_lon)
        shore_status = env.get_island_status(berth_lat, berth_lon)
        assert shore_status["shore_distance_m"] <= 200.0

        ship.lat = berth_lat
        ship.lon = berth_lon
        env._maybe_start_automatic_landing(ship)
        assert ship_name in env.pending_landing_unloads
        assert ship.task == "LANDING_UNLOADING"

        manifest = [
            name for name, status in env.ground_status.items()
            if status.get("transport") == ship_name and status.get("on_ship")
        ]
        assert manifest, "test transport must carry at least one ground unit"

        env._confirm_landing_unload(ship_name)
        assert env.landing_cargo[ship_name]["army_landed"]
        assert not env.landing_cargo[ship_name]["has_army"]
        assert all(
            env.ground_status[name]["landed"]
            and not env.ground_status[name]["on_ship"]
            for name in manifest
        )

        print(
            "PASS",
            {
                "transport": ship_name,
                "shoreline": shoreline,
                "berth": (berth_lat, berth_lon),
                "landed_ground_units": manifest,
            },
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()

