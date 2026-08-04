"""Live AFSIM validation for ground shared-reconnaissance approach/fire logic.

Runs the independent ground_shared_recon_live_test scenario.  It never changes
the main island-assault scenario.
"""
import time

from envs.afsim_env import AFSIMIslandEnv
from envs.controllers.ground_controller import GroundGroup

CONFIG_PATH = "test/configs/afsim_units_ground_shared_recon_test.json"
LOCAL_ADDRESS = ("127.0.0.1", 50050)
PLATFORM_NAMES = ("red_ground_test", "red_recon_1", "blue_ground_1")
OBJECTIVE_PRIOR = (24.0, 118.99)
GROUND_WEAPON_RANGE_M = 5000.0
MOVE_TIMEOUT_SECONDS = 3600.0


def main():
    env = AFSIMIslandEnv(
        config_path=CONFIG_PATH,
        auto_start_warlock=True,
        local_address=LOCAL_ADDRESS,
    )
    sent_messages = []
    send_to_afsim = env._send

    def record_and_send(message):
        sent_messages.append(dict(message))
        return send_to_afsim(message)

    env._send = record_and_send
    keep_warlock_running = False
    try:
        if not env.wait_for_platforms(list(PLATFORM_NAMES), timeout=45.0):
            missing = {
                name: env.platforms[name].platform_id
                for name in PLATFORM_NAMES
            }
            raise RuntimeError("platform registration timeout: {0}".format(missing))

        red_ground, recon, blue_ground = (
            env.platforms[name] for name in PLATFORM_NAMES
        )
        env.ground_status[red_ground.name] = {"on_ship": False, "landed": True}
        group = GroundGroup(
            "ground_shared_recon_live",
            {
                "name": "objective_prior",
                "lat": OBJECTIVE_PRIOR[0],
                "lon": OBJECTIVE_PRIOR[1],
                "radius_m": GROUND_WEAPON_RANGE_M,
            },
            [red_ground],
        )
        env.ground_controller.active_groups[group.group_id] = group

        # Phase 1: shared initial intelligence says the enemy is at the objective.
        env.enemy_track_memory[blue_ground.name] = {
            "Name": blue_ground.name,
            "Type": blue_ground.platform_type or blue_ground.role,
            "Lat": OBJECTIVE_PRIOR[0],
            "Lon": OBJECTIVE_PRIOR[1],
            "Alt": 0.0,
            "alive": True,
            "TrackSource": "objective_prior",
            "LastSeen": 0.0,
        }
        target_action = 2 + env.ground_fixed_target_names.index(blue_ground.name)
        assert env.apply_ground_unit_action(
            group.group_id, red_ground.name, target_action
        )
        initial = sent_messages[-1]
        print("[1] Initial prior:", initial["Task"], initial["ObjectivePosition"])
        assert initial["Task"] == "GROUND_MOVE_POINT"
        assert initial["ObjectivePosition"] == list(OBJECTIVE_PRIOR)

        # Complete the first (default-position) movement before reconnaissance
        # refreshes the target track, so the test has exactly three target actions.
        first_move = initial["MovePosition"]
        movement_deadline = time.time() + MOVE_TIMEOUT_SECONDS
        distance_to_first_move = float("inf")
        while time.time() < movement_deadline:
            env._drain_messages(timeout=0.5)
            distance_to_first_move, _ = env._distance_and_bearing(
                red_ground.lat, red_ground.lon, first_move[0], first_move[1]
            )
            if distance_to_first_move <= 150.0:
                break
        print("[1b] Initial move arrived within {:.1f} m".format(distance_to_first_move))
        if distance_to_first_move > 150.0:
            raise RuntimeError("AFSIM did not complete the initial-prior movement")

        # Phase 2: the recon aircraft's actual sensor report updates shared memory.
        recon.task, recon.task_status, recon.at_home = "RECON", "ASSIGNED", False
        env._update_red_recon_detections()
        track = env.enemy_track_memory[blue_ground.name]
        print("[2] Recon track:", track["Lat"], track["Lon"], track["TrackSource"])
        assert track["TrackSource"] == recon.name

        assert env.apply_ground_unit_action(
            group.group_id, red_ground.name, target_action
        )
        refreshed = sent_messages[-1]
        print("[3] Refreshed move:", refreshed["Task"], refreshed["ObjectivePosition"])
        assert refreshed["Task"] == "GROUND_MOVE_POINT"
        assert refreshed["ObjectivePosition"] == [blue_ground.lat, blue_ground.lon]

        # Phase 3: the refreshed second target action reaches weapon range.
        deadline = time.time() + MOVE_TIMEOUT_SECONDS
        distance_m = float("inf")
        while time.time() < deadline:
            env._drain_messages(timeout=0.5)
            distance_m, _ = env._distance_and_bearing_to_platform(
                red_ground, blue_ground
            )
            if distance_m <= GROUND_WEAPON_RANGE_M:
                break
        print("[4] Range after second target action: {:.1f} m".format(distance_m))
        if distance_m > GROUND_WEAPON_RANGE_M:
            raise RuntimeError("second target action did not enter weapon range")
        # Phase 4: stop in range, then fire on the next target action.
        assert env.apply_ground_unit_action(group.group_id, red_ground.name, 0)
        stop = sent_messages[-1]
        print("[5] Stop:", stop["Task"])
        assert stop["Task"] == "GROUND_HOLD"

        assert env.apply_ground_unit_action(
            group.group_id, red_ground.name, target_action
        )
        fire = sent_messages[-1]
        print("[6] Fire:", fire["Task"], fire["TargetName"])
        assert fire["Task"] == "GROUND_FIRE"
        assert fire["TargetName"] == blue_ground.name
        print("PASS: ground shared-reconnaissance flow verified in AFSIM.")
        keep_warlock_running = True
        print("Warlock remains running; stop it manually from the Warlock window when finished.")
    finally:
        if not keep_warlock_running:
            env.close()


if __name__ == "__main__":
    main()
