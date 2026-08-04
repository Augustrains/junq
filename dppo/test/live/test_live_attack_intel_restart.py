import json
import time
from envs.afsim_env import AFSIMIslandEnv

CONFIG = "envs/afsim_units.json"

def log(label, **data):
    print(label, json.dumps(data, ensure_ascii=False), flush=True)

def main():
    env = AFSIMIslandEnv(config_path=CONFIG, bind=True, auto_start_warlock=True)
    try:
        assert env.wait_for_platforms(["red_attack_1", "blue_attack_6"], timeout=45)
        env.reset()
        red = env.platforms["red_attack_1"]
        blue = env.platforms["blue_attack_6"]
        old = [blue.lat + 0.20, blue.lon - 0.20, blue.alt]
        env._capture_attack_target_snapshot(blue.name, {
            "Lat": old[0], "Lon": old[1], "Alt": old[2],
            "Type": blue.platform_type or blue.role,
            "CurrentHP": blue.current_hp, "MaxHP": blue.max_hp,
        }, env._current_sim_time())
        log("START", red=[red.lat, red.lon, red.alt], real_blue=[blue.lat, blue.lon, blue.alt], old_track=old)
        assert env.apply_attack_aircraft_action("attack_team_1", "red_attack_1", 7)
        deadline = time.time() + 65
        stopped = False
        while time.time() < deadline:
            env._drain_messages(timeout=0.5)
            red = env.platforms["red_attack_1"]
            blue = env.platforms["blue_attack_6"]
            if "red_attack_1" not in env.pending_attack_approaches and red.task == "ATTACK_HOLD":
                stopped = True
                break
        distance = env._slant_distance_and_bearing(red.lat, red.lon, red.alt, blue.lat, blue.lon, blue.alt)[0]
        log("RANGE_STOP", stopped=stopped, distance_m=distance, task=red.task, status=red.task_status, pending=env.pending_attack_approaches.get("red_attack_1"))
        assert stopped
        assert env.apply_attack_aircraft_action("attack_team_1", "red_attack_1", 7)
        for _ in range(8):
            env._drain_messages(timeout=0.4)
        fire = env.pending_attack_fire_commands.get("red_attack_1", {})
        log("NEXT_DECISION_FIRE", pending_fire=fire, events=env.last_reward_events[-8:])
        assert fire.get("target_name") == "blue_attack_6"
    finally:
        env.close()

    restarted = AFSIMIslandEnv(config_path=CONFIG, bind=True, auto_start_warlock=True)
    try:
        restarted.prepare_for_scenario_restart()
        priors = restarted.attack_target_snapshots
        assert not restarted.enemy_track_memory
        assert not restarted.detected_targets
        assert not restarted.attack_local_detections
        assert priors and all(v.get("source") == "initial_base_prior" and v.get("last_seen") is None for v in priors.values())
        log("RESTART_CLEAN", track_memory=len(restarted.enemy_track_memory), detections=len(restarted.detected_targets), local_detections=len(restarted.attack_local_detections), prior_count=len(priors), blue6_prior=priors["blue_attack_6"])
        print("LIVE_TEST_PASS", flush=True)
    finally:
        restarted.close()

if __name__ == "__main__":
    main()