import json
from pathlib import Path
from envs.afsim_env import AFSIMIslandEnv

def main():
    cfg = str(Path.cwd() / "test" / "configs" / "afsim_units_agm_test.json")
    env = AFSIMIslandEnv(config_path=cfg, bind=True, auto_start_warlock=True)
    try:
        ready = env.wait_for_platforms(["red_attack_1", "blue_ground_1"], timeout=30)
        print("ready", ready, flush=True)
        if not ready:
            return 1
        target = env.platforms["blue_ground_1"]
        env.detected_targets["blue_ground_1"] = {
            "Name": target.name, "Type": target.platform_type or target.role,
            "Lat": target.lat, "Lon": target.lon, "Alt": target.alt,
            "known": True, "alive": target.alive,
            "CurrentHP": target.current_hp, "MaxHP": target.max_hp,
        }
        group = None
        for team_index in range(4):
            group = env.start_attack_group("blue_ground_1", team_index)
            if group is not None:
                break
        if group is None:
            print("group_failed", env.last_reward_events[-6:], flush=True)
            return 2
        action_id = next(i for i, spec in env.attack_controller.action_specs.items()
                         if spec.get("target_name") == "blue_ground_1")
        aircraft_name = group.leader_name
        print("group", group.group_id, "aircraft", aircraft_name, "action", action_id, flush=True)
        print("action_sent", env.apply_attack_aircraft_action(
            group.group_id, aircraft_name, action_id), flush=True)
        for step in range(15):
            env._drain_messages(timeout=1.0)
            aircraft, target = env.platforms[aircraft_name], env.platforms["blue_ground_1"]
            horizontal = env._distance_and_bearing(aircraft.lat, aircraft.lon,
                                                    target.lat, target.lon)[0]
            print(json.dumps({
                "step": step, "sim_time": env._current_sim_time(),
                "horizontal_distance_m": round(horizontal, 1),
                "altitude_m": round(float(aircraft.alt), 1),
                "speed_mps": round(float(aircraft.speed), 1),
                "task": aircraft.task, "task_status": aircraft.task_status,
                "pending_approach": aircraft_name in env.pending_attack_approaches,
                "pending_fire": aircraft_name in env.pending_attack_fire_commands,
                "target_hp": target.current_hp, "target_alive": target.alive,
            }), flush=True)
        print("events", json.dumps(env.last_reward_events[-20:], ensure_ascii=False), flush=True)
        return 0
    finally:
        env.close()

if __name__ == "__main__":
    raise SystemExit(main())
