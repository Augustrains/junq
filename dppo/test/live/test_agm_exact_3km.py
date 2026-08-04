import json
from pathlib import Path

from envs.afsim_env import AFSIMIslandEnv


def main():
    cfg = Path.cwd() / "test" / "configs" / "afsim_units_agm_native_range_test.json"
    env = AFSIMIslandEnv(config_path=str(cfg), bind=True, auto_start_warlock=True)
    try:
        if not env.wait_for_platforms(["red_attack_1", "blue_ground_1"], timeout=40):
            print("READY_FAILED")
            return 1
        shooter = env.platforms["red_attack_1"]
        target = env.platforms["blue_ground_1"]
        env._send({
            "MsgType": "AssignTask", "PlatformId": shooter.platform_id,
            "PlatformName": shooter.name, "Task": "ATTACK_MOVE_POINT",
            "MovePosition": [target.lat, target.lon, 3000.0],
            "CommandSpeedMps": 138.888889,
        })
        launch_range = None
        for _ in range(80):
            env._drain_messages(timeout=1.0)
            distance = env._distance_and_bearing(
                shooter.lat, shooter.lon, target.lat, target.lon
            )[0]
            if shooter.alt >= 2990.0 and distance <= 6000.0:
                launch_range = distance
                initial_hp = target.current_hp
                env._send({
                    "MsgType": "AssignTask", "PlatformId": shooter.platform_id,
                    "PlatformName": shooter.name, "Task": "ATTACK_HOLD",
                })
                env._drain_messages(timeout=1.0)
                fire = {
                    "MsgType": "AssignTask", "PlatformId": shooter.platform_id,
                    "PlatformName": shooter.name, "Task": "FIRE_AGM",
                    "TargetName": target.name,
                    "TargetPosition": [target.lat, target.lon, target.alt],
                    "Weapon": "agm",
                }
                for _ in range(5):
                    env._send(fire)
                    env._drain_messages(timeout=0.5)
                    if env.attack_ammo.get(shooter.name, {}).get("agm", 1) == 0:
                        break
                break
        if launch_range is None:
            print("DID_NOT_REACH_3KM")
            return 2
        launch_time = env._current_sim_time()
        for _ in range(50):
            env._drain_messages(timeout=1.0)
            if target.current_hp < initial_hp:
                break
            if env._current_sim_time() - launch_time >= 120.0:
                break
        print("EXACT_3KM_RESULT=" + json.dumps({
            "launch_range_m": round(launch_range, 1),
            "launch_altitude_m": round(shooter.alt, 1),
            "initial_hp": initial_hp,
            "final_hp": target.current_hp,
            "hit": target.current_hp < initial_hp,
            "agm_remaining": env.attack_ammo.get(shooter.name, {}).get("agm"),
            "elapsed_sim_seconds": round(env._current_sim_time() - launch_time, 1),
        }))
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
