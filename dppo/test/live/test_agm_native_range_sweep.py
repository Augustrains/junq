import json
from pathlib import Path

from envs.afsim_env import AFSIMIslandEnv


RANGES_M = [3000, 5000, 8000, 10000, 15000, 20000, 25000, 30000, 35000, 39000]


def main():
    config = Path.cwd() / "test" / "configs" / "afsim_units_agm_native_range_test.json"
    env = AFSIMIslandEnv(config_path=str(config), bind=True, auto_start_warlock=True)
    try:
        names = [f"red_attack_{i}" for i in range(1, 11)]
        targets = [f"blue_ground_{i}" for i in range(1, 11)]
        if not env.wait_for_platforms(names + targets, timeout=40):
            print("READY_FAILED")
            return 1

        for shooter_name in names:
            shooter = env.platforms[shooter_name]
            env._send({
                "MsgType": "AssignTask",
                "PlatformId": shooter.platform_id,
                "PlatformName": shooter_name,
                "Task": "ATTACK_MOVE_POINT",
                "MovePosition": [shooter.lat, shooter.lon, 3000.0],
                "CommandSpeedMps": 138.888889,
            })
        for _ in range(30):
            env._drain_messages(timeout=1.0)
            if all(env.platforms[name].alt >= 2990.0 for name in names):
                break
        print("LAUNCH_ALTITUDES=" + json.dumps({
            name: round(env.platforms[name].alt, 1) for name in names
        }))

        launches = []
        for i, (shooter_name, target_name, nominal_range) in enumerate(
            zip(names, targets, RANGES_M), start=1
        ):
            shooter = env.platforms[shooter_name]
            target = env.platforms[target_name]
            actual_range = env._distance_and_bearing(
                shooter.lat, shooter.lon, target.lat, target.lon
            )[0]
            env._send({
                "MsgType": "AssignTask",
                "PlatformId": shooter.platform_id,
                "PlatformName": shooter_name,
                "Task": "FIRE_AGM",
                "TargetName": target_name,
                "TargetPosition": [target.lat, target.lon, target.alt],
                "Weapon": "agm",
            })
            launches.append({
                "shooter": shooter_name,
                "target": target_name,
                "nominal_range_m": nominal_range,
                "actual_range_m": round(actual_range, 1),
                "initial_hp": target.current_hp,
            })

        start_sim = env._current_sim_time()
        for _ in range(45):
            env._drain_messages(timeout=1.0)
            if env._current_sim_time() - start_sim >= 180.0:
                break

        results = []
        for launch in launches:
            target = env.platforms[launch["target"]]
            shooter = env.platforms[launch["shooter"]]
            result = dict(launch)
            result.update({
                "final_hp": target.current_hp,
                "hit": target.current_hp < launch["initial_hp"],
                "target_alive": target.alive,
                "shooter_agm_remaining": env.attack_ammo.get(
                    shooter.name, {}
                ).get("agm"),
                "shooter_task_status": shooter.task_status,
            })
            results.append(result)
        print("RANGE_SWEEP_RESULTS=" + json.dumps(results, ensure_ascii=False))
        print("EVENTS=" + json.dumps(env.last_reward_events[-100:], ensure_ascii=False))
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
