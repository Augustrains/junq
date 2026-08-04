"""Live Warlock proof that the powered AGM hits from approximately 45 km."""

import json
import math
import sys
import time
from pathlib import Path

from envs.afsim_env import AFSIMIslandEnv


SHOOTER_NAME = "red_attack_1"
TARGET_NAME = "blue_ground_1"
DESIRED_SLANT_M = 43850.0
LAUNCH_ALT_M = 3000.0
TRACE_PATH = Path(r"D:\junq\afsim_work\afsim-2.9.0-win64_bin\demos\air_to_air\scenarios\red_attack_trace.log")


def drain(env, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        env._drain_messages(timeout=0.1)


def main():
    env = AFSIMIslandEnv(bind=True, auto_start_warlock=True)
    try:
        if not env.wait_for_platforms(
            [SHOOTER_NAME, TARGET_NAME, "red_carrier"], timeout=45
        ):
            raise RuntimeError("required live platforms were not reported")
        drain(env, 1.0)
        shooter = env.platforms[SHOOTER_NAME]
        target = env.platforms[TARGET_NAME]
        initial_hp = float(target.current_hp)

        north, east = env._relative_north_east(
            target.lat, target.lon, shooter.lat, shooter.lon
        )
        length = max(1.0, math.hypot(north, east))
        horizontal_m = math.sqrt(
            max(0.0, DESIRED_SLANT_M ** 2 - LAUNCH_ALT_M ** 2)
        )
        launch_lat, launch_lon = env.attack_controller._offset_lat_lon(
            target.lat, target.lon,
            north / length * horizontal_m,
            east / length * horizontal_m,
        )
        env._send({
            "MsgType": "AssignTask",
            "PlatformId": shooter.platform_id,
            "PlatformName": shooter.name,
            "Task": "ATTACK_MOVE_POINT",
            "MovePosition": [launch_lat, launch_lon, LAUNCH_ALT_M],
            "CommandSpeedMps": 138.888889,
        })
        print(json.dumps({
            "phase": "moving_to_launch_point",
            "shooter": shooter.name,
            "target": target.name,
            "desired_slant_m": DESIRED_SLANT_M,
            "launch_point": [launch_lat, launch_lon, LAUNCH_ALT_M],
        }), flush=True)

        deadline = time.monotonic() + 180.0
        next_progress = time.monotonic()
        reached = False
        while time.monotonic() < deadline:
            drain(env, 0.3)
            horizontal, _ = env._distance_and_bearing(
                shooter.lat, shooter.lon, target.lat, target.lon
            )
            slant = math.sqrt(horizontal ** 2 + (shooter.alt - target.alt) ** 2)
            waypoint_error, _ = env._distance_and_bearing(
                shooter.lat, shooter.lon, launch_lat, launch_lon
            )
            if time.monotonic() >= next_progress:
                print(json.dumps({
                    "phase": "approach_progress",
                    "slant_range_m": round(slant, 1),
                    "waypoint_error_m": round(waypoint_error, 1),
                    "altitude_m": round(shooter.alt, 1),
                    "speed_mps": round(shooter.speed, 1),
                    "task": shooter.task,
                    "task_status": shooter.task_status,
                }), flush=True)
                next_progress = time.monotonic() + 5.0
            # Fire upon entering a narrow band just inside the 45 km native
            # slant-range limit; exact waypoint convergence is not required.
            if shooter.alt >= 2990.0 and 44000.0 <= slant <= 44980.0:
                reached = True
                break
        if not reached:
            raise RuntimeError(
                "aircraft did not enter the 44.0-44.98 km launch band; "
                "last slant={0:.1f} waypoint_error={1:.1f} task={2} status={3}".format(
                    slant, waypoint_error, shooter.task, shooter.task_status
                )
            )


        horizontal, _ = env._distance_and_bearing(
            shooter.lat, shooter.lon, target.lat, target.lon
        )
        launch_slant = math.sqrt(horizontal ** 2 + (shooter.alt - target.alt) ** 2)
        fire = {
            "MsgType": "AssignTask",
            "PlatformId": shooter.platform_id,
            "PlatformName": shooter.name,
            "Task": "FIRE_AGM",
            "TargetName": target.name,
            "TargetPosition": [target.lat, target.lon, target.alt],
            "Weapon": "agm",
        }
        for _ in range(3):
            env._send(fire)
            drain(env, 0.5)
            if env.attack_ammo.get(shooter.name, {}).get("agm", 1) == 0:
                break
        fired = env.attack_ammo.get(shooter.name, {}).get("agm", 1) == 0
        print(json.dumps({
            "phase": "fire_command",
            "horizontal_range_m": round(horizontal, 1),
            "slant_range_m": round(launch_slant, 1),
            "altitude_m": round(shooter.alt, 1),
            "initial_hp": initial_hp,
            "agm_remaining": env.attack_ammo.get(shooter.name, {}).get("agm"),
            "shooter_task": shooter.task,
            "shooter_task_status": shooter.task_status,
            "fired_confirmed": fired,
        }), flush=True)
        if not fired:
            raise RuntimeError("FIRE_AGM was sent but native ammo did not decrease")

        launch_sim_time = env._current_sim_time()
        deadline = time.monotonic() + 45.0
        while time.monotonic() < deadline:
            drain(env, 0.2)
            if float(target.current_hp) < initial_hp or not target.alive:
                break
            if env._current_sim_time() - launch_sim_time > 180.0:
                break
        result = {
            "phase": "result",
            "launch_slant_m": round(launch_slant, 1),
            "elapsed_sim_seconds": round(env._current_sim_time() - launch_sim_time, 1),
            "initial_hp": initial_hp,
            "final_hp": float(target.current_hp),
            "target_alive": bool(target.alive),
            "hit": float(target.current_hp) < initial_hp or not target.alive,
        }
        print(json.dumps(result), flush=True)
        if not result["hit"]:
            raise RuntimeError("AGM launched but no target damage/hit was observed")
        trace = TRACE_PATH.read_text(encoding="utf-8")
        fired_marker = "WEAPON_FIRED" in trace and SHOOTER_NAME in trace
        hit_marker = "WEAPON_HIT" in trace and "physical_contact" in trace
        print(json.dumps({"phase": "log_markers", "weapon_fired": fired_marker, "weapon_hit": hit_marker, "trace_path": str(TRACE_PATH)}), flush=True)
        if not fired_marker or not hit_marker:
            raise RuntimeError("target HP changed but required WEAPON_FIRED/WEAPON_HIT log markers were missing")
        if "--exit-after-hit" not in sys.argv:
            print(json.dumps({
                "phase": "keep_alive",
                "message": "scenario and Warlock remain running; press Ctrl+C to stop",
                "trace_path": str(TRACE_PATH),
            }), flush=True)
            try:
                while env.warlock_process is None or env.warlock_process.poll() is None:
                    drain(env, 1.0)
            except KeyboardInterrupt:
                print(json.dumps({"phase": "stopping", "reason": "keyboard_interrupt"}), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()