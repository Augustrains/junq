"""Interactive transport controller with offline and live AFSIM modes.

Offline geometry/state-machine mode:
    python train/manual_transport_timestep_sim.py

Live AFSIM/Warlock mode:
    python train/manual_transport_timestep_sim.py --live

At every prompt, enter east,north,reserved.  For example, 1,0,0 commands
one maximum-reach step east; status prints the current transport state; q exits.
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.afsim_env import AFSIMIslandEnv


DEFAULT_START = (23.700000, 120.000000)


def parse_action(raw):
    raw = raw.replace("\N{FULLWIDTH COMMA}", ",")
    values = [part.strip() for part in raw.split(",")]
    if len(values) != 3:
        raise ValueError("enter three values, e.g. 1,0,0")
    action = [float(value) for value in values]
    if any(value < -1.0 or value > 1.0 for value in action):
        raise ValueError("each component must be in [-1, 1]")
    return action


def print_status(env, ship):
    status = env.get_island_status(ship.lat, ship.lon)
    cargo = env.landing_cargo[ship.name]
    pending = env.pending_landing_unloads.get(ship.name, {})
    remaining = max(
        0.0, float(pending.get("unload_complete_at", env._current_sim_time()))
        - env._current_sim_time()
    )
    print(
        "STATUS",
        {
            "step": env.step_count,
            "sim_time_s": round(env._current_sim_time(), 1),
            "position": (round(ship.lat, 6), round(ship.lon, 6)),
            "shore_distance_m": round(status["shore_distance_m"], 1),
            "on_land": status["on_land"],
            "task": ship.task,
            "has_army": cargo["has_army"],
            "army_landed": cargo["army_landed"],
            "unload_remaining_s": round(remaining, 1),
        },
    )


def simulate_one_step(env, group_id, ship, action, live=False, live_step_timeout=20.0):
    if live:
        before = (float(ship.lat), float(ship.lon))
        preview, error = env.landing_controller.create_ship_continuous_action_message(
            group_id, ship.name, action
        )
        if error:
            raise RuntimeError(error)
        requested_target = preview["MovePosition"]
        target_lat, target_lon, _ = env._clip_transport_move_to_shore(
            ship.lat, ship.lon, requested_target[0], requested_target[1]
        )
        initial_target_distance, _ = env._distance_and_bearing(
            before[0], before[1], target_lat, target_lon
        )
        if not env.apply_landing_ship_continuous_action(group_id, ship.name, action):
            raise RuntimeError("AFSIM rejected continuous transport action")
        deadline = time.monotonic() + max(1.0, float(live_step_timeout))
        arrival_radius_m = 50.0
        is_hold_step = initial_target_distance <= 1.0
        start_sim_time = float(env._current_sim_time())
        while time.monotonic() < deadline:
            env._drain_messages(timeout=min(0.5, max(0.01, deadline - time.monotonic())))
            if is_hold_step:
                if env._current_sim_time() >= start_sim_time + env.landing_controller.decision_sim_seconds:
                    break
                continue
            remaining_m, _ = env._distance_and_bearing(
                ship.lat, ship.lon, target_lat, target_lon
            )
            if remaining_m <= arrival_radius_m or ship.name in env.pending_landing_unloads:
                break
        else:
            remaining_m, _ = env._distance_and_bearing(ship.lat, ship.lon, target_lat, target_lon)
            raise RuntimeError(
                "AFSIM did not reach this action's waypoint before timeout "
                "(remaining_m={0:.1f}, target=({1:.6f}, {2:.6f}))".format(
                    remaining_m, target_lat, target_lon
                )
            )
        env.step_count += 1
        # MoveUpdate normally invokes these hooks; call them once more so an
        # arrival in the final packet of the decision window is handled now.
        env._maybe_start_automatic_landing(ship)
        env._maybe_confirm_landing_unload(ship)
        moved_m, _ = env._distance_and_bearing(
            before[0], before[1], ship.lat, ship.lon
        )
        print(
            "STEP_RESULT",
            {
                "before": before,
                "after": (float(ship.lat), float(ship.lon)),
                "moved_m": round(moved_m, 1),
                "task": ship.task,
                "speed_mps": round(float(ship.speed), 2),
                "action_target": (round(target_lat, 6), round(target_lon, 6)) if live else None,
            },
        )
        return

    message, error = env.landing_controller.create_ship_continuous_action_message(
        group_id, ship.name, action
    )
    if error:
        raise RuntimeError(error)
    target = message["MovePosition"]
    lat, lon, shoreline = env._clip_transport_move_to_shore(
        ship.lat, ship.lon, target[0], target[1]
    )
    ship.lat, ship.lon, ship.alt = lat, lon, 0.0
    ship.speed = 0.0
    ship.last_update = env._current_sim_time() + env.landing_controller.decision_sim_seconds
    ship.task = "LANDING_MOVE_POINT"
    ship.task_status = "ASSIGNED"
    ship.task_assigned = True
    env.step_count += 1
    env._maybe_start_automatic_landing(ship)
    env._maybe_confirm_landing_unload(ship)
    if shoreline is not None:
        print("SHORE CONTACT: vessel stopped on sea side at", shoreline)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Start Warlock and control the live AFSIM scenario.")
    parser.add_argument("--config-path", default="", help="Optional environment config JSON (for example the near-shore transport scenario).")
    parser.add_argument("--platform-timeout", type=float, default=75.0)
    parser.add_argument("--live-step-timeout", type=float, default=20.0)
    parser.add_argument("--start-lat", type=float, default=DEFAULT_START[0])
    parser.add_argument("--start-lon", type=float, default=DEFAULT_START[1])
    args = parser.parse_args()

    config_path = str(Path(args.config_path).resolve()) if args.config_path else None
    env = AFSIMIslandEnv(config_path=config_path, bind=args.live, auto_start_warlock=args.live)
    try:
        transport_names = list(env.config.get("red", {}).get("commandable_transports", env.config.get("red", {}).get("transports", [])))
        if args.live:
            if not env.wait_for_platforms(transport_names, timeout=args.platform_timeout):
                raise RuntimeError("transport platforms did not register before timeout")
            print("LIVE MODE: Warlock is running; each action waits one AFSIM decision window.")
        else:
            print("OFFLINE MODE: deterministic geometry/state-machine simulation.")

        teams = env.initialize_bottom_teams()
        group_id = teams["landing"][0]
        ship = env.landing_controller.active_groups[group_id].platforms[0]
        if not args.live:
            ship.lat, ship.lon, ship.alt = args.start_lat, args.start_lon, 0.0
            ship.last_update = 0.0

        print("Action format: east,north,reserved. Enter status or q.")
        print_status(env, ship)
        while not env.landing_cargo[ship.name]["army_landed"]:
            raw = input("action > ").strip().lower()
            if raw in ("q", "quit", "exit"):
                break
            if raw in ("status", "s"):
                print_status(env, ship)
                continue
            try:
                simulate_one_step(
                    env, group_id, ship, parse_action(raw), live=args.live,
                    live_step_timeout=args.live_step_timeout,
                )
            except (ValueError, RuntimeError) as error:
                print("INVALID ACTION:", error)
                continue
            print_status(env, ship)
        if env.landing_cargo[ship.name]["army_landed"]:
            print("LANDING COMPLETE:", ship.name)
    finally:
        env.close()


if __name__ == "__main__":
    main()