"""Interactive live driver for attack team 1.

Enter one complete vector per decision, for example ``1 7 2 0 3 0``.
While this program waits for keyboard input, it continues to drain AFSIM
messages so pending approach missions can detect weapon range and stop at the launch boundary.
"""

import argparse
import json
import queue
import threading
import time
from pathlib import Path

from envs.afsim_env import AFSIMIslandEnv


MEMBERS = ("red_attack_1", "red_attack_2", "red_attack_3")


def parse_vector(text):
    tokens = text.split()
    if len(tokens) != 6:
        raise ValueError("enter exactly three pairs, e.g. 1 7 2 0 3 0")
    result = {}
    for index in range(0, len(tokens), 2):
        aircraft_number = int(tokens[index])
        if aircraft_number not in (1, 2, 3):
            raise ValueError("aircraft number must be 1, 2, or 3")
        if aircraft_number in result:
            raise ValueError("each aircraft may appear only once")
        result[aircraft_number] = int(tokens[index + 1])
    if set(result) != {1, 2, 3}:
        raise ValueError("the vector must include aircraft 1, 2, and 3")
    return [(MEMBERS[number - 1], result[number]) for number in (1, 2, 3)]


def state_summary(env, group):
    state = env.get_attack_task_state(group.group_id)
    aircraft = {}
    for name in MEMBERS:
        platform = env.platforms[name]
        mask = state["aircraft"][name]["action_mask"]
        aircraft[name] = {
            "position": [round(platform.lat, 5), round(platform.lon, 5), round(platform.alt)],
            "task": platform.task,
            "task_status": platform.task_status,
            "available_action_ids": [index for index, value in enumerate(mask) if value > 0.0],
        }
    return {
        "sim_time": round(env._current_sim_time(), 2),
        "leader": state["team"]["leader_name"],
        "aircraft": aircraft,
        "pending_approaches": dict(env.pending_attack_approaches),
        "pending_fire": dict(env.pending_attack_fire_commands),
    }


def input_worker(input_queue):
    while True:
        try:
            raw = input("actions (1 7 2 0 3 0)> ").strip()
        except (EOFError, KeyboardInterrupt):
            input_queue.put("q")
            return
        input_queue.put(raw)
        if raw.lower() in {"q", "quit", "exit"}:
            return


def action_diagnostic(env, group, aircraft_name, action_id):
    """Capture the live conditions used immediately before one action is sent."""
    state = env.get_attack_task_state(group.group_id)
    mask = state.get("aircraft", {}).get(aircraft_name, {}).get("action_mask", [])
    action = env.attack_controller.action_specs.get(int(action_id), {})
    result = {
        "allowed_now": bool(0 <= int(action_id) < len(mask) and float(mask[int(action_id)]) > 0.0),
        "action": action.get("name", "UNKNOWN"),
    }
    if str(action.get("afsim_task", "")) != "ATTACK_TARGET_SLOT":
        return result

    slot = int(action.get("target_slot", -1))
    targets = env._attack_action_target_slots()
    if slot < 0 or slot >= len(targets):
        result["target_slot_valid"] = False
        return result

    target = targets[slot]
    aircraft = env.platforms[aircraft_name]
    target_name = str(target.get("name", ""))
    distance_m, _ = env._slant_distance_and_bearing(
        aircraft.lat, aircraft.lon, aircraft.alt,
        float(target.get("lat", aircraft.lat)),
        float(target.get("lon", aircraft.lon)),
        float(target.get("alt", 0.0)),
    )
    weapon = env.attack_controller._compatible_weapon(target)
    result.update({
        "target": target_name,
        "target_alive": bool(target.get("alive", False)),
        "distance_m": round(distance_m, 1),
        "weapon": weapon,
        "weapon_range_m": env._attack_weapon_range(target),
        "in_weapon_range": env._target_in_attack_launch_range(aircraft, target, distance_m),
        "ammo": int(env.attack_ammo.get(aircraft_name, {}).get(weapon, 0)),
        "weapons_expended": sorted(group.weapons_expended.get(aircraft_name, set())),
    })
    return result


def apply_vector(env, group, commands):
    leader = env.attack_controller.ensure_group_leader(group)
    commands.sort(key=lambda item: item[0] != leader.name)
    results = []
    for aircraft_name, action_id in commands:
        diagnostic = action_diagnostic(env, group, aircraft_name, action_id)
        sent = env.apply_attack_aircraft_action(group.group_id, aircraft_name, action_id)
        results.append({
            "aircraft": aircraft_name,
            "action_id": action_id,
            "sent": sent,
            "diagnostic_before": diagnostic,
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="test/configs/afsim_units_action3_aam_test.json")
    args = parser.parse_args()

    env = AFSIMIslandEnv(
        config_path=str(Path(args.config).resolve()), bind=True, auto_start_warlock=True
    )
    try:
        if not env.wait_for_platforms(list(MEMBERS), timeout=30):
            raise RuntimeError("red_attack_1/2/3 were not reported by AFSIM")
        teams = env.initialize_bottom_teams()
        attack_group_ids = teams.get("attack", [])
        group = env.attack_controller.active_groups.get("attack_team_1")
        if group is None and attack_group_ids:
            group = env.attack_controller.active_groups.get(attack_group_ids[0])
        if group is None:
            raise RuntimeError("could not initialize fixed attack team 1")

        action_table = [
            {key: action.get(key) for key in ("id", "name", "target_name")}
            for action in env.get_attack_action_table()
        ]
        print(json.dumps({"action_table": action_table}, ensure_ascii=False))
        decision_seconds = float(env.config["scenario"].get("decision_seconds", 1.0))
        print("Team ready; decision_seconds={0}. Type q to stop.".format(decision_seconds))

        input_queue = queue.Queue()
        threading.Thread(target=input_worker, args=(input_queue,), daemon=True).start()
        decision_index = 0
        next_decision_at = time.monotonic()
        pending_raw = None
        print(json.dumps({"before_decision": decision_index, "state": state_summary(env, group)}, ensure_ascii=False))

        while True:
            # Keep the bridge live while waiting for human input. This makes
            # _check_pending_attack_approaches stop movement promptly at range entry.
            env._drain_messages(timeout=0.05)

            if pending_raw is None:
                try:
                    pending_raw = input_queue.get_nowait()
                except queue.Empty:
                    continue
                if pending_raw.lower() in {"q", "quit", "exit"}:
                    break

            if time.monotonic() < next_decision_at:
                continue
            try:
                commands = parse_vector(pending_raw)
            except ValueError as error:
                print("INPUT_ERROR: {0}".format(error))
                pending_raw = None
                continue

            results = apply_vector(env, group, commands)
            print(json.dumps({"decision": decision_index, "commands": results}, ensure_ascii=False))
            decision_index += 1
            pending_raw = None
            next_decision_at = time.monotonic() + decision_seconds
            print(json.dumps({"after_decision": decision_index - 1, "state": state_summary(env, group)}, ensure_ascii=False))
            print(json.dumps({"before_decision": decision_index, "state": state_summary(env, group)}, ensure_ascii=False))
    finally:
        env.close()


if __name__ == "__main__":
    main()