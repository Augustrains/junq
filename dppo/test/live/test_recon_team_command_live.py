"""Interactive live driver for reconnaissance team 1.

Enter one complete continuous-action vector per decision:
``1 east1 north1 altitude1 2 east2 north2 altitude2 3 east3 north3 altitude3``.
For example, ``1 1 0 0 2 0 0 0 3 0 0 0`` sends the leader east and keeps the
other two entries neutral.  The leader's command moves the full formation;
follower vectors are accepted for a fixed interface but do not redirect it.
"""

import argparse
import json
import queue
import threading
import time
from pathlib import Path

from envs.afsim_env import AFSIMIslandEnv


MEMBERS = ("red_recon_1", "red_recon_2", "red_recon_3")


def parse_vector(text):
    tokens = text.split()
    if len(tokens) != 12:
        raise ValueError(
            "enter 12 values: 1 east north alt 2 east north alt 3 east north alt"
        )
    result = {}
    for index in range(0, len(tokens), 4):
        try:
            aircraft_number = int(tokens[index])
            action = [float(value) for value in tokens[index + 1:index + 4]]
        except ValueError as error:
            raise ValueError("aircraft numbers are integers and action values are numbers") from error
        if aircraft_number not in (1, 2, 3):
            raise ValueError("aircraft number must be 1, 2, or 3")
        if aircraft_number in result:
            raise ValueError("each aircraft may appear only once")
        if any(value < -1.0 or value > 1.0 for value in action):
            raise ValueError("each continuous action value must be in [-1, 1]")
        result[aircraft_number] = action
    if set(result) != {1, 2, 3}:
        raise ValueError("the vector must include aircraft 1, 2, and 3")
    return [(MEMBERS[number - 1], result[number]) for number in (1, 2, 3)]

def state_summary(env, group):
    state = env.get_recon_task_state(group.group_id)
    aircraft = {}
    for name in MEMBERS:
        platform = env.platforms[name]
        fields = state["aircraft"].get(name, {}).get("obs_by_name", {})
        aircraft[name] = {
            "alive": platform.alive,
            "position": [round(platform.lat, 5), round(platform.lon, 5), round(platform.alt)],
            "task": platform.task,
            "task_status": platform.task_status,
            "is_leader": bool(fields.get("is_leader", 0.0)),
            "distance_to_leader_m": round(
                float(fields.get("distance_to_leader_norm", 0.0))
                * float(env.recon_state_config.get("normalization", {}).get("max_formation_spacing_m", 15000.0)),
                1,
            ),
        }
    return {
        "sim_time": round(env._current_sim_time(), 2),
        "leader": state["team"]["leader_name"],
        "members": state["team"]["members"],
        "formation_spacing_by_aircraft": state["team"]["formation_spacing_by_aircraft"],
        "aircraft": aircraft,
    }


def input_worker(input_queue):
    while True:
        try:
            raw = input("actions (1 east north alt 2 east north alt 3 east north alt)> ").strip()
        except (EOFError, KeyboardInterrupt):
            input_queue.put("q")
            return
        input_queue.put(raw)
        if raw.lower() in {"q", "quit", "exit"}:
            return


def apply_vector(env, group, commands):
    leader = env.recon_controller.ensure_group_leader(group)
    commands.sort(key=lambda item: item[0] != leader.name)
    results = []
    for aircraft_name, action in commands:
        sent = env.apply_recon_aircraft_continuous_action(
            group.group_id, aircraft_name, action
        )
        results.append({"aircraft": aircraft_name, "action": action, "sent": sent})
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
            raise RuntimeError("red_recon_1/2/3 were not reported by AFSIM")
        teams = env.initialize_bottom_teams()
        recon_group_ids = teams.get("recon", [])
        group = env.recon_controller.active_groups.get("recon_team_1")
        if group is None and recon_group_ids:
            group = env.recon_controller.active_groups.get(recon_group_ids[0])
        if group is None:
            raise RuntimeError("could not initialize fixed reconnaissance team 1")

        decision_seconds = float(env.config["scenario"].get("decision_seconds", 1.0))
        print(
            "Team ready; decision_seconds={0}. Each value is in [-1, 1]. Type q to stop."
            .format(decision_seconds)
        )
        print("Leader vector controls the formation. Follower vectors are follow acknowledgements.")

        input_queue = queue.Queue()
        threading.Thread(target=input_worker, args=(input_queue,), daemon=True).start()
        decision_index = 0
        next_decision_at = time.monotonic()
        pending_raw = None
        print(json.dumps({"before_decision": decision_index, "state": state_summary(env, group)}, ensure_ascii=False))

        while True:
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
