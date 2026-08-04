"""Live Warlock demonstration of the attack-aircraft return action lock."""

import json
import time

from envs.afsim_env import AFSIMIslandEnv


def drain_for(env, wall_seconds):
    deadline = time.monotonic() + wall_seconds
    while time.monotonic() < deadline:
        env._drain_messages(timeout=0.1)


def distance_to_carrier(env, platform):
    carrier = env._first_platform("red", "carrier")
    return env._distance_and_bearing(platform.lat, platform.lon, carrier.lat, carrier.lon)[0]


def main():
    env = AFSIMIslandEnv(bind=True, auto_start_warlock=True)
    sent = []
    real_send = env._send

    def recording_send(message):
        sent.append(dict(message))
        return real_send(message)

    env._send = recording_send
    try:
        required = ["red_attack_1", "red_attack_2", "red_attack_3", "red_carrier"]
        if not env.wait_for_platforms(required, timeout=45):
            raise RuntimeError("required live platforms were not reported")
        drain_for(env, 2.0)
        teams = env.initialize_bottom_teams()
        group = env.attack_controller.active_groups[teams["attack"][0]]
        leader = env.attack_controller.ensure_group_leader(group)
        state = env.get_attack_task_state(group.group_id)
        action_table = state["action_table"]
        return_id = next(row["id"] for row in action_table if row["name"] == "RETURN_HOME")

        candidates = []
        for row in action_table:
            if not row["name"].startswith("ATTACK_TARGET_"):
                continue
            slot = int(row.get("target_slot", -1))
            targets = env._attack_action_target_slots()
            if not (0 <= slot < len(targets)):
                continue
            target = targets[slot]
            platform = env.platforms.get(target.get("name"))
            if platform is None or not platform.alive or platform.role != "attack_aircraft":
                continue
            distance, _ = env._slant_distance_and_bearing(
                leader.lat, leader.lon, leader.alt,
                platform.lat, platform.lon, platform.alt,
            )
            candidates.append((distance, row["id"], platform.name))
        if not candidates:
            raise RuntimeError("no live blue air target is available")
        _, attack_id, target_name = min(candidates)
        print(json.dumps({"phase": "selected_target", "leader": leader.name,
                          "target": target_name, "attack_action": attack_id}, ensure_ascii=False), flush=True)

        fired = False
        deadline = time.monotonic() + 150.0
        while time.monotonic() < deadline and not fired:
            before = len(sent)
            state = env.get_attack_task_state(group.group_id)
            mask = state["aircraft"][leader.name]["action_mask"]
            if attack_id < len(mask) and mask[attack_id] > 0.0:
                env.apply_attack_aircraft_action(group.group_id, leader.name, attack_id)
            new_messages = sent[before:]
            for message in new_messages:
                if message.get("Task") == "FIRE_AAM":
                    fired = True
                    print(json.dumps({"phase": "fired", "message": message}, ensure_ascii=False), flush=True)
                    break
            drain_for(env, 1.0)
        if not fired:
            raise RuntimeError("live aircraft did not reach the AAM firing phase before timeout")

        # Wait until the short fire-command window closes/acknowledges.
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline and env._active_pending_attack_fire(leader.name):
            drain_for(env, 0.2)

        state = env.get_attack_task_state(group.group_id)
        if state["aircraft"][leader.name]["action_mask"][return_id] <= 0.0:
            raise RuntimeError("RETURN_HOME did not unlock after the live fire attempt")
        self_before = len(sent)
        if not env.apply_attack_aircraft_action(group.group_id, leader.name, return_id):
            raise RuntimeError("live RETURN_HOME command was rejected")
        retreat_messages = [m for m in sent[self_before:] if m.get("Task") == "RETREAT"]
        print(json.dumps({"phase": "return_started", "retreat_messages": retreat_messages}, ensure_ascii=False), flush=True)

        state = env.get_attack_task_state(group.group_id)
        masks = {}
        rejected = {}
        message_count = len(sent)
        for member in group.platforms:
            mask = state["aircraft"][member.name]["action_mask"]
            enabled = [index for index, value in enumerate(mask) if value > 0.0]
            masks[member.name] = enabled
            rejected[member.name] = []
            for action_id in sorted(env.attack_controller.action_specs):
                if action_id == return_id:
                    continue
                if not env.apply_attack_aircraft_action(group.group_id, member.name, action_id):
                    rejected[member.name].append(action_id)
            if not env.apply_attack_aircraft_action(group.group_id, member.name, return_id):
                raise RuntimeError("continue-return action failed for " + member.name)
        print(json.dumps({"phase": "lock_verified", "return_id": return_id,
                          "enabled_actions": masks, "rejected_actions": rejected,
                          "extra_commands_sent": len(sent) - message_count}, ensure_ascii=False), flush=True)

        before_distance = distance_to_carrier(env, leader)
        drain_for(env, 5.0)
        after_distance = distance_to_carrier(env, leader)
        print(json.dumps({"phase": "return_progress", "leader": leader.name,
                          "task": leader.task, "task_status": leader.task_status,
                          "distance_before_m": round(before_distance, 1),
                          "distance_after_m": round(after_distance, 1),
                          "distance_decreased": after_distance < before_distance}, ensure_ascii=False), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()