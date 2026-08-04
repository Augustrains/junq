import argparse
import json
import sys
import time

from envs.afsim_env import AFSIMIslandEnv


def parse_address(value):
    host, port = value.rsplit(":", 1)
    return host, int(port)


def main():
    parser = argparse.ArgumentParser(
        description="Verify that an attack aircraft can engage a locally detected opportunity target."
    )
    parser.add_argument("--local-address", default="0.0.0.0:50050")
    parser.add_argument("--primary-target", default="blue_sam_1")
    parser.add_argument("--aircraft", default="red_attack_1")
    parser.add_argument("--opportunity-target", default="blue_attack_1")
    parser.add_argument("--timeout", type=float, default=45.0)
    args = parser.parse_args()

    env = AFSIMIslandEnv(bind=True, local_address=parse_address(args.local_address))
    try:
        required = [args.aircraft, args.primary_target]
        ready = env.wait_for_platforms(required, timeout=args.timeout)
        print("live_ready", ready, flush=True)
        if not ready:
            print("DIAGNOSIS required_platforms_not_received", required, flush=True)
            return 1

        aircraft = env.platforms[args.aircraft]
        primary_platform = env.platforms[args.primary_target]
        env.detected_targets[args.primary_target] = {
            "Name": args.primary_target,
            "Type": primary_platform.platform_type or primary_platform.role,
            "Lat": primary_platform.lat,
            "Lon": primary_platform.lon,
            "Alt": primary_platform.alt,
            "known": True,
            "alive": primary_platform.alive,
            "TrackSource": "diagnostic_high_level_primary",
        }
        group = env.start_attack_group(
            args.primary_target, group_size=1, preferred_platform=args.aircraft
        )
        if group is None:
            print(
                "DIAGNOSIS attack_group_not_created",
                aircraft.task,
                aircraft.task_status,
                flush=True,
            )
            return 1
        print("attack_group", group.group_id, args.aircraft, args.primary_target, flush=True)

        deadline = time.time() + args.timeout
        selected = None
        last_state = None
        while time.time() < deadline:
            env._drain_messages(timeout=0.5)
            # Isolate the attack-aircraft-only path even if another friendly sensor
            # reports this target during the deterministic live setup.
            if args.opportunity_target in env.attack_local_detections.get(args.aircraft, {}):
                env.detected_targets.pop(args.opportunity_target, None)
            state = env.get_attack_task_state(group.group_id)
            last_state = state
            aircraft_state = state["aircraft"][args.aircraft]
            slots = aircraft_state.get("target_slots", [])
            mask = aircraft_state["action_mask"]
            actions = {item["name"]: item["id"] for item in state["action_table"]}
            for slot_index, slot in enumerate(slots):
                name = slot.get("name", "")
                if (
                    name != args.opportunity_target
                    or name == args.primary_target
                    or not slot.get("local_only", False)
                ):
                    continue
                action_name = "ATTACK_TARGET_{0}".format(slot_index + 1)
                action_id = actions.get(action_name)
                if action_id is not None and float(mask[action_id]) > 0.0:
                    selected = (slot, action_id, action_name)
                    break
            if selected:
                break

        if selected is None:
            aircraft_state = (last_state or {}).get("aircraft", {}).get(args.aircraft, {})
            slots = aircraft_state.get("target_slots", [])
            mask = aircraft_state.get("action_mask", [])
            valid_actions = [
                action.get("name", "")
                for action in (last_state or {}).get("action_table", [])
                if int(action.get("id", -1)) < len(mask)
                and float(mask[int(action.get("id", -1))]) > 0.0
            ]
            slot_summary = [
                {
                    "slot": index,
                    "target": slot.get("name", ""),
                    "local_only": bool(slot.get("local_only", False)),
                    "type": slot.get("type", ""),
                }
                for index, slot in enumerate(slots, start=1)
            ]
            print(
                "local_contact_names",
                sorted(env.attack_local_detections.get(args.aircraft, {}).keys()),
                flush=True,
            )
            print("target_slots", json.dumps(slot_summary, sort_keys=True), flush=True)
            print("valid_actions", valid_actions, flush=True)
            print(
                "aircraft_state",
                json.dumps({
                    "alt": aircraft.alt,
                    "lat": aircraft.lat,
                    "lon": aircraft.lon,
                    "task": aircraft.task,
                    "task_status": aircraft.task_status,
                }, sort_keys=True),
                flush=True,
            )
            print("DIAGNOSIS no_valid_local_opportunity_action", flush=True)
            return 1

        slot, action_id, action_name = selected
        target_name = slot["name"]
        assert target_name not in env.detected_targets, (
            "opportunity target unexpectedly entered global detections", target_name
        )
        print(
            "local_opportunity",
            json.dumps({
                "aircraft": args.aircraft,
                "target": target_name,
                "target_type": slot.get("type", ""),
                "track_source": slot.get("track_source", ""),
                "action": action_name,
                "action_id": action_id,
                "aircraft_alt": aircraft.alt,
            }, sort_keys=True),
            flush=True,
        )
        sent = env.apply_attack_aircraft_action(
            group.group_id, args.aircraft, action_id
        )
        print("action_sent", sent, flush=True)
        if not sent:
            print("DIAGNOSIS local_opportunity_action_rejected", flush=True)
            return 1

        env._drain_messages(timeout=3.0)
        pending = env.pending_attack_fire_commands.get(args.aircraft, {})
        print("pending_fire", json.dumps(pending, sort_keys=True), flush=True)
        print(
            "DIAGNOSIS attack_local_opportunity_flow_completed",
            args.aircraft,
            target_name,
            flush=True,
        )
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    sys.exit(main())
