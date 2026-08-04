"""Diagnose high-level ATTACK -> bottom attack target -> UDP pipeline."""

import argparse
import math
import sys

from envs.rl_interface import AFSIMRLInterface
from test.diagnostics.task_pipeline_diagnostics import add_common_args, attack_area_name_for_target, choose_actor_and_action, find_commander_action, inject_known_target, make_env, persistent_summary, print_json, ready_or_offline, step_bottom


def attack_slot_diagnostics(env, state, actor, group):
    platform = env.platforms.get(actor)
    if platform is None:
        return {"error": "actor_not_configured", "actor": actor}
    slots = state.get("task", {}).get("target_slots", [])
    ammo = env.attack_ammo.get(actor, {"fox3": 1, "agm": 1})
    max_weapon_range = float(env.attack_state_config.get("normalization", {}).get("max_weapon_range_m", 60000.0))
    out = []
    for index, target in enumerate(slots, start=1):
        target_name = target.get("name", "")
        target_lat = float(target.get("lat", platform.lat))
        target_lon = float(target.get("lon", platform.lon))
        target_alt = float(target.get("alt", 0.0))
        horizontal_distance, bearing = env._distance_and_bearing(platform.lat, platform.lon, target_lat, target_lon)
        distance, _ = env._slant_distance_and_bearing(
            platform.lat, platform.lon, platform.alt, target_lat, target_lon, target_alt
        )
        weapon = env.attack_controller._compatible_weapon(target)
        target_weapon_range = env._attack_weapon_range(target, max_weapon_range)
        reserved_by = env._attack_target_reserved_by(group, target_name)
        known = bool(target.get("known", False))
        alive = bool(target.get("alive", True))
        has_ammo = int(ammo.get(weapon, 0)) > 0
        in_range = distance <= target_weapon_range
        available = known and alive and has_ammo and in_range and (not reserved_by or reserved_by == actor)
        reasons = []
        if not known:
            reasons.append("not_known")
        if not alive:
            reasons.append("not_alive")
        if not has_ammo:
            reasons.append("no_compatible_ammo")
        if not in_range:
            reasons.append("out_of_weapon_range")
        if reserved_by and reserved_by != actor:
            reasons.append("reserved_by_other")
        out.append({
            "slot": index,
            "target": target_name,
            "type": target.get("type", ""),
            "distance_m": round(float(distance), 1),
            "horizontal_distance_m": round(float(horizontal_distance), 1),
            "altitude_delta_m": round(float(target_alt - platform.alt), 1),
            "bearing_deg": round(float(bearing), 2),
            "max_weapon_range_m": target_weapon_range,
            "weapon": weapon,
            "ammo": int(ammo.get(weapon, 0)),
            "reserved_by": reserved_by or "",
            "available": available,
            "reasons": reasons,
        })
    return out


def _action_names(state):
    names = {}
    for index, action in enumerate(state.get("action_table", [])):
        action_id = int(action.get("id", index))
        names[action_id] = str(action.get("name", action_id))
    return names


def _valid_actions_for_actor(state, actor):
    aircraft = state.get("aircraft", {})
    mask = aircraft.get(actor, {}).get("action_mask", [])
    names = _action_names(state)
    return {names.get(i, str(i)): i for i, value in enumerate(mask) if float(value) > 0.0}


def _move_action_toward(env, state, actor, target_name):
    platform = env.platforms.get(actor)
    target = env.detected_targets.get(target_name, {})
    if platform is None or not target:
        return "HOLD"
    target_lat = float(target.get("Lat", target.get("lat", platform.lat)))
    target_lon = float(target.get("Lon", target.get("lon", platform.lon)))
    north_m, east_m = env._relative_north_east(platform.lat, platform.lon, target_lat, target_lon)
    angle = math.degrees(math.atan2(east_m, north_m))
    if -22.5 <= angle < 22.5:
        candidates = ["MOVE_NORTH"]
    elif 22.5 <= angle < 67.5:
        candidates = ["MOVE_NORTH_EAST", "MOVE_EAST", "MOVE_NORTH"]
    elif 67.5 <= angle < 112.5:
        candidates = ["MOVE_EAST"]
    elif 112.5 <= angle < 157.5:
        candidates = ["MOVE_SOUTH_EAST", "MOVE_EAST", "MOVE_SOUTH"]
    elif angle >= 157.5 or angle < -157.5:
        candidates = ["MOVE_SOUTH"]
    elif -157.5 <= angle < -112.5:
        candidates = ["MOVE_SOUTH_WEST", "MOVE_WEST", "MOVE_SOUTH"]
    elif -112.5 <= angle < -67.5:
        candidates = ["MOVE_WEST"]
    else:
        candidates = ["MOVE_NORTH_WEST", "MOVE_WEST", "MOVE_NORTH"]
    valid = _valid_actions_for_actor(state, actor)
    for name in candidates:
        if name in valid:
            return name
    return "HOLD"


def _try_choose_action(api, agent_type, group_id, actor, action_name):
    state = api.get_agent_state(agent_type, group_id)
    valid = _valid_actions_for_actor(state, actor)
    if action_name not in valid:
        return state, None, action_name, valid
    return state, valid[action_name], action_name, valid


def main(argv=None):
    parser = argparse.ArgumentParser(description="Diagnose ATTACK high-level to bottom-agent execution.")
    add_common_args(parser)
    parser.add_argument("--attacker", default="red_attack_1")
    parser.add_argument("--target", default="blue_attack_1")
    parser.add_argument("--attack-area", default="", help="Formal high-level attack area; inferred from --target when omitted.")
    parser.add_argument("--bottom-action", default="ATTACK_TARGET_1")
    parser.add_argument("--opportunity-target", default="", help="Optional extra detected target B, used to test attacking a target other than the commander-assigned target A.")
    parser.add_argument("--expect-target", default="", help="Optional expected TargetName in the final fire UDP message.")
    parser.add_argument("--auto-approach-steps", type=int, default=0, help="If the fire action is out of range, move toward the expected target for this many tactical steps before trying to fire again.")
    parser.add_argument("--post-fire-seconds", type=float, default=5.0, help="Wall-clock seconds to keep receiving combat updates after firing.")
    args = parser.parse_args(argv)

    env, sent = make_env(args)
    try:
        # ATTACK is configured as a three-aircraft package.  Prepare the full
        # package so the commander mask tests the real task contract rather
        # than rejecting a deliberately incomplete offline fixture.
        attack_group_names = [name for name in env.platforms if getattr(env.platforms[name], 'role', '') == 'attack_aircraft'][:3]
        if args.attacker in attack_group_names:
            attack_group_names.remove(args.attacker)
        attack_group_names.insert(0, args.attacker)
        platform_names = attack_group_names + [args.target]
        if args.opportunity_target:
            platform_names.append(args.opportunity_target)
        if not ready_or_offline(env, args, platform_names):
            return 2
        api = AFSIMRLInterface(env)
        target_info = inject_known_target(env, args.target)
        print_json("injected_target", target_info)
        if args.opportunity_target:
            opportunity_info = inject_known_target(env, args.opportunity_target)
            print_json("injected_opportunity_target", opportunity_info)
        attack_area = attack_area_name_for_target(env, args.target, args.attack_area)
        if not attack_area:
            print_json("attack_area_resolution", {
                "target": args.target,
                "requested_area": args.attack_area,
                "available_areas": [area.get("name", "") for area in env.recon_areas],
            })
            print("DIAGNOSIS attack_target_outside_configured_areas")
            return 3
        action_id, action_name, valid = find_commander_action(api, "ATTACK:", attack_area + ":")
        print("attack_area", attack_area)
        print("commander_attack_action", action_id, action_name, "mask", valid)
        print_json("before_attack_state", persistent_summary(api, "attack"))
        if action_id is None or valid <= 0.0:
            print("DIAGNOSIS attack_commander_action_not_available")
            return 3
        _state, reward, done, info = api.step_commander(action_id)
        print("commander_step", "reward", round(float(reward), 4), "done", bool(done))
        print_json("commander_info", info)
        after = persistent_summary(api, "attack")
        print_json("after_attack_state", after)
        group_ids = after["active_group_ids"]
        if not group_ids:
            print("DIAGNOSIS attack_group_not_created")
            return 4
        group = getattr(env.attack_controller, "active_groups", {}).get(group_ids[0])
        preview_state = api.get_agent_state("attack", group_ids[0])
        preview_actor = args.attacker if args.attacker in preview_state.get("aircraft", {}) else sorted(preview_state.get("aircraft", {}).keys())[0]
        print_json("attack_slot_diagnostics", attack_slot_diagnostics(env, preview_state, preview_actor, group))
        actor = preview_actor
        target_for_approach = args.expect_target or (args.opportunity_target if args.bottom_action == "ATTACK_TARGET_2" and args.opportunity_target else args.target)
        state, bottom_id, bottom_name, valid_actions = _try_choose_action(api, "attack", group_ids[0], actor, args.bottom_action)
        approach_step = 0
        while bottom_id is None and approach_step < int(args.auto_approach_steps):
            move_name = _move_action_toward(env, state, actor, target_for_approach)
            _move_state, move_id, _move_name, move_valid = _try_choose_action(api, "attack", group_ids[0], actor, move_name)
            if move_id is None:
                print_json("approach_move_unavailable", {"requested": move_name, "valid": move_valid})
                break
            _move_next, move_reward, move_done, move_info = step_bottom(api, "attack", group_ids[0], actor, move_id, True)
            approach_step += 1
            if args.target:
                inject_known_target(env, args.target)
            if args.opportunity_target:
                inject_known_target(env, args.opportunity_target)
            state = api.get_agent_state("attack", group_ids[0])
            print_json("approach_step", {
                "step": approach_step,
                "move_action": move_name,
                "reward": round(float(move_reward), 4),
                "done": bool(move_done),
                "diagnostics": attack_slot_diagnostics(env, state, actor, group),
                "last_udp_send": sent[-1] if sent else {},
                "events": move_info.get("events", []),
            })
            state, bottom_id, bottom_name, valid_actions = _try_choose_action(api, "attack", group_ids[0], actor, args.bottom_action)
        if bottom_id is None:
            print_json("bottom_valid_actions", valid_actions)
            print("DIAGNOSIS attack_action_not_yet_valid", args.bottom_action)
            return 7
        print("bottom_actor", actor)
        print_json("attack_target_slots", state.get("task", {}).get("target_slots", []))
        print_json("bottom_valid_actions", valid_actions)
        _next, bottom_reward, bottom_done, bottom_info = step_bottom(api, "attack", group_ids[0], actor, bottom_id, not args.no_advance)
        print("forced_bottom_action", actor, bottom_id, bottom_name, "reward", round(float(bottom_reward), 4), "done", bool(bottom_done))
        print_json("bottom_info_actions", bottom_info.get("actions", {}))
        print_json("bottom_info_events", bottom_info.get("events", []))
        if sent and sent[-1].get("Task") in ("FIRE_AAM", "FIRE_AGM"):
            env._drain_messages(timeout=max(0.0, float(args.post_fire_seconds)))
        observed_target = env.platforms.get(args.expect_target or target_for_approach)
        if observed_target is not None:
            print_json("post_fire_target_state", {
                "name": observed_target.name,
                "platform_id": observed_target.platform_id,
                "current_hp": observed_target.current_hp,
                "max_hp": observed_target.max_hp,
                "alive": observed_target.alive,
                "last_damage_time": observed_target.last_damage_time,
            })
        print_json("post_fire_reward_events", env.last_reward_events)
        last_send = sent[-1] if sent else {}
        print_json("last_udp_send", last_send)
        expected_target = args.expect_target
        if not expected_target and args.bottom_action == "ATTACK_TARGET_1":
            expected_target = args.target
        if not expected_target and args.bottom_action == "ATTACK_TARGET_2" and args.opportunity_target:
            expected_target = args.opportunity_target
        if expected_target:
            if last_send.get("TargetName") == expected_target:
                print("DIAGNOSIS expected_attack_target_sent", expected_target)
            else:
                print("DIAGNOSIS expected_attack_target_not_sent", expected_target)
                return 6
        if sent and last_send.get("Task") in ("FIRE_AAM", "FIRE_AGM"):
            print("DIAGNOSIS attack_bottom_fire_message_sent")
            return 0
        print("DIAGNOSIS attack_bottom_fire_message_not_sent")
        return 5
    finally:
        env.close()


if __name__ == "__main__":
    sys.exit(main())

