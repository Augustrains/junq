"""Shared helpers for local high-level -> bottom-agent task diagnostics."""

import argparse
import json
from typing import Dict, Iterable, List, Optional, Tuple

from envs.afsim_env import AFSIMIslandEnv
from envs.rl_interface import AFSIMRLInterface


def parse_address(value: str) -> Tuple[str, int]:
    host, port = value.rsplit(":", 1)
    return host, int(port)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--local-address", default="0.0.0.0:50050")
    parser.add_argument("--platform-timeout", type=float, default=45.0)
    parser.add_argument("--decision-seconds", type=float, default=2.0)
    parser.add_argument("--bind", action="store_true", help="Bind UDP and interact with live Warlock.")
    parser.add_argument("--no-warlock", action="store_true", help="Do not auto-start Warlock on this machine.")
    parser.add_argument("--no-advance", action="store_true", help="Do not drain UDP after bottom action.")
    parser.add_argument("--offline-ready", action="store_true", help="Assign dummy platform ids and local state for offline diagnosis.")


def print_json(prefix: str, value) -> None:
    print(prefix, json.dumps(value, ensure_ascii=False, sort_keys=True))


def make_env(args):
    env = AFSIMIslandEnv(
        bind=bool(args.bind),
        auto_start_warlock=not bool(args.no_warlock),
        local_address=parse_address(args.local_address),
    )
    env.decision_seconds = float(args.decision_seconds)
    sent_messages: List[dict] = []
    original_send = env._send

    def traced_send(msg):
        copied = dict(msg)
        sent_messages.append(copied)
        print_json("UDP_SEND", copied)
        return original_send(msg)

    env._send = traced_send
    return env, sent_messages


def ready_or_offline(env: AFSIMIslandEnv, args, platform_names: Iterable[str]) -> bool:
    names = [name for name in platform_names if name]
    if args.bind:
        ready = env.wait_for_platforms(names, timeout=float(args.platform_timeout))
        print("live_ready", ready)
        if not ready:
            known = {name: p.platform_id for name, p in env.platforms.items() if p.platform_id is not None}
            print_json("known_platforms", known)
        return bool(ready)
    if args.offline_ready:
        for index, name in enumerate(names, start=1):
            platform = env.platforms.get(name)
            if platform is not None:
                if platform.platform_id is None:
                    platform.platform_id = 900000 + index
                platform.alive = True
                platform.task = "PARKED"
                platform.task_status = "IDLE"
                if platform.role in ("recon_aircraft", "attack_aircraft"):
                    platform.at_home = True
    return True


def action_name_by_id(action_table) -> Dict[int, str]:
    names = {}
    for index, action in enumerate(action_table or []):
        action_id = int(action.get("id", index)) if isinstance(action, dict) else index
        names[action_id] = str(action.get("name", action_id)) if isinstance(action, dict) else str(action_id)
    return names


def find_commander_action(api: AFSIMRLInterface, prefix: str, contains: Optional[str] = None):
    state = api.get_agent_state("commander")
    table = state.get("action_table", [])
    mask = state.get("action_mask", [])
    for index, action in enumerate(table):
        name = str(action.get("name", ""))
        if name.startswith(prefix) and (contains is None or contains in name):
            return index, name, float(mask[index]) if index < len(mask) else 0.0
    return None, prefix, 0.0


def attack_area_name_for_target(env, target_name: str, preferred: str = "") -> str:
    """Resolve a known target to a formal high-level attack area."""
    areas = list(getattr(env, "recon_areas", []) or [])
    if preferred:
        preferred_area = next((area for area in areas if str(area.get("name", "")) == preferred), None)
        if preferred_area is not None:
            return preferred
    target = env._attack_target_info(target_name) if target_name else None
    if target:
        for area in areas:
            normalized = env._normalize_recon_area(area)
            if env._attack_target_in_area(target, normalized):
                return str(normalized.get("name", ""))
    return ""


def active_group_ids(api: AFSIMRLInterface, agent_type: str) -> List[str]:
    return list(api.get_active_group_ids(agent_type))


def persistent_summary(api: AFSIMRLInterface, agent_type: str) -> dict:
    state = api.get_persistent_agent_state(agent_type)
    assigned = {}
    for name, agent in state.get("agents", {}).items():
        if bool(agent.get("assigned", False)):
            mask = agent.get("action_mask", [])
            assigned[name] = {
                "group_id": agent.get("group_id"),
                "mask_valid_ids": [i for i, value in enumerate(mask) if float(value) > 0.0],
                "task_context": agent.get("task_context", {}),
            }
    return {"active_group_ids": active_group_ids(api, agent_type), "assigned_agents": assigned}


def choose_actor_and_action(api: AFSIMRLInterface, agent_type: str, group_id: str, preferred_actor: Optional[str], preferred_action_names: Iterable[str]):
    state = api.get_agent_state(agent_type, group_id)
    container = "aircraft" if agent_type in ("recon", "attack") else "ships" if agent_type == "landing" else "units"
    agents = state.get(container, {})
    if not agents:
        raise RuntimeError("{0} group has no assigned entities".format(agent_type))
    actor = preferred_actor if preferred_actor in agents else sorted(agents.keys())[0]
    table = state.get("action_table", [])
    names = action_name_by_id(table)
    mask = agents[actor].get("action_mask", [])
    valid = {names.get(i, str(i)): i for i, value in enumerate(mask) if float(value) > 0.0}
    for action_name in preferred_action_names:
        for action_id, name in names.items():
            if name == action_name and action_id < len(mask) and float(mask[action_id]) > 0.0:
                return state, actor, action_id, name, valid
    raise RuntimeError("none of preferred actions are valid: {0}; valid={1}".format(list(preferred_action_names), valid))


def inject_known_target(env: AFSIMIslandEnv, target_name: str, target_type: Optional[str] = None):
    target = env.platforms.get(target_name)
    if target is None:
        raise RuntimeError("target platform not configured: {0}".format(target_name))
    resolved_type = target_type or target.platform_type or target.role or "UNKNOWN"
    if target_name.startswith("blue_attack") and "AIR" not in str(resolved_type).upper():
        resolved_type = "BLUE_ATTACK_AIRCRAFT"
    env.detected_targets[target_name] = {
        "Name": target_name,
        "Type": resolved_type,
        "Lat": float(target.lat),
        "Lon": float(target.lon),
        "Alt": float(target.alt),
        "Range": 0.0,
        "Bearing": 0.0,
        "TrackSource": "task_pipeline_diagnostics",
        "known": True,
        "alive": True,
    }
    target.detected = True
    return env.detected_targets[target_name]


def step_bottom(api: AFSIMRLInterface, agent_type: str, group_id: str, actor: str, action_id: int, advance: bool):
    return api.step_task_agent(agent_type, group_id, {actor: action_id}, advance_sim=advance)
