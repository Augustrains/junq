import argparse
import math
import os
import sys
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from envs.afsim_env import AFSIMIslandEnv
from envs.rl_interface import AFSIMRLInterface


REPRESENTATIVE_PLATFORMS = [
    "red_carrier",
    "red_recon_1",
    "red_recon_2",
    "red_recon_3",
    "red_attack_1",
    "red_transport_1",
    "red_ground_1",
]


KEY_FIELDS = {
    "recon": [
        "alive",
        "agent_id_norm",
        "group_slot_norm",
        "lat_norm",
        "lon_norm",
        "alt_norm",
        "heading_sin",
        "heading_cos",
        "speed_norm",
    ],
    "attack": [
        "lat_norm",
        "lon_norm",
        "alt_norm",
        "hp_norm",
        "aam_count_norm",
        "agm_count_norm",
        "target_slot_1_lat_norm",
        "target_slot_1_lon_norm",
        "target_slot_1_alt_norm",
        "target_slot_1_hp_norm",
        "target_slot_1_aam_ammo_norm",
        "target_slot_1_agm_ammo_norm",
    ],    "landing": [
        "alive",
        "agent_id_norm",
        "group_slot_norm",
        "lat_norm",
        "lon_norm",
        "has_army",
        "army_landed",
        "distance_to_landing_zone_norm",
        "landing_zone_slot_1_exists",
        "landing_zone_slot_2_exists",
        "landing_zone_slot_3_exists",
    ],
    "ground": [
        "alive",
        "agent_id_norm",
        "group_slot_norm",
        "lat_norm",
        "lon_norm",
        "on_ship",
        "landed",
        "distance_to_objective_norm",
        "inside_objective",
        "capture_progress_norm",
    ],
}


def first_valid_action_by_prefix(env, prefix):
    mask = env.get_action_mask()
    for idx, name in enumerate(env.action_names):
        if name.startswith(prefix) and idx < len(mask) and mask[idx] > 0:
            return idx, name
    return 0, "WAIT"


def finite_array(values) -> bool:
    arr = np.asarray(values, dtype=np.float32)
    return bool(np.all(np.isfinite(arr)))


def print_result(ok: bool, label: str, detail: str = ""):
    status = "OK" if ok else "FAIL"
    suffix = (" " + detail) if detail else ""
    print(f"{status} {label}{suffix}")


def validate_flat_state(name: str, state: Mapping[str, object], expected_obs_dim: int, expected_action_dim: int) -> bool:
    ok = True
    fields = list(state.get("fields", []))
    obs = np.asarray(state.get("obs", []), dtype=np.float32)
    obs_by_name = state.get("obs_by_name", {})
    mask = np.asarray(state.get("action_mask", []), dtype=np.float32)

    checks = [
        (len(fields) == expected_obs_dim, f"{name}.fields_len", f"{len(fields)}/{expected_obs_dim}"),
        (len(obs) == expected_obs_dim, f"{name}.obs_len", f"{len(obs)}/{expected_obs_dim}"),
        (set(fields) == set(obs_by_name.keys()), f"{name}.obs_by_name_keys", f"missing={sorted(set(fields)-set(obs_by_name.keys()))[:5]} extra={sorted(set(obs_by_name.keys())-set(fields))[:5]}"),
        (finite_array(obs), f"{name}.obs_finite"),
        (finite_array(list(obs_by_name.values())), f"{name}.obs_by_name_finite"),
        (len(mask) == expected_action_dim, f"{name}.mask_len", f"{len(mask)}/{expected_action_dim}"),
        (finite_array(mask), f"{name}.mask_finite"),
        (float(mask.sum()) > 0.0, f"{name}.mask_has_valid", f"sum={float(mask.sum()):.1f}"),
    ]
    for item in checks:
        passed, label = item[0], item[1]
        detail = item[2] if len(item) > 2 else ""
        print_result(bool(passed), label, detail)
        ok = ok and bool(passed)

    key_values = {field: round(float(obs_by_name.get(field, 0.0)), 4) for field in KEY_FIELDS.get(name.split('.')[0], []) if field in obs_by_name}
    if key_values:
        print(f"SAMPLE {name} {key_values}")
    return ok



def validate_persistent(api: AFSIMRLInterface, specs: Mapping[str, dict], agent_type: str, max_samples: int = 2) -> bool:
    state = api.get_persistent_agent_state(agent_type)
    agents = state.get("agents", {})
    expected_obs_dim = specs[agent_type]["obs_dim"]
    expected_action_dim = specs[agent_type]["action_dim"]
    print_result(len(agents) > 0, f"{agent_type}.agent_count", str(len(agents)))
    ok = len(agents) > 0
    for idx, (name, agent_state) in enumerate(sorted(agents.items())):
        sub_ok = validate_flat_state(f"{agent_type}.{name}", agent_state, expected_obs_dim, expected_action_dim)
        ok = ok and sub_ok
        if idx + 1 >= max_samples:
            break
    return ok


def validate_active_groups(api: AFSIMRLInterface, specs: Mapping[str, dict], agent_type: str, max_samples: int = 1) -> bool:
    ok = True
    group_ids = list(api.get_active_group_ids(agent_type))
    print_result(True, f"{agent_type}.active_group_count", str(len(group_ids)))
    container = "aircraft" if agent_type in ("recon", "attack") else "ships" if agent_type == "landing" else "units"
    for group_id in group_ids[:max_samples]:
        group_state = api.get_agent_state(agent_type, group_id)
        entities = group_state.get(container, {})
        print_result(len(entities) > 0, f"{agent_type}.{group_id}.entity_count", str(len(entities)))
        for name, entity_state in sorted(entities.items())[:max_samples]:
            sub_ok = validate_flat_state(f"{agent_type}.{name}.active", entity_state, specs[agent_type]["obs_dim"], specs[agent_type]["action_dim"])
            ok = ok and sub_ok
    return ok


def set_synthetic_platforms(env: AFSIMIslandEnv):
    idx = 1
    base_positions = {
        "carrier": (25.0, 120.0, 0.0),
        "recon_aircraft": (25.05, 120.05, 9144.0),
        "attack_aircraft": (25.02, 120.08, 9144.0),
        "transport": (24.95, 120.02, 0.0),
        "ground_force": (24.95, 120.02, 0.0),
        "base": (25.2, 121.1, 0.0),
        "sam": (25.16, 120.95, 0.0),
        "radar": (25.12, 120.9, 0.0),
    }
    for p in env.platforms.values():
        p.platform_id = idx
        idx += 1
        p.alive = True
        p.task = "PARKED"
        p.task_status = "IDLE"
        p.at_home = p.side == "red" and p.role in ("recon_aircraft", "attack_aircraft")
        lat, lon, alt = base_positions.get(p.role, (25.1, 120.5, 0.0))
        offset = (idx % 10) * 0.005
        p.lat = lat + offset
        p.lon = lon + offset
        p.alt = alt
        p.heading = 90.0
        p.speed = 0.0
        p.last_update = 0.0

    target_name = env.config.get("blue", {}).get("sams", ["blue_sam_1"])[0]
    first_area = env._normalize_recon_area(env.recon_areas[0]) if env.recon_areas else {
        "lat": 25.16,
        "lon": 120.95,
    }
    target_lat = float(first_area["lat"])
    target_lon = float(first_area["lon"])
    target_platform = env.platforms.get(target_name)
    if target_platform is not None:
        target_platform.lat = target_lat
        target_platform.lon = target_lon
        target_platform.alt = 0.0
    env.detected_targets[target_name] = {
        "Name": target_name,
        "Type": "SAM",
        "Lat": target_lat,
        "Lon": target_lon,
        "Alt": 0.0,
        "lat": target_lat,
        "lon": target_lon,
        "alt": 0.0,
    }

    for name, status in env.ground_status.items():
        status["on_ship"] = False
        status["landed"] = True
        platform = env.platforms.get(name)
        if platform is not None:
            platform.lat = 24.98
            platform.lon = 120.08
            platform.alt = 0.0
            platform.at_home = False


def create_synthetic_active_groups(env: AFSIMIslandEnv):
    if env.recon_areas:
        env.start_recon_group(env.recon_areas[0], fixed_team_index=0)
    target_names = list(env.detected_targets.keys())
    if target_names:
        env.start_attack_group(target_names[0], fixed_team_index=0)
    landing_zones = env.config.get("landing_zones", [])
    if landing_zones:
        env.start_landing_group(landing_zones[0], group_size=1)
    objectives = env.config.get("ground_objectives", [])
    if objectives:
        env.start_ground_group(objectives[0], group_size=3)


def run_live_recon_probe(api: AFSIMRLInterface, env: AFSIMIslandEnv):
    api.ensure_rule_driven_tasks()
    accepted = bool(env.recon_controller.active_groups)
    print_result(
        accepted,
        "live.recon_rule_tasks_active",
        "groups={0}".format(sorted(env.recon_controller.active_groups)),
    )
    state = api.get_persistent_agent_state("recon")
    actions = {
        name: ([0.0, 1.0, 0.0] if agent_state.get("assigned", False) else [0.0, 0.0, 0.0])
        for name, agent_state in state.get("agents", {}).items()
    }
    api.step_persistent_agents("recon", actions, advance_sim=False)
    _reward, _done, bottom_info = api.step_rule_driven()
    next_state = api.get_persistent_agent_state("recon")
    moving = []
    for name, agent_state in next_state.get("agents", {}).items():
        obs = agent_state.get("obs_by_name", {})
        if float(obs.get("speed_norm", 0.0)) > 0.01:
            moving.append(name)
    print_result(
        len(moving) >= 3,
        "live.recon_speed_observed",
        "moving={0}".format(moving[:5]),
    )
    print("LIVE_RECON_EVENTS {0}".format(bottom_info.get("events", [])))
    return accepted and len(moving) >= 3


def parse_args():
    parser = argparse.ArgumentParser(description="Validate rule-driven bottom-agent state fields.")
    parser.add_argument("--bind", action="store_true", help="Use live AFSIM/Warlock UDP state.")
    parser.add_argument("--auto-start-warlock", action="store_true")
    parser.add_argument("--platform-timeout", type=float, default=75.0)
    parser.add_argument("--synthetic-active", action="store_true", help="Create synthetic active groups to validate all active bottom state builders.")
    parser.add_argument("--max-samples", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    env = AFSIMIslandEnv(bind=args.bind, auto_start_warlock=args.auto_start_warlock)
    try:
        if args.bind:
            ready = env.wait_for_platforms(REPRESENTATIVE_PLATFORMS, timeout=args.platform_timeout)
            known = sum(1 for p in env.platforms.values() if p.platform_id is not None)
            print_result(ready, "live.platforms_ready", f"known_count={known}")
            if not ready:
                raise SystemExit(1)
        api = AFSIMRLInterface(env)
        api.reset()
        if args.synthetic_active:
            set_synthetic_platforms(env)
            create_synthetic_active_groups(env)
        specs = api.get_agent_specs()
        overall = True
        for agent_type in ("recon", "attack", "landing", "ground"):
            spec = specs[agent_type]
            print_result(spec["obs_dim"] > 0 and spec["action_dim"] > 0, f"spec.{agent_type}", f"obs_dim={spec['obs_dim']} action_dim={spec['action_dim']}")
            overall = overall and spec["obs_dim"] > 0 and spec["action_dim"] > 0
        for agent_type in ("recon", "attack", "landing", "ground"):
            overall = validate_persistent(api, specs, agent_type, max_samples=args.max_samples) and overall
            overall = validate_active_groups(api, specs, agent_type, max_samples=args.max_samples) and overall
        if args.bind:
            overall = run_live_recon_probe(api, env) and overall
        print_result(overall, "STATE_FIELD_DIAGNOSTIC")
        if not overall:
            raise SystemExit(1)
    finally:
        env.close()


if __name__ == "__main__":
    main()
