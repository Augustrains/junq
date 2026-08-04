"""Print and verify the tensors actually received by every policy network.

The default probe is completely offline.  It builds deterministic synthetic
platform state, captures the input at each network boundary, and then changes
one enemy entity that is intentionally hidden from all red actors.  Actor
inputs must remain unchanged while centralized critic inputs and values must
respond to the omniscient entity fields.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, Mapping

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from envs.afsim_env import AFSIMIslandEnv
from envs.rl_interface import AFSIMRLInterface
from train.happo_trainer import HAPPOConfig, HAPPOTrainer
from test.tools.state_field_diagnostic import create_synthetic_active_groups, set_synthetic_platforms


REPRESENTATIVES = {
    "recon": "red_recon_1",
    "attack": "red_attack_1",
    "landing": "red_transport_1",
    "ground": "red_ground_1",
}
LIVE_REQUIRED_PLATFORMS = tuple(REPRESENTATIVES.values())


RAW_FEATURE_EXPRESSIONS = {
    "registered": ("platform_id",),
    "alive": ("alive_bool",),
    "indestructible": ("indestructible_bool",),
    "side_id": ("side_label",),
    "role_id": ("role_label",),
    "east_norm": ("lon_deg", "east_m", "east_scale_m"),
    "north_norm": ("lat_deg", "north_m", "north_scale_m"),
    "alt_norm": ("alt_m",),
    "velocity_east_norm": ("velocity_east_mps",),
    "velocity_north_norm": ("velocity_north_mps",),
    "velocity_up_norm": ("velocity_up_mps",),
    "hp_norm": ("current_hp", "max_hp"),
    "fire_cooldown_remaining_norm": (
        "sim_time_sec", "fire_cooldown_until_sec", "fire_cooldown_remaining_sec"
    ),
    "combat_lock_remaining_norm": (
        "sim_time_sec", "combat_lock_until_sec", "combat_lock_remaining_sec"
    ),
    "aam_ammo_norm": ("aam_rounds",),
    "agm_ammo_norm": ("agm_rounds",),
    "ground_ammo_norm": ("ground_rounds",),
    "aam_reserve_norm": (
        "aam_reserve_rounds", "initial_aam_reserve_rounds"
    ),
    "agm_reserve_norm": (
        "agm_reserve_rounds", "initial_agm_reserve_rounds"
    ),
    "task_id": ("task_label", "task_status_label"),
    "at_home": ("at_home_bool", "at_home_applicable_bool"),
    "on_ship": ("on_ship_bool",),
    "landed": ("landed_bool",),
    "has_army": ("has_army_bool",),
    "army_landed": ("army_landed_bool",),
    "ammo_reserve_unlimited": ("ammo_reserve_unlimited_bool",),
    "rearm_remaining_norm": (
        "sim_time_sec", "rearm_complete_at_sec", "rearm_remaining_sec"
    ),
}


def _flat(tensor) -> np.ndarray:
    if hasattr(tensor, "detach"):
        tensor = tensor.detach().cpu().numpy()
    return np.asarray(tensor).reshape(-1).astype(np.float32, copy=False)


def _summary(label: str, values: np.ndarray):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    print(
        "{0} shape={1} finite={2} nonzero={3} min={4:.6f} max={5:.6f} mean={6:.6f}".format(
            label,
            tuple(values.shape),
            bool(np.isfinite(values).all()),
            int(np.count_nonzero(np.abs(values) > 1.0e-9)),
            float(values.min()) if values.size else 0.0,
            float(values.max()) if values.size else 0.0,
            float(values.mean()) if values.size else 0.0,
        )
    )


def _raw_value(value):
    if isinstance(value, float):
        return "{0:.6f}".format(value)
    return repr(value)


def _print_raw_entity_traces(env):
    traces = env.get_critic_entity_state_trace()
    representatives = []
    seen = set()
    for platform in env.platforms.values():
        key = (platform.side, platform.role)
        if key in seen:
            continue
        seen.add(key)
        representatives.append(platform.name)

    print("\n=== RAW PHYSICAL/CACHE VALUES -> 27-FIELD ENTITY STATE ===")
    print(
        "RAW_SOURCE_NOTE motion fields are decoded UDP physical values; "
        "ammo/task/landing values use the named authoritative caches."
    )
    for entity_name in representatives:
        trace = traces[entity_name]
        raw = trace["raw"]
        state = trace["state"]
        print(
            "[RAW_ENTITY] name={0} side={1} role={2} platform_id={3}".format(
                entity_name,
                raw["side_label"],
                raw["role_label"],
                raw["platform_id"],
            )
        )
        print(
            "  UDP_MOTION lat_deg={0:.8f} lon_deg={1:.8f} alt_m={2:.3f} "
            "heading_deg={3:.3f} speed_mps={4:.3f} last_update_sim_time_sec={5:.3f}".format(
                raw["lat_deg"], raw["lon_deg"], raw["alt_m"],
                raw["heading_deg"], raw["speed_mps"],
                raw["last_update_sim_time_sec"],
            )
        )
        for feature in env.CRITIC_ENTITY_FEATURES:
            raw_names = RAW_FEATURE_EXPRESSIONS[feature]
            raw_text = ", ".join(
                "{0}={1}".format(name, _raw_value(raw[name]))
                for name in raw_names
            )
            print(
                "  RAW_TO_STATE {0}: {1} -> state={2:.6f}".format(
                    feature, raw_text, float(state[feature])
                )
            )


def _print_named(
    label: str,
    fields: Iterable[str],
    values: np.ndarray,
    full: bool,
    prefixes: tuple[str, ...] = (),
):
    fields = list(fields)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    assert len(fields) == values.size, (label, len(fields), values.size)
    if full or not prefixes:
        selected = range(len(fields))
    else:
        selected = [
            index for index, field in enumerate(fields)
            if field.startswith(prefixes)
        ]
    print("{0}_FIELDS shown={1}/{2}".format(label, len(selected), len(fields)))
    for index in selected:
        print("  {0:04d} {1}={2:.6f}".format(index, fields[index], float(values[index])))


def _run_all_networks(api, specs, bottom, full: bool):
    global_packet = api.get_critic_global_state()
    global_state = np.asarray(global_packet["obs"], dtype=np.float32)
    global_fields = list(global_packet["fields"])

    outputs = {}
    for agent_type, entity_name in REPRESENTATIVES.items():
        persistent = api.get_persistent_agent_state(agent_type)
        agent_state = persistent["agents"][entity_name]
        decision = bottom.act_bottom(
            agent_type,
            entity_name,
            agent_state,
            global_state=global_state,
        )
        actor_input = _flat(agent_state["obs"])
        critic_input = np.concatenate((global_state, actor_input), axis=0)
        critic_fields = global_fields + [
            "local.{0}".format(field) for field in specs[agent_type]["fields"]
        ]
        label = agent_type.upper()
        _summary("[{0}_ACTOR]".format(label), actor_input)
        _print_named(
            "[{0}_ACTOR]".format(label),
            specs[agent_type]["fields"],
            actor_input,
            full,
            prefixes=tuple(specs[agent_type]["fields"]),
        )
        _summary("[{0}_CRITIC]".format(label), critic_input)
        _print_named(
            "[{0}_CRITIC]".format(label),
            critic_fields,
            critic_input,
            full,
            prefixes=("entity.blue_ground_1.", "local."),
        )
        print(
            "[{0}_OUTPUT] entity={1} action={2} value={3:.9f}".format(
                label, entity_name, decision["action"], decision["value"]
            )
        )
        outputs[agent_type] = {
            "actor": actor_input.copy(),
            "critic": critic_input.copy(),
            "value": float(decision["value"]),
        }
    return outputs, global_fields


def _assert_dimensions(outputs, specs, global_dim):
    for agent_type in REPRESENTATIVES:
        local_dim = int(specs[agent_type]["obs_dim"])
        assert outputs[agent_type]["actor"].size == local_dim
        assert outputs[agent_type]["critic"].size == global_dim + local_dim


def _effect_probe(env, api, specs, bottom, before, fields, full):
    hidden = env.platforms["blue_ground_1"]
    assert "blue_ground_1" not in env.detected_targets
    hidden.lat += 0.55
    hidden.lon += 0.65
    hidden.current_hp = 1.0
    hidden.velocity_east_mps = 42.0
    hidden.velocity_north_mps = -18.0
    hidden.velocity_up_mps = 2.5

    print("\n=== HIDDEN-ENTITY FIELD EFFECT PROBE ===")
    after, after_fields = _run_all_networks(api, specs, bottom, full=False)
    assert fields == after_fields
    _assert_dimensions(after, specs, len(fields))

    for agent_type in REPRESENTATIVES:
        actor_delta = float(np.max(np.abs(after[agent_type]["actor"] - before[agent_type]["actor"])))
        critic_delta = float(np.max(np.abs(after[agent_type]["critic"] - before[agent_type]["critic"])))
        value_delta = abs(after[agent_type]["value"] - before[agent_type]["value"])
        print(
            "[FIELD_EFFECT:{0}] actor_max_delta={1:.9g} critic_max_delta={2:.9g} value_delta={3:.9g}".format(
                agent_type.upper(), actor_delta, critic_delta, value_delta
            )
        )
        assert actor_delta == 0.0, (agent_type, actor_delta)
        assert critic_delta > 0.0, (agent_type, critic_delta)
        assert value_delta > 1.0e-10, (agent_type, value_delta)

    field_index = {name: index for index, name in enumerate(fields)}
    expected = (
        "entity.blue_ground_1.east_norm",
        "entity.blue_ground_1.north_norm",
        "entity.blue_ground_1.velocity_east_norm",
        "entity.blue_ground_1.velocity_north_norm",
        "entity.blue_ground_1.velocity_up_norm",
        "entity.blue_ground_1.hp_norm",
    )
    for field in expected:
        index = field_index[field]
        old = before["attack"]["critic"][index]
        new = after["attack"]["critic"][index]
        print("[FIELD_EFFECT] {0}: {1:.6f} -> {2:.6f}".format(field, old, new))
        assert abs(float(new - old)) > 1.0e-9, field

    assert not any(name.endswith(".detected_by_red") for name in fields)
    assert not any(name.endswith(".detected_by_blue") for name in fields)

    carrier_field = "entity.red_carrier.aam_reserve_norm"
    carrier_index = field_index[carrier_field]
    stock_before = np.asarray(api.get_critic_global_state()["obs"], dtype=np.float32)
    old_stock = int(env.carrier_ammo_stock["fox3"])
    env.carrier_ammo_stock["fox3"] = max(0, old_stock - 1)
    stock_after = np.asarray(api.get_critic_global_state()["obs"], dtype=np.float32)
    print(
        "[FIELD_EFFECT] {0}: {1:.6f} -> {2:.6f}".format(
            carrier_field,
            float(stock_before[carrier_index]),
            float(stock_after[carrier_index]),
        )
    )
    assert float(stock_after[carrier_index]) < float(stock_before[carrier_index])

    blue_base = env.get_critic_global_state_dict()
    assert blue_base["entity.blue_base.aam_reserve_norm"] == 1.0
    assert blue_base["entity.blue_base.agm_reserve_norm"] == 1.0
    assert blue_base["entity.blue_base.ammo_reserve_unlimited"] == 1.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print real Actor/Critic inputs and verify global-state field effects."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print every critic field and per-unit-type raw-to-state traces.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print raw physical/cache values and their 27-field state conversion.",
    )
    parser.add_argument(
        "--bind",
        action="store_true",
        help="Read the current live AFSIM/Warlock scene; no actions are sent.",
    )
    parser.add_argument("--auto-start-warlock", action="store_true")
    parser.add_argument("--platform-timeout", type=float, default=75.0)
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(17)
    torch.manual_seed(17)
    env = AFSIMIslandEnv(
        bind=args.bind,
        auto_start_warlock=args.auto_start_warlock,
    )
    try:
        api = AFSIMRLInterface(env)
        if args.bind:
            ready = env.wait_for_platforms(
                LIVE_REQUIRED_PLATFORMS,
                timeout=args.platform_timeout,
            )
            registered = sum(
                platform.platform_id is not None for platform in env.platforms.values()
            )
            print(
                "LIVE_SCENE ready={0} registered={1}/{2}".format(
                    ready, registered, len(env.platforms)
                )
            )
            if not ready:
                raise RuntimeError("live scene did not register required platforms")
        else:
            api.reset()
            set_synthetic_platforms(env)
            env.detected_targets.clear()
            create_synthetic_active_groups(env)
            # Keep the effect-test entity actor-hidden even if a synthetic group
            # populated some other detected target.
            env.detected_targets.pop("blue_ground_1", None)
            env.ground_detected_targets.pop("blue_ground_1", None)
            if "blue_ground_1" in env.enemy_track_memory:
                env.enemy_track_memory["blue_ground_1"]["CurrentlyDetected"] = False

        specs = api.get_agent_specs()
        global_dim = int(specs["global"]["obs_dim"])
        entity_count = len(env.platforms)
        entity_feature_dim = len(env.CRITIC_ENTITY_FEATURES)
        bottom = HAPPOTrainer(
            specs,
            global_state_dim=global_dim,
            hidden_sizes=(32, 32),
            config=HAPPOConfig(update_epochs=1),
        )

        print(
            "NETWORK_LAYOUT entities={0} entity_fields={1} global_dim={2}".format(
                entity_count, entity_feature_dim, global_dim
            )
        )
        if args.full or args.raw:
            _print_raw_entity_traces(env)
        before, fields = _run_all_networks(api, specs, bottom, args.full)
        _assert_dimensions(before, specs, global_dim)
        if args.bind:
            print("live_network_input_capture_ok True")
        else:
            _effect_probe(env, api, specs, bottom, before, fields, args.full)
            print("network_input_diagnostic_ok True")
    finally:
        env.close()


if __name__ == "__main__":
    main()
