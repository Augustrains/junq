"""Materialize a dynamic stage snapshot as an AFSIM input scenario."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Mapping

from train.stage_snapshots import validate_snapshot


def _dms(value: float, latitude: bool) -> str:
    suffix = ("n" if value >= 0 else "s") if latitude else ("e" if value >= 0 else "w")
    absolute = abs(float(value))
    degrees = int(absolute)
    minutes_value = (absolute - degrees) * 60.0
    minutes = int(minutes_value)
    seconds = (minutes_value - minutes) * 60.0
    return "{0}:{1:02d}:{2:06.3f}{3}".format(degrees, minutes, seconds, suffix)


def _platform_pattern(name: str):
    return re.compile(
        r"^platform\s+" + re.escape(name) + r"(?:\s|$).*?\bend_platform\s*$",
        re.MULTILINE | re.DOTALL,
    )


def _set_aux_bool(block: str, name: str, value: bool) -> str:
    replacement = "bool {0} = {1}".format(name, "true" if value else "false")
    pattern = re.compile(r"bool\s+" + re.escape(name) + r"\s*=\s*(?:true|false)", re.IGNORECASE)
    if pattern.search(block):
        return pattern.sub(replacement, block, count=1)
    return block.replace("end_aux_data", "   {0}\n   end_aux_data".format(replacement), 1)


def _set_aux_double(block: str, name: str, value: float) -> str:
    replacement = "double {0} = {1:.6f}".format(name, float(value))
    pattern = re.compile(r"double\s+" + re.escape(name) + r"\s*=\s*[-+0-9.eE]+", re.IGNORECASE)
    if pattern.search(block):
        return pattern.sub(replacement, block, count=1)
    return block.replace("end_aux_data", "   {0}\n   end_aux_data".format(replacement), 1)


def _set_aux_int(block: str, name: str, value: int) -> str:
    replacement = "int {0} = {1}".format(name, int(value))
    pattern = re.compile(r"int\s+" + re.escape(name) + r"\s*=\s*-?\d+", re.IGNORECASE)
    if pattern.search(block):
        return pattern.sub(replacement, block, count=1)
    return block.replace("end_aux_data", "   {0}\n   end_aux_data".format(replacement), 1)


def _materialize_platform(block: str, name: str, platform: Mapping[str, object], state: Mapping[str, object]) -> str:
    lat = _dms(float(platform.get("lat", 0.0)), latitude=True)
    lon = _dms(float(platform.get("lon", 0.0)), latitude=False)
    alt = float(platform.get("alt", 0.0))
    position = re.compile(
        r"position\s+\S+\s+\S+\s+altitude\s+[-+0-9.eE]+\s+(?:ft|m)\s+(msl|agl)",
        re.IGNORECASE,
    )
    match = position.search(block)
    reference = match.group(1) if match else "msl"
    replacement = "position {0} {1} altitude {2:.3f} m {3}".format(lat, lon, alt, reference)
    if match:
        block = position.sub(replacement, block, count=1)
    heading = float(platform.get("heading", 0.0)) % 360.0
    block = re.sub(r"\bheading\s+[-+0-9.eE]+\s+deg(?:rees)?", "heading {0:.3f} deg".format(heading), block, count=1)

    landing_cargo = state.get("landing_cargo", {}).get(name, {})
    ground_status = state.get("ground_status", {}).get(name, {})
    if landing_cargo:
        block = _set_aux_bool(block, "HAS_ARMY", bool(landing_cargo.get("has_army", False)))
        block = _set_aux_bool(block, "ARMY_LANDED", bool(landing_cargo.get("army_landed", False)))
    if ground_status:
        block = _set_aux_bool(block, "ON_SHIP", bool(ground_status.get("on_ship", False)))
        block = _set_aux_bool(block, "LANDED", bool(ground_status.get("landed", False)))

    source_time = float(state.get("source_sim_time", 0.0))
    if "max_hp" in platform:
        block = _set_aux_double(block, "MAX_HP", float(platform.get("max_hp", 0.0)))
        block = _set_aux_double(block, "CURRENT_HP", float(platform.get("current_hp", platform.get("max_hp", 0.0))))
        block = _set_aux_bool(block, "IS_INDESTRUCTIBLE", bool(platform.get("indestructible", False)))
        block = _set_aux_double(block, "LAST_DAMAGE_TIME", -1.0)
        block = _set_aux_double(block, "LAST_ATTACK_TIME", -1.0)
        block = _set_aux_double(block, "FIRE_COOLDOWN_UNTIL", max(0.0, float(platform.get("fire_cooldown_until", 0.0)) - source_time))
        block = _set_aux_double(block, "COMBAT_LOCK_UNTIL", max(0.0, float(platform.get("combat_lock_until", 0.0)) - source_time))

    ammo = state.get("attack_ammo", {}).get(name, {})
    ground_ammo = state.get("ground_ammo", {}).get(name, {})
    carrier_stock = state.get("carrier_ammo_stock", {}) if name == "red_carrier" else {}
    if carrier_stock:
        block = _set_aux_int(block, "AAM_STOCK", int(carrier_stock.get("fox3", 0)))
        block = _set_aux_int(block, "AGM_STOCK", int(carrier_stock.get("agm", 0)))
    edits = []
    if ammo:
        edits.extend([
            "   edit weapon fox3 quantity {0} end_weapon".format(max(0, int(ammo.get("fox3", 0)))),
            "   edit weapon agm quantity {0} end_weapon".format(max(0, int(ammo.get("agm", 0)))),
        ])
        block = _set_aux_int(block, "CURRENT_AAM", int(ammo.get("fox3", 0)))
        block = _set_aux_int(block, "CURRENT_AGM", int(ammo.get("agm", 0)))
    if ground_ammo:
        edits.append("   edit weapon ground_fire quantity {0} end_weapon".format(max(0, int(ground_ammo.get("ground_fire", 0)))))
    if edits:
        block = block.rsplit("end_platform", 1)[0].rstrip() + "\n" + "\n".join(edits) + "\nend_platform"
    return block


def render_snapshot_scenario(source_text: str, snapshot: Mapping[str, object]) -> str:
    errors = validate_snapshot(snapshot)
    if errors:
        raise ValueError("cannot render invalid snapshot: {0}".format(", ".join(errors)))
    text = source_text
    state = dict(snapshot.get("python_state", {}))
    state["source_sim_time"] = float(snapshot.get("source_sim_time", 0.0))
    for name, platform in snapshot["platforms"].items():
        pattern = _platform_pattern(name)
        match = pattern.search(text)
        if not match:
            raise RuntimeError("source scenario is missing platform {0}".format(name))
        if not bool(platform.get("alive", True)):
            text = pattern.sub("# snapshot removed destroyed platform {0}".format(name), text, count=1)
            continue
        replacement = _materialize_platform(match.group(0), name, platform, state)
        text = pattern.sub(lambda _match, value=replacement: value, text, count=1)
    header = (
        "# GENERATED DYNAMIC CURRICULUM SNAPSHOT\n"
        "# snapshot_id={0} stage={1} policy={2}\n"
    ).format(snapshot.get("snapshot_id", ""), snapshot.get("stage", ""), snapshot.get("policy_fingerprint", ""))
    return header + text


def materialize_snapshot(snapshot_path, source_scenario, output_scenario, output_config=None, base_config="afsim_units.json"):
    snapshot_path = Path(snapshot_path).resolve()
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    source_scenario = Path(source_scenario).resolve()
    output_scenario = Path(output_scenario).resolve()
    output_scenario.parent.mkdir(parents=True, exist_ok=True)
    rendered = render_snapshot_scenario(source_scenario.read_text(encoding="utf-8"), snapshot)
    output_scenario.write_text(rendered, encoding="utf-8", newline="\n")
    result = {"snapshot": str(snapshot_path), "scenario": str(output_scenario), "stage": snapshot["stage"]}
    if output_config:
        output_config = Path(output_config).resolve()
        output_config.parent.mkdir(parents=True, exist_ok=True)
        config = {
            "extends": str(base_config),
            "scenario": {
                "name": "dynamic_{0}".format(snapshot["stage"]),
                "scenario_file": str(output_scenario).replace("\\", "/"),
                "stage_snapshot_file": str(snapshot_path).replace("\\", "/"),
            },
        }
        output_config.write_text(json.dumps(config, ensure_ascii=True, indent=2), encoding="utf-8")
        result["config"] = str(output_config)
    return result


def restore_env_from_snapshot(env, snapshot_or_path):
    if isinstance(snapshot_or_path, (str, Path)):
        snapshot = json.loads(Path(snapshot_or_path).read_text(encoding="utf-8"))
    else:
        snapshot = deepcopy(snapshot_or_path)
    errors = validate_snapshot(snapshot)
    if errors:
        raise ValueError("invalid stage snapshot: {0}".format(", ".join(errors)))
    for name, values in snapshot["platforms"].items():
        platform = env.platforms.get(name)
        if platform is None:
            continue
        for key in ("alive", "lat", "lon", "alt", "heading", "speed", "detected", "at_home", "max_hp", "current_hp", "indestructible", "last_damage_time", "last_attack_time"):
            if key in values:
                setattr(platform, key, values[key])
        source_time = float(snapshot.get("source_sim_time", 0.0))
        current_time = float(env._current_sim_time())
        platform.fire_cooldown_until = current_time + max(0.0, float(values.get("fire_cooldown_until", 0.0)) - source_time)
        platform.combat_lock_until = current_time + max(0.0, float(values.get("combat_lock_until", 0.0)) - source_time)
        platform.task = "PARKED" if platform.alive else "DEFEATED"
        platform.task_status = "IDLE"
    state = snapshot.get("python_state", {})
    env.detected_targets = deepcopy(state.get("detected_targets", {}))
    env.attack_local_detections = {}
    env.enemy_track_memory = deepcopy(state.get("enemy_track_memory", env.detected_targets))
    env.ground_detected_targets = deepcopy(state.get("ground_detected_targets", {}))
    env.attack_ammo = deepcopy(state.get("attack_ammo", env.attack_ammo))
    env.carrier_ammo_stock = deepcopy(state.get("carrier_ammo_stock", env.carrier_ammo_stock))
    env.attack_target_reservations = {}
    env.ground_target_reservations = {}
    env.pending_attack_fire_commands = {}
    env.pending_attack_returns = {}
    env.landing_cargo = deepcopy(state.get("landing_cargo", env.landing_cargo))
    env.pending_landing_unloads = {}
    env.ground_status = deepcopy(state.get("ground_status", env.ground_status))
    env.ground_ammo = deepcopy(state.get("ground_ammo", env.ground_ammo))
    env.task_flags = deepcopy(state.get("task_flags", env.task_flags))
    env.last_reward_events = []
    env.last_reward_details = []
    env.capture_condition_since = None
    for controller in (env.recon_controller, env.attack_controller, env.landing_controller, env.ground_controller):
        controller.active_groups.clear()
        controller.next_group_index = 1
    env.reward_manager.reset(env)
    env.loaded_stage_snapshot = {
        "snapshot_id": snapshot.get("snapshot_id", ""),
        "stage": snapshot.get("stage", ""),
        "policy_fingerprint": snapshot.get("policy_fingerprint", ""),
        "source_sim_time": snapshot.get("source_sim_time", 0.0),
    }
    return dict(env.loaded_stage_snapshot)


class SnapshotRestoringInterface:
    """Transparent interface wrapper that restores a snapshot after reset."""

    def __init__(self, interface, snapshot_path):
        self._interface = interface
        self.env = interface.env
        self.snapshot_path = str(snapshot_path)

    def reset(self):
        return self.reset_rule_driven()

    def reset_rule_driven(self):
        self._interface.env.reset()
        restore_env_from_snapshot(self.env, self.snapshot_path)
        assignments = self._interface.ensure_rule_driven_tasks()
        return {
            "control_mode": "rule_driven_bottom",
            "assignments": assignments,
        }

    def __getattr__(self, name):
        return getattr(self._interface, name)
