import json
import math
import os
import socket
import subprocess
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from gym import spaces
except ImportError:
    class _Box(object):
        def __init__(self, low, high, shape, dtype=np.float32):
            self.low = low
            self.high = high
            self.shape = tuple(shape)
            self.dtype = dtype

    class _Tuple(object):
        def __init__(self, spaces):
            self.spaces = tuple(spaces)

    class _Spaces(object):
        Box = _Box
        Tuple = _Tuple

    spaces = _Spaces()


try:
    from envs.controllers.recon_controller import ReconController
    from envs.controllers.attack_controller import AttackController
    from envs.controllers.ground_controller import GroundController
    from envs.controllers.landing_controller import LandingController
    from envs.reward_manager import RewardManager
    from envs.landing_navigation import SeaGridDistanceField
except Exception:
    from controllers.recon_controller import ReconController
    from controllers.attack_controller import AttackController
    from controllers.ground_controller import GroundController
    from controllers.landing_controller import LandingController
    from reward_manager import RewardManager
    from landing_navigation import SeaGridDistanceField


@dataclass
class PlatformState:
    name: str
    role: str
    side: str
    platform_id: Optional[int] = None
    platform_type: str = ""
    alive: bool = True
    lat: float = 0.0
    lon: float = 0.0
    alt: float = 0.0
    task: str = "PARKED"
    task_status: str = "IDLE"
    task_assigned: bool = False
    at_home: bool = False
    detected: bool = False
    heading: float = 0.0
    speed: float = 0.0
    velocity_east_mps: float = 0.0
    velocity_north_mps: float = 0.0
    velocity_up_mps: float = 0.0
    last_update: float = 0.0
    max_hp: float = 0.0
    current_hp: float = 0.0
    indestructible: bool = False
    last_damage_time: float = -1.0
    last_attack_time: float = -1.0
    fire_cooldown_until: float = 0.0
    combat_lock_until: float = 0.0
    rearm_complete_at: float = 0.0


def _add_friendly_carrier(values, platforms, red_cfg, carrier_ammo, bounds):
    names = red_cfg.get("carrier", [])
    if not names:
        values["friendly_carrier_lat_norm"] = 0.0
        values["friendly_carrier_lon_norm"] = 0.0
        values["friendly_carrier_aam_stock_norm"] = 0.0
        values["friendly_carrier_agm_stock_norm"] = 0.0
        return
    carrier = platforms.get(names[0])
    if carrier and carrier.alive:
        values["friendly_carrier_lat_norm"] = _safe_norm_lat(carrier, bounds)
        values["friendly_carrier_lon_norm"] = _safe_norm_lon(carrier, bounds)
    else:
        values["friendly_carrier_lat_norm"] = 0.0
        values["friendly_carrier_lon_norm"] = 0.0
    values["friendly_carrier_aam_stock_norm"] = min(1.0, float(carrier_ammo.get("fox3", 0)) / 30.0)
    values["friendly_carrier_agm_stock_norm"] = min(1.0, float(carrier_ammo.get("agm", 0)) / 50.0)


def _add_friendly_recon(values, platforms, red_cfg, max_speed, bounds, own_name=""):
    names = red_cfg.get("recon_aircraft", [])
    for slot, name in enumerate(names, start=1):
        prefix = "friendly_recon_{0}_".format(slot)
        if name == own_name:
            _zero_friendly_slot(values, prefix, ["lat_norm", "lon_norm", "speed_norm", "hp_norm"])
            continue
        p = platforms.get(name)
        if p and p.alive:
            values[prefix + "lat_norm"] = _safe_norm_lat(p, bounds)
            values[prefix + "lon_norm"] = _safe_norm_lon(p, bounds)
            values[prefix + "speed_norm"] = min(1.0, max(0.0, float(getattr(p, "speed", 0.0)) / max(1.0, max_speed)))
            values[prefix + "hp_norm"] = _safe_hp_norm(p)
        else:
            _zero_friendly_slot(values, prefix, ["lat_norm", "lon_norm", "speed_norm", "hp_norm"])


def _add_friendly_attack(values, platforms, red_cfg, group, attack_ammo, max_speed, bounds, own_name=""):
    names = red_cfg.get("attack_aircraft", [])
    for slot, name in enumerate(names, start=1):
        prefix = "friendly_attack_{0}_".format(slot)
        if name == own_name:
            _zero_friendly_slot(values, prefix, ["lat_norm", "lon_norm", "speed_norm", "hp_norm", "aam_norm", "agm_norm"])
            continue
        p = platforms.get(name)
        if p and p.alive:
            values[prefix + "lat_norm"] = _safe_norm_lat(p, bounds)
            values[prefix + "lon_norm"] = _safe_norm_lon(p, bounds)
            values[prefix + "speed_norm"] = min(1.0, max(0.0, float(getattr(p, "speed", 0.0)) / max(1.0, max_speed)))
            values[prefix + "hp_norm"] = _safe_hp_norm(p)
            ammo = attack_ammo.get(name, {"fox3": 1, "agm": 1})
            values[prefix + "aam_norm"] = 1.0 if int(ammo.get("fox3", 0)) > 0 else 0.0
            values[prefix + "agm_norm"] = 1.0 if int(ammo.get("agm", 0)) > 0 else 0.0
        else:
            _zero_friendly_slot(values, prefix, ["lat_norm", "lon_norm", "speed_norm", "hp_norm", "aam_norm", "agm_norm"])


def _add_friendly_ground(values, platforms, red_cfg, ground_ammo, max_speed, bounds):
    names = red_cfg.get("ground_forces", [])
    for slot, name in enumerate(names, start=1):
        prefix = "friendly_ground_{0}_".format(slot)
        p = platforms.get(name)
        if p and p.alive:
            values[prefix + "lat_norm"] = _safe_norm_lat(p, bounds)
            values[prefix + "lon_norm"] = _safe_norm_lon(p, bounds)
            values[prefix + "speed_norm"] = min(1.0, max(0.0, float(getattr(p, "speed", 0.0)) / max(1.0, max_speed)))
            values[prefix + "hp_norm"] = _safe_hp_norm(p)
            ammo = ground_ammo.get(name, {"ground_fire": 0})
            if isinstance(ammo, dict):
                ammo = ammo.get("ground_fire", 0)
            values[prefix + "ammo_norm"] = min(1.0, float(ammo) / 15.0)
        else:
            _zero_friendly_slot(values, prefix, ["lat_norm", "lon_norm", "speed_norm", "hp_norm", "ammo_norm"])


def _zero_friendly_slot(values, prefix, suffixes):
    for s in suffixes:
        values[prefix + s] = 0.0


def _safe_norm_lat(p, bounds=None):
    if bounds is None:
        lat_min, lat_max = 23.5, 25.8
    else:
        lat_min = float(bounds.get("lat_min", 23.5))
        lat_max = float(bounds.get("lat_max", 25.8))
    return (float(getattr(p, "lat", lat_min)) - lat_min) / max(1e-6, lat_max - lat_min)


def _safe_norm_lon(p, bounds=None):
    if bounds is None:
        lon_min, lon_max = 118.8, 122.2
    else:
        lon_min = float(bounds.get("lon_min", 118.8))
        lon_max = float(bounds.get("lon_max", 122.2))
    return (float(getattr(p, "lon", lon_min)) - lon_min) / max(1e-6, lon_max - lon_min)


def _safe_hp_norm(p):
    max_hp = float(getattr(p, "max_hp", 1.0))
    if max_hp <= 0.0:
        return 0.0
    return min(1.0, max(0.0, float(getattr(p, "current_hp", 0.0)) / max_hp))


def _expand_field_block(expanded, marker_prefix, templates, replacement):
    """Find a template block starting with marker_prefix and replace it with expanded slots."""
    idx = next((i for i, f in enumerate(expanded) if str(f.get("name", "")).startswith(marker_prefix)), -1)
    if idx < 0 or not templates:
        return
    before = expanded[:idx]
    after = expanded[idx + len(templates):]
    expanded[:] = before + replacement + after


def _expand_friendly_slots(prefix, names, templates):
    """Expand friendly_<prefix>_1_<suffix> templates to one slot per unit."""
    slots = []
    for slot, _name in enumerate(names, start=1):
        for suffix, template in templates:
            field = dict(template)
            field["name"] = "friendly_{0}_{1}_{2}".format(prefix, slot, suffix)
            field["description"] = "Friendly {0} {1}: {2}".format(
                prefix, slot, str(template.get("description", ""))
            )
            slots.append(field)
    return slots


def _expand_target_slots(target_names, templates, role_fn):
    """Expand target_slot_1_<suffix> templates to one slot per enemy target."""
    slots = []
    for slot, target_name in enumerate(target_names, start=1):
        role = role_fn(target_name)
        allowed_suffixes = {"lat_norm", "lon_norm", "alt_norm", "hp_norm"}
        if role == "attack_aircraft":
            allowed_suffixes.update({"aam_ammo_norm", "agm_ammo_norm"})
        elif role == "ground_force":
            allowed_suffixes.add("ground_ammo_norm")
        elif role == "sam":
            allowed_suffixes.add("sam_ammo_norm")
        for suffix, template in templates:
            if suffix not in allowed_suffixes:
                continue
            field = dict(template)
            field["name"] = "target_slot_{0}_{1}".format(slot, suffix)
            field["description"] = "Enemy {0} ({1}): {2}".format(
                slot, target_name,
                str(template.get("description", "")).split(":", 1)[-1].strip(),
            )
            slots.append(field)
    return slots




class AFSIMIslandEnv(object):
    """AFSIM environment for rule-assigned bottom-level policies."""

    CRITIC_ENTITY_FEATURES = (
        "registered",
        "alive",
        "indestructible",
        "side_id",
        "role_id",
        "east_norm",
        "north_norm",
        "alt_norm",
        "velocity_east_norm",
        "velocity_north_norm",
        "velocity_up_norm",
        "hp_norm",
        "fire_cooldown_remaining_norm",
        "combat_lock_remaining_norm",
        "aam_ammo_norm",
        "agm_ammo_norm",
        "ground_ammo_norm",
        "aam_reserve_norm",
        "agm_reserve_norm",
        "task_id",
        "at_home",
        "on_ship",
        "landed",
        "has_army",
        "army_landed",
        "ammo_reserve_unlimited",
        "rearm_remaining_norm",
    )
    # High-level Actor information is intentionally asymmetric with the
    # omniscient Critic. Own red units use fixed named slots because the
    # commander can query them directly. Blue contacts use anonymous,
    # first-seen slots populated only from red-side track memory.
    COMMANDER_OWN_ENTITY_FEATURES = (
        "registered",
        "alive",
        "indestructible",
        "role_id_norm",
        "lat_norm",
        "lon_norm",
        "alt_norm",
        "velocity_east_norm",
        "velocity_north_norm",
        "velocity_up_norm",
        "hp_norm",
        "fire_cooldown_remaining_norm",
        "combat_lock_remaining_norm",
        "aam_ammo_norm",
        "agm_ammo_norm",
        "ground_ammo_norm",
        "aam_reserve_norm",
        "agm_reserve_norm",
        "task_id_norm",
        "at_home",
        "on_ship",
        "landed",
        "has_army",
        "army_landed",
        "ammo_reserve_unlimited",
        "rearm_remaining_norm",
    )
    COMMANDER_KNOWN_CONTACT_FEATURES = (
        "known",
        "currently_detected",
        "role_id_norm",
        "lat_norm",
        "lon_norm",
        "alt_norm",
        "hp_known",
        "hp_norm",
        "alive_last_known",
        "track_age_norm",
    )
    CRITIC_ROLE_IDS = {
        "carrier": 0,
        "recon_aircraft": 1,
        "attack_aircraft": 2,
        "transport": 3,
        "ground_force": 4,
        "base": 5,
        "radar": 6,
        "sam": 7,
    }
    CRITIC_TASK_IDS = {
        "idle": 0,
        "moving": 1,
        "recon": 2,
        "attack": 3,
        "landing": 4,
        "unloading": 5,
        "ground": 6,
        "return": 7,
        "service": 8,
        "destroyed": 9,
    }

    def __init__(self, config_path=None, observation_fields_path=None, bind=True, auto_start_warlock=False, local_address=None, stage_snapshot_path=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = config_path or os.path.join(base_dir, "afsim_units.json")
        self.config = self._load_config(self.config_path)
        self.observation_fields_path = observation_fields_path or os.path.join(base_dir, "agent_observation_fields.json")
        self.agent_observation_fields = self._load_config(self.observation_fields_path)
        scenario_cfg = self.config.get("scenario", {})
        self.recon_areas_path = os.path.join(base_dir, scenario_cfg.get("recon_areas_file", ""))
        self.recon_actions_path = os.path.join(base_dir, scenario_cfg.get("recon_actions_file", "recon_actions.json"))
        self.recon_state_fields_path = os.path.join(base_dir, scenario_cfg.get("recon_state_fields_file", "recon_state_fields.json"))
        self.attack_actions_path = os.path.join(base_dir, scenario_cfg.get("attack_actions_file", "attack_actions.json"))
        self.attack_state_fields_path = os.path.join(base_dir, scenario_cfg.get("attack_state_fields_file", "attack_state_fields.json"))
        self.landing_actions_path = os.path.join(base_dir, scenario_cfg.get("landing_actions_file", "landing_actions.json"))
        self.landing_state_fields_path = os.path.join(base_dir, scenario_cfg.get("landing_state_fields_file", "landing_state_fields.json"))
        self.ground_actions_path = os.path.join(base_dir, scenario_cfg.get("ground_actions_file", "ground_actions.json"))
        self.ground_state_fields_path = os.path.join(base_dir, scenario_cfg.get("ground_state_fields_file", "ground_state_fields.json"))
        self.commander_state_fields_path = os.path.join(base_dir, scenario_cfg.get("commander_state_fields_file", "commander_state_fields.json"))
        self.reward_rules_path = os.path.join(base_dir, scenario_cfg.get("reward_rules_file", "reward_rules.json"))
        self.reward_target_priorities_path = os.path.join(
            base_dir, scenario_cfg.get("reward_target_priorities_file", "reward_target_priorities.json")
        )
        self.combat_model_path = os.path.join(base_dir, scenario_cfg.get("combat_model_file", "combat_model.json"))
        self.done_rules_path = os.path.join(base_dir, scenario_cfg.get("done_rules_file", "done_rules.json"))
        self.recon_area_config = (
            self._load_config(self.recon_areas_path)
            if scenario_cfg.get("recon_areas_file") else {"areas": []}
        )
        self.recon_action_config = self._load_config(self.recon_actions_path)
        self.recon_state_config = self._load_config(self.recon_state_fields_path)
        self.attack_action_config = self._load_config(self.attack_actions_path)
        self.attack_state_config = self._load_config(self.attack_state_fields_path)
        self.attack_fixed_target_names = self._configured_attack_target_names()
        self._configure_fixed_recon_schema()
        self._configure_fixed_attack_schema()
        self.landing_action_config = self._load_config(self.landing_actions_path)
        self.landing_state_config = self._load_config(self.landing_state_fields_path)
        self.ground_action_config = self._load_config(self.ground_actions_path)
        self.ground_state_config = self._load_config(self.ground_state_fields_path)
        self.ground_fixed_target_names = self._configured_ground_target_names()
        self._configure_fixed_ground_schema()
        self.commander_state_config = self._load_config(self.commander_state_fields_path)
        self.reward_target_priority_config = self._load_config(self.reward_target_priorities_path)
        self.done_rules_config = self._load_config(self.done_rules_path)
        self.combat_model_config = self._load_config(self.combat_model_path)
        self.recon_areas = self.recon_area_config.get("areas", [])
        self.recon_area_coverage: Dict[str, set] = {
            str(self._normalize_recon_area(area).get("name", "")): set()
            for area in self.recon_areas
        }
        self.recon_area_last_observed_time: Dict[str, Optional[float]] = {
            str(self._normalize_recon_area(area).get("name", "")): None
            for area in self.recon_areas
        }
        self.recon_state_fields = [field["name"] for field in self.recon_state_config.get("fields", [])]
        self.attack_state_fields = [field["name"] for field in self.attack_state_config.get("fields", [])]
        self.landing_state_fields = [field["name"] for field in self.landing_state_config.get("fields", [])]
        self.ground_state_fields = [field["name"] for field in self.ground_state_config.get("fields", [])]
        self.commander_state_fields = [field["name"] for field in self.commander_state_config.get("fields", [])]
        self.observation_mode = scenario_cfg.get("observation_mode", "commander")
        configured_local_address = tuple(scenario_cfg.get("local_address", ["127.0.0.1", 50050]))
        local_address = tuple(local_address or configured_local_address)
        self.local_address = (local_address[0], int(local_address[1]))
        self.decision_seconds = float(scenario_cfg.get("decision_seconds", 1.8))
        self.sim_time_window_control = bool(scenario_cfg.get("sim_time_window_control", False))
        self.lockstep_pause_control = bool(scenario_cfg.get("lockstep_pause_control", False))
        self.native_decision_pause_control = bool(scenario_cfg.get("native_decision_pause_control", False))
        self.native_decision_pause_timeout = float(scenario_cfg.get("native_decision_pause_timeout", 45.0))
        self.native_decision_pause_retries = int(scenario_cfg.get("native_decision_pause_retries", 3))
        self.native_decision_retry_timeout = float(scenario_cfg.get("native_decision_retry_timeout", 10.0))
        self.native_decision_ready = False
        self.native_decision_ready_time = 0.0
        self.native_decision_ready_seen = False
        self._native_restart_pending = False
        self._native_restart_boundary_time = None
        self._native_restart_last_send = 0.0
        self._native_restart_resend_seconds = 0.10
        self.debug_native_decision_pause = str(os.environ.get("AFSIM_DEBUG_NATIVE_PAUSE", "")).lower() in ("1", "true", "yes", "on")
        self.simulation_paused = False
        self.sim_time_window_max_wall_seconds = float(scenario_cfg.get("sim_time_window_max_wall_seconds", 4.0))
        self.sim_time_window_early_tolerance_seconds = float(scenario_cfg.get("sim_time_window_early_tolerance_seconds", 7.5))
        self._next_decision_target_sim_time = None
        self.last_drain_metadata = {}
        self.message_timeout_seconds = float(scenario_cfg.get("message_timeout_seconds", 0.2))
        self.max_steps = int(scenario_cfg.get("max_steps", 2000))
        self.bounds = scenario_cfg.get("bounds", {})
        self._landing_distance_fields = {}
        configured_snapshot = scenario_cfg.get("stage_snapshot_file", "")
        self.stage_snapshot_path = stage_snapshot_path or configured_snapshot or ""
        if self.stage_snapshot_path and not os.path.isabs(self.stage_snapshot_path):
            self.stage_snapshot_path = os.path.join(base_dir, self.stage_snapshot_path)
        self.loaded_stage_snapshot = None

        self.live_mode = bool(bind)
        self.sock = None
        self.remote_addr = None
        self.last_platform_state_wall_time = 0.0
        self.last_decision_ready_wall_time = 0.0
        self.platform_state_sequence = 0
        self.warlock_process = None
        self.warlock_log_handle = None
        if bind:
            self._open_socket()

        self.platforms: Dict[str, PlatformState] = {}
        self.detected_targets: Dict[str, dict] = {}
        self.attack_local_detections: Dict[str, Dict[str, dict]] = {}
        # Actor-facing target observations. Unlike enemy_track_memory, these
        # snapshots change only when a real detection/contact report arrives.
        self.attack_target_snapshots: Dict[str, dict] = {}
        self.debug_attack_contact_reports = str(os.environ.get("AFSIM_DEBUG_ATTACK_CONTACTS", "")).lower() in ("1", "true", "yes", "on")
        self.debug_combat_events = str(os.environ.get("AFSIM_DEBUG_COMBAT_EVENTS", "" )).lower() in ("1", "true", "yes", "on")
        self.last_attack_contact_reports: List[dict] = []
        self.attack_local_track_ttl_sec = float(scenario_cfg.get("attack_local_track_ttl_sec", 2.0))
        self._decision_window_id = 0
        self._decision_window_sim_time = 0.0
        self._decision_window_observer_targets: Dict[str, set] = {}
        self.enemy_track_memory: Dict[str, dict] = {}
        self.commander_contact_slots: Dict[str, int] = {}
        self.ground_detected_targets: Dict[str, dict] = {}
        self.blue_detected_targets = set()
        self.step_count = 0
        self.episode_id = 0
        self.episode_result = ""
        self.episode_done_reason = "none"
        self.final_score_raw = 0.0
        self.final_score_norm = 0.0
        self.final_score_unit_count = 0
        self.final_score_sim_time = 0.0
        self.final_score_units = []
        self.final_score_settled = False
        self.capture_condition_since = None
        self.last_reward_events: List[dict] = []
        self.task_flags = {
            "landing_executed": False,
            "ground_landed": False,
            "current_recon": "",
            "current_attack": "",
        }
        self.last_command = self._empty_last_command()
        self.action_names: List[str] = []
        self._actions: List[object] = []
        self._build_platform_registry()
        self.recon_controller = ReconController(self.config.get("red", {}), self.recon_action_config)
        self.attack_controller = AttackController(self.config.get("red", {}), self.attack_action_config)
        self.landing_controller = LandingController(self.config.get("red", {}), self.landing_action_config)
        self.ground_controller = GroundController(self.config.get("red", {}), self.ground_action_config)
        for plat in self.platforms.values():
            if plat.role == "recon_aircraft" and (not self.live_mode or plat.task in ("PARKED", "")):
                plat.task_assigned = False
        self.attack_ammo = self._initial_attack_ammo()
        self.sam_ammo = self._initial_sam_ammo()
        self.carrier_ammo_stock = self._initial_carrier_ammo_stock()
        self.attack_target_reservations: Dict[str, str] = {}
        self.attack_last_shooter_by_target: Dict[str, str] = {}
        self.attack_last_weapon_by_target: Dict[str, str] = {}
        self.target_reward_contributors: Dict[str, dict] = {}
        self.ground_target_reservations: Dict[str, dict] = {}
        self.pending_attack_fire_commands: Dict[str, dict] = {}
        self.pending_attack_approaches: Dict[str, dict] = {}
        self.pending_attack_rejoins: Dict[str, dict] = {}

        self.pending_attack_returns = {}
        self.landing_cargo = self._initial_landing_cargo()
        self.pending_landing_unloads = {}
        self.ground_status = self._initial_ground_status()
        self.ground_ammo = self._initial_ground_ammo()
        self._apply_stage_initial_state()
        self.reward_manager = RewardManager(self.reward_rules_path)
        self.last_reward_details: List[dict] = []
        self.negative_rewards_enabled = bool(
            scenario_cfg.get(
                "negative_rewards_enabled",
                self.reward_manager.negative_rewards_enabled))
        self.reward_manager.set_negative_rewards_enabled(self.negative_rewards_enabled)
        self._build_actions()
        # Bottom policies expose their spaces through AFSIMRLInterface.
        self.action_space = None
        self.observation_space = None

        if auto_start_warlock:
            self.start_warlock()

    @classmethod
    def _load_config(cls, path):
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        parent = data.pop("extends", None) if isinstance(data, dict) else None
        if not parent:
            return data
        parent_path = parent if os.path.isabs(parent) else os.path.join(os.path.dirname(path), parent)
        return cls._merge_config(cls._load_config(parent_path), data)

    @classmethod
    def _merge_config(cls, base, override):
        if not isinstance(base, dict) or not isinstance(override, dict):
            return override
        merged = dict(base)
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = cls._merge_config(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _configured_attack_target_names(self):
        """Return the permanent target order used by the attack actor."""
        blue = self.config.get("blue", {})
        names = []
        for key in ("attack_aircraft", "ground_forces", "radars", "sams"):
            for name in blue.get(key, []):
                if name not in names:
                    names.append(str(name))
        return names

    def _configure_fixed_recon_schema(self):
        """Expand recon_state_fields with friendly and enemy target slots."""
        target_names = list(self.attack_fixed_target_names)
        fields = [dict(field) for field in self.recon_state_config.get("fields", [])]
        red_cfg = self.config.get("red", {})
        recon_names = red_cfg.get("recon_aircraft", [])
        attack_names = red_cfg.get("attack_aircraft", [])
        ground_names = red_cfg.get("ground_forces", [])

        recon_tmpl, atk_tmpl, gnd_tmpl, tgt_tmpl = [], [], [], []
        for field in fields:
            name = str(field.get("name", ""))
            parts = name.split("_", 3)
            if not (len(parts) == 4 and parts[2].isdigit()):
                continue
            if parts[0] == "friendly" and parts[1] == "recon" and int(parts[2]) == 1:
                recon_tmpl.append((parts[3], field))
            elif parts[0] == "friendly" and parts[1] == "attack" and int(parts[2]) == 1:
                atk_tmpl.append((parts[3], field))
            elif parts[0] == "friendly" and parts[1] == "ground" and int(parts[2]) == 1:
                gnd_tmpl.append((parts[3], field))
            elif parts[0] == "target" and parts[1] == "slot" and int(parts[2]) == 1:
                tgt_tmpl.append((parts[3], field))

        expanded = list(fields)
        _expand_field_block(expanded, "friendly_recon_1_", recon_tmpl,
                            _expand_friendly_slots("recon", recon_names, recon_tmpl))
        _expand_field_block(expanded, "friendly_attack_1_", atk_tmpl,
                            _expand_friendly_slots("attack", attack_names, atk_tmpl))
        _expand_field_block(expanded, "friendly_ground_1_", gnd_tmpl,
                            _expand_friendly_slots("ground", ground_names, gnd_tmpl))
        _expand_field_block(expanded, "target_slot_1_", tgt_tmpl,
                            _expand_target_slots(target_names, tgt_tmpl, self._configured_blue_target_role))

        self.recon_state_config["fields"] = expanded

    def _configure_fixed_attack_schema(self):
        """Replace the legacy eight dynamic slots with one slot per target."""
        target_names = list(self.attack_fixed_target_names)
        configured_actions = sorted(
            (dict(action) for action in self.attack_action_config.get("actions", [])),
            key=lambda item: int(item.get("id", 0)),
        )
        legacy_target_actions = [
            action for action in configured_actions
            if action.get("afsim_task") == "ATTACK_TARGET_SLOT"
        ]
        target_action_start = min(
            (int(action.get("id", 0)) for action in legacy_target_actions),
            default=9,
        )
        prefix_actions = [
            action for action in configured_actions
            if action.get("afsim_task") != "ATTACK_TARGET_SLOT"
            and int(action.get("id", 0)) < target_action_start
        ]
        suffix_actions = [
            action for action in configured_actions
            if action.get("afsim_task") != "ATTACK_TARGET_SLOT"
            and int(action.get("id", 0)) >= target_action_start
        ]
        fixed_target_actions = []
        for slot, target_name in enumerate(target_names):
            fixed_target_actions.append({
                "id": target_action_start + slot,
                "name": "ATTACK_TARGET_{0}".format(slot + 1),
                "description": "Attack fixed target {0} with an automatically selected compatible weapon.".format(target_name),
                "afsim_task": "ATTACK_TARGET_SLOT",
                "target_slot": slot,
                "target_name": target_name,
                "always_available": True,
            })
        next_action_id = target_action_start + len(fixed_target_actions)
        for offset, action in enumerate(suffix_actions):
            action["id"] = next_action_id + offset
        self.attack_action_config["target_slot_count"] = len(target_names)
        self.attack_action_config["fixed_target_names"] = target_names
        self.attack_action_config["actions"] = (
            prefix_actions + fixed_target_actions + suffix_actions
        )
        self.attack_action_config["description"] = (
            "SMAC-style attack actions with one permanent slot per configured "
            "destructible blue combat unit."
        )

        fields = [dict(field) for field in self.attack_state_config.get("fields", [])]
        red_cfg = self.config.get("red", {})
        recon_names = red_cfg.get("recon_aircraft", [])
        attack_names = red_cfg.get("attack_aircraft", [])
        ground_names = red_cfg.get("ground_forces", [])

        # Collect template entries and their positions.
        recon_tmpl = []
        atk_tmpl = []
        gnd_tmpl = []
        tgt_tmpl = []
        for idx, field in enumerate(fields):
            name = str(field.get("name", ""))
            parts = name.split("_", 3)
            if not (len(parts) == 4 and parts[2].isdigit()):
                continue
            if parts[0] == "friendly" and parts[1] == "recon" and int(parts[2]) == 1:
                recon_tmpl.append((parts[3], field))
            elif parts[0] == "friendly" and parts[1] == "attack" and int(parts[2]) == 1:
                atk_tmpl.append((parts[3], field))
            elif parts[0] == "friendly" and parts[1] == "ground" and int(parts[2]) == 1:
                gnd_tmpl.append((parts[3], field))
            elif parts[0] == "target" and parts[1] == "slot" and int(parts[2]) == 1:
                tgt_tmpl.append((parts[3], field))

        if not tgt_tmpl:
            raise ValueError("attack_state_fields.json has no target_slot_1 template")

        # Start with template entries in their original positions, then expand each block.
        expanded = list(fields)
        _expand_field_block(expanded, "friendly_recon_1_", recon_tmpl,
                            _expand_friendly_slots("recon", recon_names, recon_tmpl))
        _expand_field_block(expanded, "friendly_attack_1_", atk_tmpl,
                            _expand_friendly_slots("attack", attack_names, atk_tmpl))
        _expand_field_block(expanded, "friendly_ground_1_", gnd_tmpl,
                            _expand_friendly_slots("ground", ground_names, gnd_tmpl))
        _expand_field_block(expanded, "target_slot_1_", tgt_tmpl,
                            _expand_target_slots(target_names, tgt_tmpl, self._configured_blue_target_role))

        self.attack_state_config["fields"] = expanded

    def _configured_blue_target_role(self, target_name):
        blue = self.config.get("blue", {})
        role_by_key = {
            "attack_aircraft": "attack_aircraft",
            "ground_forces": "ground_force",
            "radars": "radar",
            "sams": "sam",
        }
        for key, role in role_by_key.items():
            if target_name in blue.get(key, []):
                return role
        return ""

    def _configured_ground_target_names(self):
        """Return the permanent target order used by every ground actor."""
        blue = self.config.get("blue", {})
        names = []
        for key in ("ground_forces", "radars", "sams"):
            for name in blue.get(key, []):
                if name not in names:
                    names.append(str(name))
        return names

    def _configure_fixed_ground_schema(self):
        """Create one permanent ground attack action and observation slot per target."""
        target_names = list(self.ground_fixed_target_names)
        configured_actions = sorted(
            (dict(action) for action in self.ground_action_config.get("actions", [])),
            key=lambda item: int(item.get("id", 0)),
        )
        fire_actions = [a for a in configured_actions if a.get("afsim_task") == "GROUND_TARGET_SLOT"]
        target_action_start = min((int(a.get("id", 0)) for a in fire_actions), default=10)
        prefix_actions = [
            a for a in configured_actions
            if a.get("afsim_task") != "GROUND_TARGET_SLOT" and int(a.get("id", 0)) < target_action_start
        ]
        suffix_actions = [
            a for a in configured_actions
            if a.get("afsim_task") != "GROUND_TARGET_SLOT" and int(a.get("id", 0)) >= target_action_start
        ]
        fixed_actions = []
        for slot, target_name in enumerate(target_names):
            fixed_actions.append({
                "id": target_action_start + slot,
                "name": "ATTACK_TARGET_{0}".format(slot + 1),
                "description": "Attack permanent ground target slot {0} ({1}).".format(slot + 1, target_name),
                "afsim_task": "GROUND_TARGET_SLOT",
                "target_slot": slot,
                "target_name": target_name,
            })
        next_action_id = target_action_start + len(fixed_actions)
        for offset, action in enumerate(suffix_actions):
            action["id"] = next_action_id + offset
        self.ground_action_config["target_slot_count"] = len(target_names)
        self.ground_action_config["fixed_target_names"] = target_names
        self.ground_action_config["actions"] = prefix_actions + fixed_actions + suffix_actions

        fields = [dict(field) for field in self.ground_state_config.get("fields", [])]
        target_fields = []
        non_target_fields = []
        first_target_index = None
        for field in fields:
            name = str(field.get("name", ""))
            parts = name.split("_", 3)
            is_target_field = len(parts) == 4 and parts[:2] == ["target", "slot"] and parts[2].isdigit()
            if is_target_field:
                if first_target_index is None:
                    first_target_index = len(non_target_fields)
                if int(parts[2]) == 1:
                    target_fields.append((parts[3], field))
            else:
                non_target_fields.append(field)
        if first_target_index is None or not target_fields:
            raise ValueError("ground_state_fields.json has no target_slot_1 template")
        expanded_fields = []
        for slot, target_name in enumerate(target_names, start=1):
            for suffix, template in target_fields:
                field = dict(template)
                field["name"] = "target_slot_{0}_{1}".format(slot, suffix)
                field["description"] = "Fixed ground target {0} ({1}): {2}".format(
                    slot, target_name, str(template.get("description", ""))
                )
                expanded_fields.append(field)
        self.ground_state_config["fields"] = (
            non_target_fields[:first_target_index]
            + expanded_fields
            + non_target_fields[first_target_index:]
        )

    def _open_socket(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(self.local_address)
        self.sock.settimeout(self.message_timeout_seconds)

    def close(self):
        if self.warlock_process and self.warlock_process.poll() is None:
            self.warlock_process.terminate()
            try:
                self.warlock_process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.warlock_process.kill()
        self.warlock_process = None
        if self.warlock_log_handle:
            self.warlock_log_handle.close()
            self.warlock_log_handle = None
        if self.sock:
            self.sock.close()
            self.sock = None

    def start_warlock(self):
        scenario_cfg = self.config.get("scenario", {})
        warlock_path = scenario_cfg.get("warlock_path")
        scenario_dir = scenario_cfg.get("scenario_dir")
        scenario_file = scenario_cfg.get("scenario_file")
        warlock_args = scenario_cfg.get("warlock_args", [])
        if not warlock_path or not scenario_dir or not scenario_file:
            raise ValueError("warlock_path, scenario_dir, and scenario_file must be configured")
        if self.warlock_process and self.warlock_process.poll() is None:
            return self.warlock_process
        cmd = [warlock_path] + list(warlock_args) + [scenario_file]
        log_path = scenario_cfg.get("warlock_log_path")
        if log_path:
            os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
            self.warlock_log_handle = open(log_path, "w", encoding="utf-8", buffering=1)
        self.warlock_process = subprocess.Popen(
            cmd,
            cwd=scenario_dir,
            stdout=self.warlock_log_handle or subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return self.warlock_process

    def discard_pending_udp_messages(self):
        """Discard datagrams left by a stopped scenario without handling them."""
        if self.sock is None:
            return 0
        previous_timeout = self.sock.gettimeout()
        discarded = 0
        try:
            self.sock.setblocking(False)
            while True:
                try:
                    self.sock.recvfrom(65535)
                    discarded += 1
                except (BlockingIOError, socket.timeout):
                    break
                except OSError:
                    break
        finally:
            self.sock.settimeout(previous_timeout)
        return discarded

    def prepare_for_scenario_restart(self):
        """Forget registrations from a stopped live scenario while retaining the socket."""
        if not self.live_mode:
            raise RuntimeError("scenario restart preparation requires a live environment")
        self.remote_addr = None
        self.last_platform_state_wall_time = 0.0
        self.last_decision_ready_wall_time = 0.0
        self.platform_state_sequence = 0
        # A restarted scenario resets its simulation clock to zero. Keeping
        # the previous episode's boundary timestamp would make every new
        # DecisionReady look like a stale UDP duplicate.
        self.native_decision_ready = False
        self.native_decision_ready_time = 0.0
        self.native_decision_ready_seen = False
        self._native_restart_pending = False
        self._native_restart_boundary_time = None
        self._native_restart_last_send = 0.0
        self.step_count = 0
        self._next_decision_target_sim_time = None
        self._decision_window_id = 0
        self._decision_window_sim_time = 0.0
        # A new Mission must not inherit a terminal verdict from the prior one.
        # Otherwise is_done() returns True at T=0 and produces an empty rollout.
        self.episode_result = ""
        self.episode_done_reason = "none"
        self.final_score_raw = 0.0
        self.final_score_norm = 0.0
        self.final_score_unit_count = 0
        self.final_score_units = []
        self.final_score_sim_time = 0.0
        self.final_score_settled = False
        self.platforms = {}
        self._build_platform_registry()
        self._apply_stage_initial_state()
        self.detected_targets.clear()
        self.attack_local_detections.clear()
        self.attack_target_snapshots.clear()
        self._initialize_attack_target_priors()
        self.enemy_track_memory.clear()
        self.commander_contact_slots.clear()
        self.ground_detected_targets.clear()
        self.last_reward_events.clear()
        self.last_reward_details.clear()
    def _send_native_restart(self, boundary_time=None):
        """Resume one known native pause and arm loss-safe retransmission."""
        if boundary_time is None:
            boundary_time = float(self.native_decision_ready_time)
        self._native_restart_pending = True
        self._native_restart_boundary_time = float(boundary_time)
        self._native_restart_last_send = time.monotonic()
        self._send({"MsgType": "SimRestart"})

    def verify_native_decision_pause(self, timeout=None):
        """Prove that the live scenario produces a boundary after SimRestart.

        Seeing only the startup DecisionReady packet is insufficient: a scenario
        without ``decision_pause_interval`` registers every platform normally,
        then runs forever while Python waits for a second boundary.
        """
        if not self.native_decision_pause_control:
            return True
        if not self.native_decision_ready_seen:
            raise RuntimeError(
                "native decision pause was not armed at the startup boundary"
            )
        previous_boundary = float(self.native_decision_ready_time)
        start_sim_time = float(self._current_sim_time())
        self.native_decision_ready = False
        self._send_native_restart(previous_boundary)
        self._drain_messages(
            timeout=float(
                self.native_decision_pause_timeout if timeout is None else timeout
            ),
            until_decision_ready=True,
        )
        if (
            self.native_decision_ready
            and float(self.native_decision_ready_time) > previous_boundary + 1.0e-6
        ):
            return True
        raise RuntimeError(
            "AFSIM native decision pause handshake failed: startup boundary "
            "{:.3f}, no newer DecisionReady after SimRestart, last_sim_time "
            "{:.3f}. Verify the worker scenario contains a non-zero "
            "decision_pause_interval matching the policy interval.".format(
                previous_boundary, max(start_sim_time, self._current_sim_time())
            )
        )
    def wait_for_platforms(self, names=None, timeout=30.0):
        names = names or list(self.platforms.keys())
        deadline = time.time() + timeout
        while time.time() < deadline:
            self._drain_messages(timeout=0.5)
            missing = [
                name for name in names
                if self.platforms.get(name)
                and self.platforms[name].alive
                and self.platforms[name].platform_id is None
            ]
            if not missing:
                return True
        return False
    def _add_platforms(self, side, role, names):
        for name in names:
            max_hp = float(self.combat_model_config.get("max_hp_by_role", {}).get(role, 0.0))
            indestructible = role in self.combat_model_config.get("indestructible_roles", [])
            self.platforms[name] = PlatformState(
                name=name,
                role=role,
                side=side,
                at_home=side == "red" and role in ("recon_aircraft", "attack_aircraft"),
                max_hp=max_hp,
                current_hp=max_hp,
                indestructible=indestructible,
            )

    def _reset_platform_combat_state(self, platform):
        max_hp = float(self.combat_model_config.get("max_hp_by_role", {}).get(platform.role, 0.0))
        platform.max_hp = max_hp
        platform.current_hp = max_hp
        platform.indestructible = platform.role in self.combat_model_config.get("indestructible_roles", [])
        platform.last_damage_time = -1.0
        platform.last_attack_time = -1.0
        platform.fire_cooldown_until = 0.0
        platform.combat_lock_until = 0.0
        platform.rearm_complete_at = 0.0

    def _apply_platform_combat_message(self, platform, msg):
        previous_alive = bool(platform.alive)
        previous_hp = float(platform.current_hp)
        external_task = msg.get("ExternalTask", msg.get("EXTERNAL_TASK"))
        if external_task is not None:
            platform.task = str(external_task)
        task_status = msg.get("TaskStatus", msg.get("TASK_STATUS"))
        if task_status is not None:
            previous_status = str(platform.task_status or "")
            next_status = str(task_status)
            if "REARMING" in next_status.upper():
                explicit_complete_at = msg.get(
                    "RearmCompleteTime", msg.get("REARM_COMPLETE_TIME")
                )
                if explicit_complete_at is not None:
                    platform.rearm_complete_at = max(
                        0.0, float(explicit_complete_at)
                    )
                elif "REARMING" not in previous_status.upper():
                    duration = float(
                        self.config.get("scenario", {}).get(
                            "attack_rearm_duration_seconds", 600.0
                        )
                    )
                    message_time = float(
                        msg.get("WallTime", self._current_sim_time())
                    )
                    platform.rearm_complete_at = message_time + duration
            elif "REARMING" not in next_status.upper():
                platform.rearm_complete_at = 0.0
            platform.task_status = next_status
        has_current_hp = "CurrentHP" in msg
        if "MaxHP" in msg:
            platform.max_hp = max(0.0, float(msg.get("MaxHP", platform.max_hp)))
        if has_current_hp:
            platform.current_hp = max(
                0.0,
                min(
                    platform.max_hp,
                    float(msg.get("CurrentHP", previous_hp)),
                ),
            )
        if "Indestructible" in msg:
            platform.indestructible = bool(msg.get("Indestructible"))
        if "LastDamageTime" in msg:
            platform.last_damage_time = float(msg.get("LastDamageTime", -1.0))
        if "LastAttackTime" in msg:
            platform.last_attack_time = float(msg.get("LastAttackTime", -1.0))
        if "FireCooldownUntil" in msg:
            platform.fire_cooldown_until = float(msg.get("FireCooldownUntil", 0.0))
        if "CombatLockUntil" in msg:
            platform.combat_lock_until = float(msg.get("CombatLockUntil", 0.0))
        blue_detection_keys = (
            "DetectedByBlue", "DETECTED_BY_BLUE",
            "DetectedByBlueAir", "DETECTED_BY_BLUE_AIR",
            "DetectedByBlueGround", "DETECTED_BY_BLUE_GROUND",
        )
        for key in blue_detection_keys:
            if key not in msg:
                continue
            if bool(msg.get(key)):
                self.blue_detected_targets.add(platform.name)
            else:
                self.blue_detected_targets.discard(platform.name)
            break
        if "Alive" in msg and not platform.indestructible:
            platform.alive = bool(msg.get("Alive"))
            # Some AFSIM status messages report a direct kill only as
            # Alive=false. Normalize that transition to HP=0 so durability
            # rewards do not silently disappear for one-hit kills.
            if not platform.alive and not has_current_hp:
                platform.current_hp = 0.0
        elif has_current_hp and not platform.indestructible and platform.max_hp > 0.0:
            platform.alive = platform.current_hp > 0.0
        hp_lost = max(0.0, previous_hp - float(platform.current_hp))
        if hp_lost > 0.0:
            damage_event = {
                "type": (
                    "damage_received"
                    if platform.side == "red"
                    else "damage_dealt"
                ),
                "platform": platform.name,
                "side": platform.side,
                "role": platform.role,
                "amount": hp_lost,
                "max_hp": float(platform.max_hp),
                "reward": 0.0,
            }
            shooter = self.attack_last_shooter_by_target.get(platform.name, "")
            if shooter:
                self.register_target_reward_contributor(
                    platform.name, shooter, "attack")
                damage_event["actor"] = shooter
                weapon = str(self.attack_last_weapon_by_target.get(platform.name, "")).lower()
                shooter_platform = self.platforms.get(shooter)
                is_agm_hit = (
                    weapon == "agm"
                    or "agm" in weapon
                    or (
                        shooter_platform is not None
                        and shooter_platform.role == "attack_aircraft"
                        and platform.side == "blue"
                        and platform.role != "attack_aircraft"
                    )
                )
                if is_agm_hit and self.debug_combat_events:
                    print(
                        "[AGM_HIT] shooter={0} target={1} damage={2:.1f} hp={3:.1f}".format(
                            shooter, platform.name, hp_lost, float(platform.current_hp)
                        ),
                        flush=True,
                    )
            self.last_reward_events.append(damage_event)
        if previous_alive and not platform.alive:
            shooter = self.attack_last_shooter_by_target.pop(platform.name, "")
            if shooter:
                self.register_target_reward_contributor(
                    platform.name, shooter, "attack")
                weapon = str(self.attack_last_weapon_by_target.get(platform.name, "")).lower()
                shooter_platform = self.platforms.get(shooter)
                is_agm_hit = (
                    weapon == "agm"
                    or "agm" in weapon
                    or (
                        shooter_platform is not None
                        and shooter_platform.role == "attack_aircraft"
                        and platform.side == "blue"
                        and platform.role != "attack_aircraft"
                    )
                )
                if is_agm_hit and hp_lost <= 0.0 and self.debug_combat_events:
                    print(
                        "[AGM_HIT] shooter={0} target={1} destroyed".format(
                            shooter, platform.name
                        ),
                        flush=True,
                    )
            self.last_reward_events.append({
                "type": "target_destroyed",
                "platform": platform.name,
                "actor": shooter,
                "side": platform.side,
                "role": platform.role,
                "sim_time": float(self._current_sim_time()),
                # Team kill credit is already computed from destroyed-count
                # deltas. This event exists only for local credit assignment.
                "reward": 0.0,
            })
        if platform.role == "attack_aircraft" and ("CurrentAAM" in msg or "CurrentAGM" in msg):
            ammo = self.attack_ammo.setdefault(platform.name, {"fox3": 1, "agm": 1})
            if "CurrentAAM" in msg:
                ammo["fox3"] = max(0, int(msg.get("CurrentAAM", ammo.get("fox3", 0))))
            if "CurrentAGM" in msg:
                ammo["agm"] = max(0, int(msg.get("CurrentAGM", ammo.get("agm", 0))))
        if platform.role == "ground_force" and "CurrentGroundAmmo" in msg:
            ammo = self.ground_ammo.setdefault(platform.name, {"ground_fire": 15})
            ammo["ground_fire"] = max(
                0,
                int(msg.get("CurrentGroundAmmo", ammo.get("ground_fire", 0))),
            )
        if platform.role == "sam" and "CurrentSAMAmmo" in msg:
            self.sam_ammo[platform.name] = max(
                0,
                int(msg.get("CurrentSAMAmmo", self.sam_ammo.get(platform.name, 0))),
            )
        self._sync_platform_combat_caches(platform)

    def _sync_platform_combat_caches(self, platform):
        """Propagate authoritative platform HP into every state-facing cache."""
        name = platform.name
        alive = bool(platform.alive)
        hp_state = {
            "CurrentHP": float(platform.current_hp),
            "MaxHP": float(platform.max_hp),
            "alive": alive,
        }
        detected = self.detected_targets.get(name)
        if detected is not None:
            detected.update(hp_state)
        memory = self.enemy_track_memory.get(name)
        if memory is not None:
            memory.update(hp_state)
            if not alive:
                memory["CurrentlyDetected"] = False
                for observer in memory.get("Observers", {}).values():
                    observer["CurrentlyDetected"] = False
        for contacts in self.attack_local_detections.values():
            contact = contacts.get(name)
            if contact is not None:
                contact.update(hp_state)
        ground_detection = self.ground_detected_targets.get(name)
        if ground_detection is not None:
            ground_detection.update(hp_state)
        for group in self.attack_controller.active_groups.values():
            if group.target_name == name:
                group.target.update({
                    "CurrentHP": float(platform.current_hp),
                    "MaxHP": float(platform.max_hp),
                    "hp_norm": self._hp_norm(platform),
                    "alive": alive,
                })
        if not alive:
            self.attack_target_reservations.pop(name, None)
            self.ground_target_reservations.pop(name, None)

    @staticmethod
    def _hp_norm(platform):
        if platform is None or platform.max_hp <= 0.0:
            return 1.0
        return max(0.0, min(1.0, platform.current_hp / platform.max_hp))

    def _combat_remaining_norm(self, until, duration):
        remaining = max(0.0, float(until) - self._current_sim_time())
        return min(1.0, remaining / max(1.0, float(duration)))

    def _initial_attack_ammo(self):
        attack_names = list(
            self.config.get("red", {}).get("attack_aircraft", [])
        ) + list(
            self.config.get("blue", {}).get("attack_aircraft", [])
        )
        return {
            name: {"fox3": 1, "agm": 1}
            for name in attack_names
        }

    def _initial_sam_ammo(self):
        return {
            name: 10
            for name in self.config.get("blue", {}).get("sams", [])
        }

    def _initial_carrier_ammo_stock(self):
        scenario_cfg = self.config.get("scenario", {})
        return {
            "fox3": max(0, int(scenario_cfg.get("carrier_initial_aam_stock", 30))),
            "agm": max(0, int(scenario_cfg.get("carrier_initial_agm_stock", 50))),
        }

    def _initial_landing_cargo(self):
        cargo = {
            name: {"has_army": True, "army_landed": False}
            for name in self.config.get("red", {}).get("transports", [])
        }
        overrides = self.config.get("stage_initial_state", {}).get("landing_cargo", {})
        for name, values in overrides.items():
            if name in cargo:
                cargo[name].update({
                    "has_army": bool(values.get("has_army", cargo[name]["has_army"])),
                    "army_landed": bool(values.get("army_landed", cargo[name]["army_landed"])),
                })
        return cargo

    def _initial_ground_status(self):
        status = {
            name: {"on_ship": True, "landed": False, "transport": self.config.get("red", {}).get("ground_transport_map", {}).get(name, "")}
            for name in self.config.get("red", {}).get("ground_forces", [])
        }
        overrides = self.config.get("stage_initial_state", {}).get("ground_status", {})
        for name, values in overrides.items():
            if name in status:
                status[name].update({
                    "on_ship": bool(values.get("on_ship", status[name]["on_ship"])),
                    "landed": bool(values.get("landed", status[name]["landed"])),
                    "transport": str(values.get("transport", status[name]["transport"])),
                })
        return status

    def _apply_stage_initial_state(self):
        stage_state = self.config.get("stage_initial_state", {})
        platform_states = {}
        destroyed_platforms = set(stage_state.get("destroyed_platforms", []))
        platform_states_file = str(stage_state.get("platform_states_file", "") or "")
        if platform_states_file:
            if not os.path.isabs(platform_states_file):
                platform_states_file = os.path.join(
                    os.path.dirname(os.path.abspath(self.config_path)),
                    platform_states_file,
                )
            loaded_states = self._load_config(platform_states_file)
            loaded_platform_states = loaded_states.get("platforms", loaded_states)
            platform_states.update(loaded_platform_states)
            if bool(loaded_states.get("destroyed", False)):
                destroyed_platforms.update(loaded_platform_states)
        platform_states.update(stage_state.get("platform_states", {}))
        for name, values in platform_states.items():
            platform = self.platforms.get(name)
            if platform is None or not isinstance(values, dict):
                continue
            for field_name in ("lat", "lon", "alt", "heading", "speed"):
                if field_name in values:
                    setattr(platform, field_name, float(values[field_name]))
        for name in destroyed_platforms:
            platform = self.platforms.get(name)
            if platform is not None:
                platform.alive = False
                if not platform.indestructible:
                    platform.current_hp = 0.0
                platform.task = "DEFEATED"
                platform.task_status = "IDLE"
        self.task_flags["ground_landed"] = any(
            values.get("landed", False) and not values.get("on_ship", False)
            for values in self.ground_status.values()
        )

    def _initial_ground_ammo(self):
        ground_names = list(
            self.config.get("red", {}).get("ground_forces", [])
        ) + list(
            self.config.get("blue", {}).get("ground_forces", [])
        )
        return {
            name: {"ground_fire": 15}
            for name in ground_names
        }

    def _build_platform_registry(self):
        red = self.config.get("red", {})
        blue = self.config.get("blue", {})
        self._add_platforms("red", "carrier", red.get("carrier", []))
        self._add_platforms("red", "recon_aircraft", red.get("recon_aircraft", []))
        self._add_platforms("red", "attack_aircraft", red.get("attack_aircraft", []))
        self._add_platforms("red", "transport", red.get("transports", []))
        self._add_platforms("red", "ground_force", red.get("ground_forces", []))
        self._add_platforms("blue", "base", blue.get("base", []))
        self._add_platforms("blue", "attack_aircraft", blue.get("attack_aircraft", []))
        self._add_platforms("blue", "ground_force", blue.get("ground_forces", []))
        self._add_platforms("blue", "radar", blue.get("radars", []))
        self._add_platforms("blue", "sam", blue.get("sams", []))        # udpnet MoveUpdate packets contain only the numeric platform id.
        # Keep registration order as a late-join fallback when PlatFormAdd was missed.
        self._platform_name_by_expected_id = {
            index + 1: name for index, name in enumerate(self.platforms)
        }

    def _platform_for_message(self, msg):
        name = str(msg.get("PlatformName", ""))
        platform_id = msg.get("PlatformId")
        if name:
            if name not in self.platforms:
                return None
            platform = self.platforms[name]
            platform.platform_id = platform_id
            return platform
        return self._platform_for_id(platform_id)

    def _platform_for_id(self, platform_id):
        """Resolve a platform id, including after a late UDP join."""
        for platform in self.platforms.values():
            if platform.platform_id == platform_id:
                return platform
        try:
            expected_name = self._platform_name_by_expected_id.get(int(platform_id))
        except (TypeError, ValueError):
            expected_name = None
        if not expected_name:
            return None
        platform = self.platforms.get(expected_name)
        if platform is None or platform.platform_id not in (None, platform_id):
            return None
        platform.platform_id = platform_id
        self.last_reward_events.append({
            "type": "late_join_platform_bound",
            "reward": 0.0,
            "platform": platform.name,
            "platform_id": platform_id,
        })
        return platform

    def _build_actions(self):
        """High-level actions were removed; rules create all missions."""
        self._actions = []
        self.action_names = []

    def reward_team_id_for_entity(self, agent_type, entity_name):
        """Return the fixed-team id used only for reward target lookup."""
        controller = {"recon": self.recon_controller, "attack": self.attack_controller}.get(str(agent_type))
        teams = (
            list(getattr(controller, "fixed_recon_teams", []))
            if agent_type == "recon"
            else list(getattr(controller, "fixed_attack_teams", []))
        )
        for index, members in enumerate(teams):
            if str(entity_name) in members:
                return "{0}_fixed_team_{1}".format(agent_type, index + 1)
        return ""

    def _reward_target_team_spec(self, agent_type, fixed_team_id):
        agent_type = str(agent_type or "").lower()
        team_id = str(fixed_team_id or "")
        team_bucket = self.reward_target_priority_config.get("teams", {}).get(agent_type, {})
        team_spec = team_bucket.get(team_id, {})
        if not team_spec:
            return {"description": "", "targets": []}
        if isinstance(team_spec, list):
            return {
                "description": "",
                "targets": list(team_spec),
            }
        team_spec = dict(team_spec)
        targets = team_spec.get("targets")
        if targets is None:
            targets = team_spec.get("items", [])
        if targets is None:
            targets = []
        return {
            "description": str(team_spec.get("description", "")),
            "targets": list(targets),
        }
    def get_reward_target_for_team(self, agent_type, fixed_team_id):
        """Return a living blue platform's live position for reward shaping only.

        This deliberately reads ``self.platforms`` rather than reconnaissance
        track memory. Action execution may navigate against a shared track's
        last reported position, whereas reward progress is evaluated against
        the target's current simulated position.
        """
        spec = self._reward_target_team_spec(agent_type, fixed_team_id)
        agent_type = str(agent_type or "").lower()
        detected_delta = float(self.reward_target_priority_config.get("recon_detected_priority_delta", -1000.0))
        candidates = []
        for order, entry in enumerate(spec.get("targets", [])):
            target_name = str(entry.get("target", ""))
            target = self.platforms.get(target_name)
            if target is None or target.side != "blue" or not target.alive:
                continue
            effective_priority = float(entry.get("priority", 0.0))
            if agent_type == "recon" and (target_name in self.detected_targets or bool(target.detected)):
                effective_priority += detected_delta
            candidates.append((effective_priority, -order, target_name, target, entry))
        if not candidates:
            return None
        effective_priority, _neg_order, target_name, target, entry = max(candidates)
        return {
            "name": target_name,
            "lat": float(target.lat),
            "lon": float(target.lon),
            "alt": float(target.alt),
            "base_priority": float(entry.get("priority", 0.0)),
            "effective_priority": float(effective_priority),
            "alive": bool(target.alive),
            "team_description": spec.get("description", ""),
        }

    def reward_movement_scale(self, agent_type=None):
        """Return an agent-specific movement shaping scale."""
        agent_type = str(agent_type or "").lower()
        specific_key = "{0}_movement_reward_scale".format(agent_type)
        return float(self.reward_target_priority_config.get(
            specific_key,
            self.reward_target_priority_config.get("movement_reward_scale", 1.0),
        ))

    def _blue_target_names(self):
        blue = self.config.get("blue", {})
        names = []
        for key in ["attack_aircraft", "ground_forces", "radars", "sams"]:
            names.extend(blue.get(key, []))
        return names

    def _empty_last_command(self):
        return {
            "action_id": -1,
            "action_name": "",
            "kind": "NONE",
            "accepted": False,
            "rejected": False,
            "activated_count": 0,
            "reject_reason": "",
            "step": -1,
            "reward_norm": 0.5,
            "progress_delta_norm": 0.0,
            "requested_count": 0,
        }

    def _begin_commander_command(self, action_id, action):
        self.last_command = self._empty_last_command()
        self.last_command.update({
            "action_id": int(action_id),
            "action_name": action.name if action else "INVALID",
            "kind": action.kind if action else "INVALID",
            "step": int(self.step_count),
            "requested_count": int(max(0, getattr(action, "group_size", 0))) if action else 0,
        })

    def _record_command_result(self, accepted=False, reject_reason="", activated_count=0):
        self.last_command["accepted"] = bool(accepted)
        self.last_command["rejected"] = not bool(accepted)
        self.last_command["reject_reason"] = str(reject_reason or "")
        self.last_command["activated_count"] = int(max(0, activated_count))

    def _update_last_command_outcome(self, reward, before_progress):
        kind = str(self.last_command.get("kind", "")).upper()
        after_progress = self._commander_task_complete_ratio(kind)
        before_value = float(before_progress.get(kind, 0.0)) if isinstance(before_progress, dict) else 0.0
        self.last_command["progress_delta_norm"] = max(0.0, min(1.0, after_progress - before_value))
        self.last_command["reward_norm"] = max(0.0, min(1.0, 0.5 + float(reward) / 200.0))

    def _commander_command_denominator(self, kind, requested_count=0):
        if int(requested_count) > 0:
            return int(requested_count)
        kind = str(kind).upper()
        scenario_cfg = self.config.get("scenario", {})
        if kind == "RECON":
            return max(1, int(scenario_cfg.get("recon_group_size", 3)))
        if kind == "ATTACK":
            return max(1, int(scenario_cfg.get("attack_group_size", 3)))
        if kind == "LANDING":
            return max(1, int(scenario_cfg.get("landing_group_size", 1)))
        if kind == "GROUND":
            return max(1, int(scenario_cfg.get("ground_group_size", 3)))
        return 1

    def _commander_task_complete_ratios(self):
        return {
            "WAIT": 0.0,
            "INVALID": 0.0,
            "RECON": self._commander_task_complete_ratio("RECON"),
            "ATTACK": self._commander_task_complete_ratio("ATTACK"),
            "LANDING": self._commander_task_complete_ratio("LANDING"),
            "GROUND": self._commander_task_complete_ratio("GROUND"),
        }

    def _commander_task_complete_ratio(self, kind):
        kind = str(kind).upper()
        if kind == "RECON":
            return self._commander_recon_coverage_ratio()
        if kind == "ATTACK":
            groups = list(getattr(self.attack_controller, "active_groups", {}).values())
            if not groups:
                return 0.0
            progress = []
            for group in groups:
                if getattr(group, "mission_area", {}):
                    for platform in group.platforms:
                        if not platform.alive:
                            progress.append(1.0)
                            continue
                        ammo = self.attack_ammo.get(platform.name, {"fox3": 1, "agm": 1})
                        remaining = min(2.0, max(0.0, float(ammo.get("fox3", 0)) + float(ammo.get("agm", 0))))
                        progress.append(1.0 - remaining / 2.0)
                else:
                    target = self._attack_target_info(group.target_name) or group.target
                    progress.append(1.0 if target and not bool(target.get("alive", True)) else 0.0)
            return min(1.0, sum(progress) / float(max(1, len(progress))))
        if kind == "LANDING":
            groups = list(getattr(self.landing_controller, "active_groups", {}).values())
            if not groups:
                return 0.0
            values = []
            for group in groups:
                count = 0
                for platform in group.platforms:
                    cargo = self.landing_cargo.get(platform.name, {"army_landed": False})
                    if cargo.get("army_landed", False):
                        count += 1
                values.append(count / float(max(1, len(group.platforms))))
            return min(1.0, sum(values) / float(max(1, len(values))))
        if kind == "GROUND":
            groups = list(getattr(self.ground_controller, "active_groups", {}).values())
            if not groups:
                return 0.0
            values = []
            for group in groups:
                count = 0
                for platform in group.platforms:
                    obs = self._build_ground_unit_obs(platform, group)
                    if obs.get("task_complete", 0.0) > 0.0:
                        count += 1
                values.append(count / float(max(1, len(group.platforms))))
            return min(1.0, sum(values) / float(max(1, len(values))))
        return 0.0

    def _last_command_state_values(self):
        command = getattr(self, "last_command", self._empty_last_command())
        kind = str(command.get("kind", "NONE")).upper()
        reason = str(command.get("reject_reason", "")).lower()
        age = 0.0
        if int(command.get("step", -1)) >= 0 and kind != "WAIT":
            age = min(1.0, (self.step_count - int(command.get("step", 0))) / float(max(1, self.max_steps)))
        activated_den = self._commander_command_denominator(kind, command.get("requested_count", 0))
        return {
            "last_command_type_wait": 1.0 if kind == "WAIT" else 0.0,
            "last_command_type_recon": 1.0 if kind == "RECON" else 0.0,
            "last_command_type_attack": 1.0 if kind == "ATTACK" else 0.0,
            "last_command_type_landing": 1.0 if kind == "LANDING" else 0.0,
            "last_command_type_ground": 1.0 if kind == "GROUND" else 0.0,
            "last_command_accepted": 1.0 if command.get("accepted", False) else 0.0,
            "last_command_rejected": 1.0 if command.get("rejected", False) else 0.0,
            "last_command_activated_count_norm": min(1.0, float(command.get("activated_count", 0)) / float(max(1, activated_den))),
            "last_command_reject_reason_busy": 1.0 if any(token in reason for token in ["busy", "available", "not_enough", "no_"]) else 0.0,
            "last_command_reject_reason_invalid": 1.0 if "invalid" in reason or "masked" in reason else 0.0,
            "last_command_reject_reason_unknown_target": 1.0 if "unknown_target" in reason or "target_not_known" in reason else 0.0,
            "last_command_reject_reason_platform_not_ready": 1.0 if "platform_not_ready" in reason else 0.0,
            "last_command_reject_reason_udp": 1.0 if "udp" in reason else 0.0,
            "last_command_age_norm": age,
            "last_command_reward_norm": float(command.get("reward_norm", 0.5)),
            "last_command_progress_delta_norm": float(command.get("progress_delta_norm", 0.0)),
            "current_recon_complete_ratio": self._commander_task_complete_ratio("RECON"),
            "current_attack_complete_ratio": self._commander_task_complete_ratio("ATTACK"),
            "current_landing_complete_ratio": self._commander_task_complete_ratio("LANDING"),
            "current_ground_complete_ratio": self._commander_task_complete_ratio("GROUND"),
        }

    def reset(self):
        self.step_count = 0
        self._next_decision_target_sim_time = None
        self.last_drain_metadata = {}
        self.episode_id += 1
        self.episode_result = ""
        self.episode_done_reason = "none"
        self.final_score_raw = 0.0
        self.final_score_norm = 0.0
        self.final_score_unit_count = 0
        self.final_score_sim_time = 0.0
        self.final_score_units = []
        self.final_score_settled = False
        self.capture_condition_since = None
        for controller in (
            self.recon_controller,
            self.attack_controller,
            self.landing_controller,
            self.ground_controller,
        ):
            controller.active_groups.clear()
            controller.next_group_index = 1
        self.recon_area_coverage = {
            str(self._normalize_recon_area(area).get("name", "")): set()
            for area in self.recon_areas
        }
        self.recon_area_last_observed_time = {name: None for name in self.recon_area_coverage}
        self.detected_targets.clear()
        self.attack_local_detections.clear()
        self.attack_target_snapshots.clear()
        self._initialize_attack_target_priors()
        self.enemy_track_memory.clear()
        self.commander_contact_slots.clear()
        self.ground_detected_targets.clear()
        self.blue_detected_targets.clear()
        self.last_reward_events.clear()
        self.task_flags = {
            "landing_executed": False,
            "ground_landed": False,
            "current_recon": "",
            "current_attack": "",
        }
        self.last_command = self._empty_last_command()
        if not self.live_mode:
            for plat in self.platforms.values():
                self._reset_platform_combat_state(plat)
                plat.alive = True
                plat.task = "PARKED"
                plat.task_status = "IDLE"
                plat.task_assigned = False
                plat.detected = False
                plat.at_home = plat.side == "red" and plat.role in ["recon_aircraft", "attack_aircraft"]
        for plat in self.platforms.values():
            if plat.role == "recon_aircraft" and (not self.live_mode or plat.task in ("PARKED", "")):
                plat.task_assigned = False
        self.attack_ammo = self._initial_attack_ammo()
        self.sam_ammo = self._initial_sam_ammo()
        self.carrier_ammo_stock = self._initial_carrier_ammo_stock()
        self.attack_target_reservations = {}
        self.attack_last_shooter_by_target = {}
        self.attack_last_weapon_by_target = {}
        self.target_reward_contributors = {}
        self.ground_target_reservations = {}
        self.pending_attack_fire_commands = {}
        self.pending_attack_approaches = {}
        self.pending_attack_rejoins = {}

        self.pending_attack_returns = {}
        self.landing_cargo = self._initial_landing_cargo()
        self.pending_landing_unloads = {}
        self.ground_status = self._initial_ground_status()
        self.ground_ammo = self._initial_ground_ammo()

        reset_drain_seconds = 2.0
        if self.live_mode:
            reset_drain_seconds = float(
                self.config.get("scenario", {}).get(
                    "live_reset_drain_seconds",
                    min(0.1, max(0.0, float(self.decision_seconds))),
                )
            )
        self._drain_messages(timeout=max(0.0, reset_drain_seconds))
        self._apply_stage_initial_state()
        self.reward_manager.reset(self)
        self.last_reward_details = []
        self.initialize_bottom_teams()
        mask = self.get_action_mask()
        return (self.get_observation(), mask)

    def initialize_bottom_teams(self):
        """Ensure every air unit has a stable team context.

        Team creation does not assign a mission, area, target, or dispatch
        state. It only supplies shared teammate context to always-on actors.
        """
        recon_groups = self.recon_controller.initialize_teams(self.platforms)
        attack_groups = self.attack_controller.initialize_teams(self.platforms)
        landing_zone = next(iter(self.config.get("landing_zones", [])), {})
        # Landing teams explore the island shoreline; no berth is pre-assigned.
        landing_zone = {}
        objective = next(iter(self.config.get("ground_objectives", [])), {})
        scenario = self.config.get("scenario", {})
        landing_groups = self.landing_controller.initialize_teams(
            self.platforms, landing_zone, scenario.get("landing_group_size", 3)
        )
        ground_groups = self.ground_controller.initialize_teams(
            self.platforms, objective, scenario.get("ground_group_size", 3)
        )
        return {
            "recon": [group.group_id for group in recon_groups],
            "attack": [group.group_id for group in attack_groups],
            "landing": [group.group_id for group in landing_groups],
            "ground": [group.group_id for group in ground_groups],
        }

    def synchronize_sim_time_window(self, timeout=0.5):
        """Discard startup backlog, then anchor on a genuinely new state update.

        AFSIM advances while Python is registering platforms.  Draining the
        socket alone can still leave an old timestamp as the first training
        observation.  Require a subsequent MoveUpdate before enabling the
        first decision window.
        """
        timeout = max(0.1, float(timeout))
        self._drain_messages(timeout=timeout)
        sequence_before = int(self.platform_state_sequence)
        deadline = time.monotonic() + max(1.5, timeout + max(0.25, self.message_timeout_seconds))
        while time.monotonic() < deadline and self.platform_state_sequence <= sequence_before:
            remaining = max(0.0, deadline - time.monotonic())
            self._drain_messages(timeout=min(self.message_timeout_seconds, remaining))
        if self.platform_state_sequence <= sequence_before:
            raise RuntimeError("no fresh PlatformState/MoveUpdate after startup UDP drain")
        self._next_decision_target_sim_time = None
        return self._current_sim_time()
    def warmup_sim_time_window(self):
        """Consume one startup timing window without recording a policy sample."""
        self.initialize_bottom_teams()
        self.step_flat()
        return self._current_sim_time()
    def step_flat(self):
        """Advance one interval after all bottom agents have acted."""
        self.last_reward_events = []
        self.initialize_bottom_teams()
        if self.native_decision_pause_control:
            # AFSIM emits no further DecisionReady after the fixed horizon.
            # Do not resume and wait for a packet that can never arrive.
            if self.is_done():
                reward = self.compute_reward()
                return reward, True, {
                    "control_mode": "flat_always_on",
                    "events": list(self.last_reward_events),
                    "reward_details": list(self.last_reward_details),
                    "known_targets": list(self.detected_targets.keys()),
                    "episode_result": self.episode_result,
                    "done_reason": self.episode_done_reason,
                    "final_score_raw": self.final_score_raw,
                    "final_score_norm": self.final_score_norm,
                    "final_score_unit_count": self.final_score_unit_count,
                    "final_score_units": list(self.final_score_units),
                    "final_score_sim_time": self.final_score_sim_time,
                    "drain": dict(self.last_drain_metadata),
                }
            native_wait_started = time.monotonic()
            attempts = 1 + max(0, int(self.native_decision_pause_retries))
            for attempt in range(1, attempts + 1):
                self.native_decision_ready = False
                if self.debug_native_decision_pause:
                    print("native_pause_resume_sent", self.local_address[1], "attempt", attempt, "sim", self._current_sim_time(), flush=True)
                self._send_native_restart(float(self.native_decision_ready_time))
                wait_timeout = self.native_decision_pause_timeout if attempt == 1 else self.native_decision_retry_timeout
                self._drain_messages(timeout=wait_timeout, until_decision_ready=True)
                if self.native_decision_ready or self.is_done():
                    break
                print(
                    "native_pause_retry", self.local_address[1],
                    "attempt", attempt, "of", attempts,
                    "last_sim_time", round(self._current_sim_time(), 3),
                    flush=True,
                )
            if self.debug_native_decision_pause:
                print("native_pause_boundary", self.local_address[1], "ready", self.native_decision_ready, "sim", self.native_decision_ready_time, "wait_wall", round(time.monotonic() - native_wait_started, 3), flush=True)
            if not self.native_decision_ready and not self.is_done():
                raise RuntimeError(
                    "AFSIM native decision pause timed out after {} attempts "
                    "(initial_timeout={:.1f}s, retry_timeout={:.1f}s, last_sim_time={:.3f})".format(
                        attempts, self.native_decision_pause_timeout,
                        self.native_decision_retry_timeout, self._current_sim_time()
                    )
                )
        elif self.lockstep_pause_control and self.simulation_paused:
            # Resume the Python-owned pause and then perform the same timing
            # window in this call. Returning immediately would create a
            # zero-progress policy sample after every SimRestart.
            self._send({"MsgType": "SimRestart"})
            self.simulation_paused = False
            return self.step_flat()
        elif self.sim_time_window_control:
            current = self._current_sim_time()
            interval = float(getattr(self, "decision_sim_seconds", 72.0))
            heartbeat = max(1.0, float(self.config.get("scenario", {}).get(
                "state_update_interval_seconds", 15.0
            )))
            # Startup can contain a backlog of old UDP state packets. If the
            # current watermark has already passed the scheduled target, rebase
            # instead of emitting an immediate zero-length catch-up window.
            if (self._next_decision_target_sim_time is None
                    or current >= self._next_decision_target_sim_time):
                self._next_decision_target_sim_time = heartbeat * np.ceil(
                    (current + interval) / heartbeat
                )
            else:
                self._next_decision_target_sim_time += interval
            rate = max(1.0, float(getattr(self, "simulation_clock_rate", 1.0)))
            max_wall = max(float(self.decision_seconds), interval / rate + 0.5)
            max_wall = min(max_wall, self.sim_time_window_max_wall_seconds)
            receive_target = self._next_decision_target_sim_time - max(0.0, self.sim_time_window_early_tolerance_seconds)
            self._drain_messages(timeout=max_wall, until_sim_time=receive_target)
            if self.lockstep_pause_control and self.last_drain_metadata.get("target_reached", False):
                self._send({"MsgType": "SimStop"})
                self.simulation_paused = True
        else:
            self._drain_messages(timeout=self.decision_seconds)
        self.step_count += 1
        done = self.is_done()
        reward = self.compute_reward()
        info = {
            "control_mode": "flat_always_on",
            "events": list(self.last_reward_events),
            "reward_details": list(self.last_reward_details),
            "known_targets": list(self.detected_targets.keys()),
            "episode_result": self.episode_result,
            "done_reason": self.episode_done_reason,
            "final_score_raw": self.final_score_raw,
            "final_score_norm": self.final_score_norm,
            "final_score_unit_count": self.final_score_unit_count,
            "final_score_units": list(self.final_score_units),
            "final_score_sim_time": self.final_score_sim_time,
            "drain": dict(self.last_drain_metadata),
        }
        return reward, done, info

    def step_rule_driven(self):
        """Compatibility wrapper for older callers."""
        return self.step_flat()

    def step(self, action):
        raise RuntimeError(
            "high-level actions are removed; use step_flat()"
        )

    def _send(self, msg):
        if not self.sock or not self.remote_addr:
            self.last_reward_events.append({"type": "udp_not_ready", "reward": -1.0})
            return
        self.sock.sendto(json.dumps(msg).encode("utf-8"), self.remote_addr)


    def start_recon_group(self, area, fixed_team_index):
        self._prepare_recon_area_cycle(area)
        group, assignments, error = self.recon_controller.start_group(
            area,
            self.platforms,
            self._is_busy,
            self._normalize_recon_area,
            fixed_team_index=fixed_team_index,
        )
        if error:
            self.last_reward_events.append({"type": error, "reward": -1.0})
            return None
        for assignment in assignments:
            self._send(assignment.message)
        group.start_step = self.step_count
        group.start_sim_time = float(self._current_sim_time())
        group.survey_started_at = -1.0
        self.recon_controller.mark_group_assigned(group)
        return group

    def apply_recon_aircraft_continuous_action(self, group_id, aircraft_name, action):
        group = self.recon_controller.active_groups.get(group_id)
        if group is None:
            self.last_reward_events.append({
                "type": "recon_action_masked", "reward": -1.0, "platform": aircraft_name
            })
            return False
        msg, error = self.recon_controller.create_aircraft_continuous_action_message(
            group_id, aircraft_name, action
        )
        if error:
            self.last_reward_events.append({
                "type": error, "reward": -1.0, "platform": aircraft_name
            })
            return False
        if msg is not None:
            self._send(msg)

        # A leader continuous action is a team command. Dispatch formation
        # waypoints immediately so followers move with the leader even if the
        # caller does not submit their actions in a particular order.
        leader = self.recon_controller.ensure_group_leader(group)
        if msg is not None and leader is not None and aircraft_name == leader.name:
            for member in group.platforms:
                if not member.alive or member.name == leader.name:
                    continue
                follower_msg, follower_error = self.recon_controller.create_aircraft_continuous_action_message(
                    group_id, member.name, (0.0, 0.0, 0.0)
                )
                if follower_error:
                    self.last_reward_events.append({
                        "type": follower_error, "reward": -1.0, "platform": member.name
                    })
                    continue
                if follower_msg is not None:
                    self._send(follower_msg)
        return True
    def start_attack_area_group(self, mission_area, fixed_team_index):
        group, assignments, error = self.attack_controller.start_area_group(
            self._normalize_recon_area(mission_area),
            self.platforms,
            self._is_busy,
            fixed_team_index=fixed_team_index,
        )
        if error:
            self.last_reward_events.append({"type": error, "reward": -1.0, "area": (mission_area or {}).get("name", "")})
            return None
        for assignment in assignments:
            if assignment.message:
                self._send(assignment.message)
        group.start_step = self.step_count
        self.attack_controller.mark_group_assigned(group)
        self.task_flags["current_attack"] = "RULE_ATTACK:{0}:{1}".format(
            group.fixed_team_id, group.mission_area.get("name", "")
        )
        return group
    def start_attack_group(self, target_name, fixed_team_index):
        target = self._attack_target_info(target_name)
        if not target or not target.get("known", False):
            self.last_reward_events.append({"type": "attack_target_not_known", "reward": -1.0, "target": target_name})
            return None
        group, assignments, error = self.attack_controller.start_group(
            target_name,
            target,
            self.platforms,
            self._is_busy,
            fixed_team_index=fixed_team_index,
        )
        if error:
            self.last_reward_events.append({"type": error, "reward": -1.0, "target": target_name})
            return None
        for assignment in assignments:
            if assignment.message:
                self._send(assignment.message)
        group.start_step = self.step_count
        self.attack_controller.mark_group_assigned(group)
        self.task_flags["current_attack"] = "ATTACK:{0}".format(target_name)
        return group

    def apply_attack_aircraft_action(
        self, group_id, aircraft_name, action_id,
        _force_in_range=False, _allow_locked_return=False,
    ):
        action = self.attack_controller.action_specs.get(int(action_id), {})
        pending_fire = self._active_pending_attack_fire(aircraft_name)
        if pending_fire and action.get("name") == "HOLD":
            return True
        allow_return = bool(_allow_locked_return and action.get("name") == "RETURN_HOME")
        # A fresh attack/rejoin decision supersedes any previous fire fallback
        # and approach watcher for this aircraft. A new fire command, if any,
        # is registered below after the action is accepted.
        if action.get("afsim_task") in ("ATTACK_TARGET_SLOT", "ATTACK_REJOIN_FORMATION"):
            self.pending_attack_fire_commands.pop(aircraft_name, None)
            self.pending_attack_approaches.pop(aircraft_name, None)
        if not allow_return and not self._attack_action_allowed(group_id, aircraft_name, action_id):
            self.last_reward_events.append({"type": "attack_action_masked", "reward": -1.0, "platform": aircraft_name})
            return False
        group = self.attack_controller.active_groups.get(group_id)
        platform = self.platforms.get(aircraft_name)
        service_phase = str(self.pending_attack_returns.get(aircraft_name, {}).get("phase", ""))
        if service_phase in ("returning", "rearming") and action.get("name") == "RETURN_HOME":
            # Re-selecting RETURN_HOME means continue the active RETREAT.
            # Do not emit ATTACK_HOLD, which would stop the aircraft.
            return True
        leader = self.attack_controller.ensure_group_leader(group) if group else None
        if (
            group is not None
            and leader is not None
            and aircraft_name == leader.name
            and action.get("afsim_task") == "RETREAT"
        ):
            members = [member for member in group.platforms if self.platforms.get(member.name, member).alive]
            any_sent = False
            for member in members:
                member_action_id = int(action_id)
                msg, error = self.attack_controller.create_aircraft_action_message(
                    group_id,
                    member.name,
                    member_action_id,
                )
                if error:
                    self.last_reward_events.append({"type": error, "reward": -1.0, "platform": member.name})
                    return False
                if msg:
                    self._send(msg)
                    any_sent = True
                self._mark_attack_action_local(group_id, member.name, member_action_id, msg=msg)
            return any_sent
        target_name = None
        target = None
        if group and action.get("afsim_task") == "ATTACK_TARGET_SLOT":
            slot = int(action.get("target_slot", -1))
            target_slots = self._attack_action_target_slots()
            if slot < 0 or slot >= len(target_slots):
                self.last_reward_events.append({"type": "attack_target_slot_empty", "reward": -1.0, "platform": aircraft_name})
                return False
            target_name = target_slots[slot].get("name")
            target = dict(target_slots[slot])
            # Navigation uses only the initial prior or latest observed track.
            # Live AFSIM geometry is intentionally used only for the launch gate.
            actual = self.platforms.get(target_name)
            target_lat = float(target.get("lat", platform.lat))
            target_lon = float(target.get("lon", platform.lon))
            target_alt = float(target.get("alt", 0.0))
            actual_target = dict(target)
            if actual is not None:
                actual_target.update({
                    "lat": float(actual.lat), "lon": float(actual.lon), "alt": float(actual.alt),
                    "alive": bool(actual.alive),
                    "type": actual.platform_type or actual.role or target.get("type", ""),
                })
            target_distance, _ = self._slant_distance_and_bearing(
                platform.lat, platform.lon, platform.alt,
                float(actual_target.get("lat", target_lat)),
                float(actual_target.get("lon", target_lon)),
                float(actual_target.get("alt", target_alt)),
            )
            target["weapon_range_m"] = self._attack_weapon_range(target)
            # Current real geometry controls firing; stored intel controls movement.
            target["_in_weapon_range"] = bool(_force_in_range) or self._target_in_attack_launch_range(
                platform, actual_target, target_distance
            )
        msg, error = self.attack_controller.create_aircraft_action_message(
            group_id,
            aircraft_name,
            action_id,
            target_override_name=target_name,
            target_override=target,
        )
        if error:
            self.last_reward_events.append({"type": error, "reward": -1.0, "platform": aircraft_name})
            return False
        if msg and target_name and action.get("afsim_task") == "ATTACK_TARGET_SLOT":
            target_platform = self.platforms.get(target_name)
            navigation_distance, _ = self._slant_distance_and_bearing(
                platform.lat, platform.lon, platform.alt,
                target_lat, target_lon, target_alt,
            )
            actual_horizontal, _ = self._distance_and_bearing(
                platform.lat, platform.lon,
                float(actual_target.get("lat", target_lat)),
                float(actual_target.get("lon", target_lon)),
            )
            navigation_horizontal, _ = self._distance_and_bearing(
                platform.lat, platform.lon, target_lat, target_lon,
            )
            track_error, _ = self._distance_and_bearing(
                target_lat, target_lon,
                float(actual_target.get("lat", target_lat)),
                float(actual_target.get("lon", target_lon)),
            )
            previous_approach = self.pending_attack_approaches.get(
                aircraft_name, {}
            )
            weapon = self.attack_controller._compatible_weapon(target)
            launch_range = (
                float(self.attack_state_config.get("normalization", {}).get(
                    "agm_horizontal_launch_range_m", 45000.0
                ))
                if weapon == "agm"
                else self._attack_weapon_range(target)
            )
            self.last_reward_events.append({
                "type": "attack_target_selected",
                "platform": aircraft_name,
                "group_id": group_id,
                "target": target_name,
                "target_role": str(getattr(target_platform, "role", target.get("type", "unknown"))),
                "action_id": int(action_id),
                "phase": "fire" if str(msg.get("Task", "")) in ("FIRE_AAM", "FIRE_AGM") else "approach",
                "task": str(msg.get("Task", "")),
                "sim_time": float(self._current_sim_time()),
                "weapon": weapon,
                "in_weapon_range": bool(target.get("_in_weapon_range", False)),
                "launch_range_m": float(launch_range),
                "actual_slant_distance_m": float(target_distance),
                "actual_horizontal_distance_m": float(actual_horizontal),
                "navigation_slant_distance_m": float(navigation_distance),
                "navigation_horizontal_distance_m": float(navigation_horizontal),
                "track_error_m": float(track_error),
                "track_source": str(target.get("source", "")),
                "track_last_seen": target.get("last_seen"),
                "previous_target": str(previous_approach.get("target_name", "")),
                "target_changed": bool(
                    previous_approach
                    and str(previous_approach.get("target_name", "")) != target_name
                ),
            })
        if msg:
            sent_task = str(msg.get("Task", ""))
            if sent_task == "ATTACK_MOVE_POINT":
                # A new movement command supersedes any stale fire fallback.
                self.pending_attack_fire_commands.pop(aircraft_name, None)
            self._send(msg)
            if sent_task in ("FIRE_AAM", "FIRE_AGM"): 
                now = float(self._current_sim_time())
                fired_target = str(
                    msg.get("TargetName", target_name or "")
                )
                self.pending_attack_fire_commands[aircraft_name] = {
                    "group_id": group_id,
                    "target_name": fired_target,
                    "weapon": str(msg.get("Weapon", action.get("weapon", ""))),
                    "sent_at": now,
                    "execute_until": now + 5.0,
                    "previous_shooter": self.attack_last_shooter_by_target.get(fired_target, ""),
                }
                if fired_target:
                    self.attack_last_shooter_by_target[fired_target] = aircraft_name
                    self.attack_last_weapon_by_target[fired_target] = str(msg.get("Weapon", action.get("weapon", "")))
        if (
            msg
            and str(msg.get("Task", "")) == "ATTACK_MOVE_POINT"
            and target_name
            and action.get("afsim_task") == "ATTACK_TARGET_SLOT"
        ):
            current_time = float(self._current_sim_time())
            self.pending_attack_approaches[aircraft_name] = {
                "group_id": group_id,
                "target_name": target_name,
                "action_id": int(action_id),
                "started_at": current_time,
                "estimated_arrival_at": current_time + max(
                    20.0,
                    float(target_distance) / max(1.0, float(self.attack_controller.max_speed_mps)),
                ),
                "last_distance_m": float(target_distance),
                "member_mode": str(getattr(group, "member_modes", {}).get(aircraft_name, "LEADER")),
            }
        if (
            msg
            and str(msg.get("Task", "")) == "ATTACK_MOVE_POINT"
            and action.get("afsim_task") == "ATTACK_REJOIN_FORMATION"
        ):
            self.pending_attack_rejoins[aircraft_name] = {"group_id": group_id}
        self._mark_attack_action_local(group_id, aircraft_name, action_id, msg=msg)
        return True

    def register_target_reward_contributor(self, target_name, entity_name, contribution_type):
        """Record target-specific discovery/attack credit for delayed rewards."""
        target_name = str(target_name or "")
        entity_name = str(entity_name or "")
        if not target_name or not entity_name:
            return
        target = self.platforms.get(target_name)
        if target is not None and target.side != "blue":
            return
        ledger = self.target_reward_contributors.setdefault(
            target_name, {"discovered_by": set(), "attacked_by": set()})
        key = "attacked_by" if str(contribution_type) == "attack" else "discovered_by"
        ledger[key].add(entity_name)

    def get_target_reward_contributors(self, target_name):
        ledger = self.target_reward_contributors.get(str(target_name or ""), {})
        discovered = set(ledger.get("discovered_by", set()))
        attacked = set(ledger.get("attacked_by", set()))
        return {
            "discovered_by": sorted(discovered),
            "attacked_by": sorted(attacked),
            "all": sorted(discovered | attacked),
        }

    def _active_pending_attack_fire(self, aircraft_name):
        pending = self.pending_attack_fire_commands.get(aircraft_name)
        if pending is None:
            return None
        if float(self._current_sim_time()) >= float(pending.get("execute_until", 0.0)):
            self.pending_attack_fire_commands.pop(aircraft_name, None)
            return None
        return pending

    def _reject_pending_attack_fire(self, aircraft_name):
        pending = self.pending_attack_fire_commands.pop(aircraft_name, None)
        if pending is None:
            return
        weapon = str(pending.get("weapon", ""))
        if weapon in ("fox3", "agm"):
            ammo = self.attack_ammo.setdefault(aircraft_name, {"fox3": 1, "agm": 1})
            ammo[weapon] = min(1, int(ammo.get(weapon, 0)) + 1)
        target_name = str(pending.get("target_name", ""))
        if target_name and self.attack_last_shooter_by_target.get(target_name) == aircraft_name:
            previous_shooter = str(pending.get("previous_shooter", ""))
            if previous_shooter:
                self.attack_last_shooter_by_target[target_name] = previous_shooter
            else:
                self.attack_last_shooter_by_target.pop(target_name, None)
        if target_name and self.attack_target_reservations.get(target_name) == aircraft_name:
            self.attack_target_reservations.pop(target_name, None)
        group = self.attack_controller.active_groups.get(pending.get("group_id"))
        if group and target_name and group.target_reservations.get(target_name) == aircraft_name:
            group.target_reservations.pop(target_name, None)

    def _hold_attack_aircraft_after_fire(self, aircraft_name, pending=None):
        """Send a single stop command after an attack launch is accepted."""
        platform = self.platforms.get(str(aircraft_name))
        if platform is None or platform.role != "attack_aircraft" or not platform.alive:
            return False
        pending = pending or self.pending_attack_fire_commands.get(platform.name, {})
        # A fire fallback must not leave the old approach watcher active.
        self.pending_attack_approaches.pop(platform.name, None)
        if pending.get("hold_sent"):
            return True
        self._send(self.attack_controller._build_hold_message(platform))
        platform.task = "ATTACK_HOLD"
        platform.task_status = "ASSIGNED"
        platform.task_assigned = True
        platform.speed = 0.0
        group = self.attack_controller.active_groups.get(pending.get("group_id"))
        if group is not None and group.leader_name == platform.name:
            group.leader_task = "ATTACK_HOLD"
            group.leader_target_position = []
            group.leader_command_version += 1
        if pending:
            pending["hold_sent"] = True
            pending["hold_sent_at"] = float(self._current_sim_time())
        self.last_reward_events.append({
            "type": "attack_hold_after_fire",
            "reward": 0.0,
            "platform": platform.name,
            "target": str(pending.get("target_name", "")),
        })
        return True

    def _check_pending_attack_fire_holds(self):
        """Fallback stop when AFSIM does not return a fire TaskAck."""
        now = float(self._current_sim_time())
        for aircraft_name, pending in list(self.pending_attack_fire_commands.items()):
            if pending.get("hold_sent"):
                continue
            if now - float(pending.get("sent_at", now)) >= 3.0:
                self._hold_attack_aircraft_after_fire(aircraft_name, pending)

    def _maybe_confirm_attack_approach(self, platform):
        """Stop an approach at launch range; firing requires a later decision."""
        if platform is None or not platform.alive:
            return
        if str(getattr(platform, "task", "")).upper() == "RETREAT":
            self.pending_attack_approaches.pop(platform.name, None)
            return
        pending = self.pending_attack_approaches.get(platform.name)
        if not pending:
            return

        group_id = str(pending.get("group_id", ""))
        group = self.attack_controller.active_groups.get(group_id)
        if group is None:
            self.pending_attack_approaches.pop(platform.name, None)
            return

        target_name = str(pending.get("target_name", ""))
        target = self.platforms.get(target_name)
        if target is None or not target.alive:
            self.pending_attack_approaches.pop(platform.name, None)
            return

        target_info = self._fixed_attack_action_target_info(target_name)
        range_target_info = dict(target_info)
        range_target_info.update({
            "lat": float(target.lat), "lon": float(target.lon), "alt": float(target.alt),
            "alive": bool(target.alive),
            "type": target.platform_type or target.role or target_info.get("type", ""),
        })
        distance, _ = self._slant_distance_and_bearing(            platform.lat, platform.lon, platform.alt,
            target.lat, target.lon, target.alt,
        )
        if not self._target_in_attack_launch_range(platform, range_target_info, distance):
            pending["last_distance_m"] = float(distance)
            return

        # A detached wingman reaches launch range independently and must not
        # stop the leader or other members that are still following it.
        if (platform.name != group.leader_name and
                str(pending.get("member_mode", "")) == "INDEPENDENT_ATTACK"):
            self.pending_attack_approaches.pop(platform.name, None)
            self._send(self.attack_controller._build_hold_message(platform))
            platform.task = "ATTACK_HOLD"
            platform.task_status = "ASSIGNED"
            platform.task_assigned = True
            platform.speed = 0.0
            self.last_reward_events.append({
                "type": "independent_attack_in_weapon_range",
                "reward": 0.0,
                "platform": platform.name,
                "group_id": group_id,
                "target": target_name,
            })
            return
        # A move action ends at the weapon boundary.  Do not convert it into a
        # fire action here: the caller must make a new, explicit decision.
        self.pending_attack_approaches.pop(platform.name, None)
        for member in group.platforms:
            live_member = self.platforms.get(member.name, member)
            if not live_member.alive:
                continue
            if (live_member.name != group.leader_name and
                    str(getattr(group, "member_modes", {}).get(live_member.name, "FOLLOWING")) != "FOLLOWING"):
                continue
            self._send(self.attack_controller._build_hold_message(live_member))
            live_member.task = "ATTACK_HOLD"
            live_member.task_status = "ASSIGNED"
            live_member.task_assigned = True
            live_member.speed = 0.0
            live_member.at_home = False

        group.leader_task = "ATTACK_HOLD"
        group.leader_target_position = []
        group.leader_command_version += 1
        for member in group.platforms:
            if member.name != group.leader_name:
                group.follower_command_versions[member.name] = group.leader_command_version

        weapon = self.attack_controller._compatible_weapon(target_info)
        horizontal_distance, _ = self._distance_and_bearing(
            platform.lat, platform.lon, target.lat, target.lon
        )
        weapon_range = self._attack_weapon_range(target_info)
        self.last_reward_events.append(
            {
                "type": "attack_approach_in_weapon_range",
                "reward": 0.0,
                "platform": platform.name,
                "group_id": group_id,
                "target": target_name,
                "weapon": weapon,
                "distance_m": float(horizontal_distance),
                "weapon_range_m": float(weapon_range),
            }
        )
    def _maybe_confirm_attack_rejoin(self, platform):
        pending = self.pending_attack_rejoins.get(platform.name)
        if not pending or platform is None or not platform.alive:
            return
        group = self.attack_controller.active_groups.get(str(pending.get("group_id", "")))
        if group is None:
            self.pending_attack_rejoins.pop(platform.name, None)
            return
        leader = self.attack_controller.ensure_group_leader(group)
        if leader is None or leader.name == platform.name:
            self.pending_attack_rejoins.pop(platform.name, None)
            return
        distance, _ = self._distance_and_bearing(platform.lat, platform.lon, leader.lat, leader.lon)
        slot_radius = self.attack_controller.formation_spacing(group, platform.name) + 1800.0
        if distance > slot_radius:
            return
        self.pending_attack_rejoins.pop(platform.name, None)
        group.member_modes[platform.name] = "FOLLOWING"
        group.follower_command_versions[platform.name] = group.leader_command_version
        self._send(self.attack_controller._build_hold_message(platform))
        platform.task = "ATTACK_HOLD"
        platform.task_status = "ASSIGNED"
        platform.task_assigned = True
        platform.speed = 0.0
        self.last_reward_events.append({"type": "attack_rejoin_complete", "reward": 0.0, "platform": platform.name, "group_id": group.group_id})
    def _check_pending_attack_approaches(self):
        """Continuously stop approaches that reach their launch boundary."""
        self._check_pending_attack_fire_holds()
        for aircraft_name in list(self.pending_attack_rejoins):
            platform = self.platforms.get(aircraft_name)
            if platform is not None:
                self._maybe_confirm_attack_rejoin(platform)
        for aircraft_name in list(self.pending_attack_approaches):
            platform = self.platforms.get(aircraft_name)
            if platform is not None:
                self._maybe_confirm_attack_approach(platform)

    def _attack_action_allowed(self, group_id, aircraft_name, action_id):
        state = self.get_attack_task_state(group_id)
        if not state or aircraft_name not in state.get("aircraft", {}):
            return False
        mask = state["aircraft"][aircraft_name].get("action_mask", [])
        return 0 <= int(action_id) < len(mask) and float(mask[int(action_id)]) > 0.0
    def _mark_attack_action_local(self, group_id, aircraft_name, action_id, msg=None):
        action = self.attack_controller.action_specs.get(int(action_id), {})
        weapon = action.get("weapon") or ((msg or {}).get("Weapon"))
        platform = self.platforms.get(aircraft_name)
        if weapon in ["fox3", "agm"]:
            ammo = self.attack_ammo.setdefault(aircraft_name, {"fox3": 1, "agm": 1})
            ammo[weapon] = max(0, int(ammo.get(weapon, 0)) - 1)
            if platform:
                platform.task = "ATTACK"
                platform.task_status = "ASSIGNED"
                platform.at_home = False
        sent_task = str((msg or {}).get("Task", action.get("afsim_task", "")))
        if platform and sent_task == "ATTACK_MOVE_POINT":
            self.pending_attack_returns.pop(aircraft_name, None)
        if platform and sent_task == "RETREAT":
            # A retreat supersedes any stale attack approach/rejoin watcher.
            self.pending_attack_approaches.pop(aircraft_name, None)
            self.pending_attack_rejoins.pop(aircraft_name, None)
            self.pending_attack_returns[aircraft_name] = {
                "group_id": group_id,
                "phase": "returning",
                "return_started_at": float(self._current_sim_time()),
            }
            platform.task = "RETREAT"
            platform.task_status = "ASSIGNED"
            platform.at_home = False

    def start_ground_group(self, objective, group_size=3):
        group, assignments, error = self.ground_controller.start_group(
            objective,
            self.platforms,
            self._is_busy,
            self._is_landed_ground_available,
            group_size=group_size,
        )
        if error:
            self.last_reward_events.append({"type": error, "reward": -1.0})
            return None
        group.start_step = self.step_count
        self.ground_controller.mark_group_assigned(group)
        return group

    def apply_ground_unit_action(self, group_id, unit_name, action_id):
        if not self._ground_action_allowed(group_id, unit_name, action_id):
            self.last_reward_events.append({"type": "ground_action_masked", "reward": -1.0, "platform": unit_name})
            return False
        action = self.ground_controller.action_specs.get(int(action_id), {})
        target = None
        if action.get("afsim_task") == "GROUND_TARGET_SLOT":
            group = self.ground_controller.active_groups.get(group_id)
            platform = self.platforms.get(unit_name)
            target_slots = self._ground_target_slots(group, platform)
            slot = int(action.get("target_slot", -1))
            if slot < 0 or slot >= len(target_slots):
                self.last_reward_events.append({"type": "ground_target_slot_empty", "reward": -1.0, "platform": unit_name})
                return False
            target = target_slots[slot]
            actual = self.platforms.get(target.get("name", ""))
            weapon_range = float(self.ground_state_config.get("normalization", {}).get("ground_weapon_range_m", 5000.0))
            actual_distance = float("inf")
            if actual is not None:
                actual_distance, _ = self._distance_and_bearing_to_platform(platform, actual)
            target["_in_weapon_range"] = bool(actual is not None and actual.alive and actual_distance <= weapon_range)
        msg, error = self.ground_controller.create_ground_action_message(group_id, unit_name, action_id, target=target)
        if error:
            self.last_reward_events.append({"type": error, "reward": -1.0, "platform": unit_name})
            return False
        if msg:
            self._send(msg)
        if str((msg or {}).get("Task", "")) == "GROUND_FIRE" and target:
            target_name = target["name"]
            self.attack_target_reservations[target_name] = unit_name
            flight_seconds = max(1.0, float(target.get("distance_m", 0.0)) / 500.0)
            self.ground_target_reservations[target_name] = {
                "owner": unit_name,
                "expires_at": float(self._current_sim_time()) + flight_seconds + 1.0,
            }
        self._mark_ground_action_local(unit_name, action_id, msg=msg)
        return True

    def _ground_action_allowed(self, group_id, unit_name, action_id):
        state = self.get_ground_task_state(group_id)
        if not state or unit_name not in state.get("units", {}):
            return False
        mask = state["units"][unit_name].get("action_mask", [])
        return 0 <= int(action_id) < len(mask) and float(mask[int(action_id)]) > 0.0

    def _mark_ground_action_local(self, unit_name, action_id, msg=None):
        action = self.ground_controller.action_specs.get(int(action_id), {})
        platform = self.platforms.get(unit_name)
        if not platform:
            return
        # Match attack-aircraft HOLD semantics: a no-message action leaves the
        # currently executing AFSIM task and its mirrored local state intact.
        if not msg:
            return
        afsim_task = str(msg.get("Task", action.get("afsim_task", "")))
        if afsim_task == "GROUND_FIRE":
            ammo = self.ground_ammo.setdefault(unit_name, {"ground_fire": 15})
            ammo["ground_fire"] = max(0, int(ammo.get("ground_fire", 0)) - 1)

    def _mark_ground_landed_from_transport(self, transport_name, landing_zone=None, ground_name=None):
        for name, status in self.ground_status.items():
            if status.get("transport") == transport_name:
                if ground_name is not None and name != ground_name:
                    continue
                status["on_ship"] = False
                status["landed"] = True
                p = self.platforms.get(name)
                ship = self.platforms.get(transport_name)
                if p:
                    p.task = "PARKED"
                    p.task_status = "IDLE"
                    p.at_home = False
                    if landing_zone:
                        p.lat = float(landing_zone.get("lat", p.lat))
                        p.lon = float(landing_zone.get("lon", p.lon))
                        p.alt = 0.0
                    elif ship:
                        p.lat, p.lon, p.alt = ship.lat, ship.lon, 0.0

    def _is_landed_ground_available(self, platform):
        status = self.ground_status.get(platform.name, {"on_ship": False, "landed": True})
        return bool(status.get("landed", False)) and not bool(status.get("on_ship", False))

    def _is_loaded_transport_available(self, platform):
        cargo = self.landing_cargo.get(platform.name, {})
        return bool(cargo.get("has_army", False)) and not bool(cargo.get("army_landed", False))

    def start_landing_group(self, landing_zone, group_size=1):
        # Kept for caller compatibility; transports now explore rather than target a zone.
        landing_zone = {}
        group, assignments, error = self.landing_controller.start_group(
            landing_zone,
            self.platforms,
            self._is_busy,
            group_size=group_size,
            is_available_transport=self._is_loaded_transport_available,
        )
        if error:
            self.last_reward_events.append({"type": error, "reward": -1.0})
            return None
        group.start_step = self.step_count
        navigation_lat, navigation_lon = self._landing_navigation_point(group.landing_zone, group.platforms[0])
        group.landing_zone["navigation_lat"] = navigation_lat
        group.landing_zone["navigation_lon"] = navigation_lon
        self.landing_controller.mark_group_assigned(group)
        return group

    def _landing_navigation_point(self, landing_zone, platform):
        """Return the fixed sea-side berth associated with a shoreline landing point."""
        if "berth_lat" in landing_zone and "berth_lon" in landing_zone:
            return float(landing_zone["berth_lat"]), float(landing_zone["berth_lon"])
        return (
            float(landing_zone.get("lat", landing_zone.get("center_lat", platform.lat))),
            float(landing_zone.get("lon", landing_zone.get("center_lon", platform.lon))),
        )
    def apply_landing_ship_action(self, group_id, ship_name, action_id):
        spec = self.landing_controller.action_specs.get(int(action_id), {})
        if ship_name in self.pending_landing_unloads and spec.get("name") == "HOLD":
            return True
        msg, error = self.landing_controller.create_ship_action_message(group_id, ship_name, action_id)
        if error:
            self.last_reward_events.append({"type": error, "reward": -1.0, "platform": ship_name})
            return False
        if not self._landing_action_allowed(group_id, ship_name, action_id):
            self.last_reward_events.append({"type": "landing_action_masked", "reward": -1.0, "platform": ship_name})
            return False
        if msg:
            self._send(msg)
        self._mark_landing_action_local(group_id, ship_name, action_id)
        return True

    def apply_landing_ship_continuous_action(self, group_id, ship_name, action):
        """Apply a bounded surface move; landing itself is environment-driven."""
        if ship_name in self.pending_landing_unloads:
            return True
        if not self._is_landing_window_open():
            self.last_reward_events.append({
                "type": "landing_window_closed",
                "reward": 0.0,
                "platform": ship_name,
            })
            return False
        msg, error = self.landing_controller.create_ship_continuous_action_message(
            group_id, ship_name, action
        )
        if error:
            self.last_reward_events.append({"type": error, "reward": -1.0, "platform": ship_name})
            return False
        platform = self.platforms.get(ship_name)
        if platform is None:
            return False
        target = list(msg.get("MovePosition", [platform.lat, platform.lon, 0.0]))
        target_lat, target_lon, shore = self._clip_transport_move_to_shore(
            platform.lat, platform.lon, target[0], target[1]
        )
        msg["MovePosition"] = [target_lat, target_lon, 0.0]
        if shore is not None:
            msg["ShoreContactPoint"] = list(shore)
        self._send(msg)
        platform.task = "LANDING_MOVE_POINT"
        platform.task_status = "ASSIGNED"
        platform.task_assigned = True
        platform.at_home = False
        return True
    def _landing_action_allowed(self, group_id, ship_name, action_id):
        state = self.get_landing_task_state(group_id)
        if not state or ship_name not in state.get("ships", {}):
            return False
        mask = state["ships"][ship_name].get("action_mask", [])
        return 0 <= int(action_id) < len(mask) and float(mask[int(action_id)]) > 0.0

    def _mark_landing_action_local(self, group_id, ship_name, action_id):
        action = self.landing_controller.action_specs.get(int(action_id), {})
        afsim_task = action.get("afsim_task", "")
        platform = self.platforms.get(ship_name)
        if platform:
            if afsim_task in ["LANDING_MOVE_POINT", "LANDING_RETURN_WAIT"]:
                platform.task = afsim_task
                platform.task_status = "ASSIGNED"
                platform.at_home = False
            elif afsim_task == "LANDING_HOLD":
                platform.task = "LANDING_HOLD"
                platform.task_status = "ASSIGNED"
            elif afsim_task == "LANDING_UNLOAD":
                group = self.landing_controller.active_groups.get(group_id)
                landing_zone = dict(group.landing_zone) if group else {}
                now = float(self._current_sim_time())
                duration = float(self.config.get("scenario", {}).get("landing_unload_duration_seconds", 900.0))
                self.pending_landing_unloads.setdefault(ship_name, {
                    "group_id": group_id,
                    "landing_zone": landing_zone,
                    "phase": "unloading",
                    "unload_started_at": now,
                    "unload_complete_at": now + duration,
                })
                platform.task = "LANDING_UNLOADING"
                platform.task_status = "ASSIGNED"

    def _maybe_confirm_recon_return(self, platform):
        if platform is None or platform.role != "recon_aircraft":
            return
        if platform.task != "RETREAT":
            return
        carrier = next((item for item in self.platforms.values() if item.role == "carrier" and item.alive), None)
        if carrier is None:
            return
        distance, _ = self._distance_and_bearing(platform.lat, platform.lon, carrier.lat, carrier.lon)
        if distance > 1000.0:
            return
        platform.task = "PARKED"
        platform.task_status = "IDLE"
        platform.task_assigned = False
        platform.at_home = True
        # Fixed reconnaissance teams persist across return-home events.  Their
        # original members remain the team context even after losses.

    def _maybe_confirm_attack_rearm(self, platform):
        if platform is None or platform.role != "attack_aircraft":
            return
        pending = self.pending_attack_returns.get(platform.name)
        if not pending:
            return
        carrier = next((item for item in self.platforms.values() if item.role == "carrier" and item.alive), None)
        if carrier is None:
            return
        now = float(self._current_sim_time())
        duration = float(self.config.get("scenario", {}).get("attack_rearm_duration_seconds", 600.0))
        phase = str(pending.get("phase", "returning"))
        distance, _ = self._distance_and_bearing(platform.lat, platform.lon, carrier.lat, carrier.lon)
        if phase == "returning":
            if distance > 1000.0:
                return
            pending["phase"] = "rearming"
            pending["rearm_started_at"] = now
            pending["rearm_complete_at"] = now + duration
            platform.task = "ATTACK_REARMING"
            platform.task_status = "ASSIGNED"
            platform.at_home = False
            self.last_reward_events.append({
                "type": "attack_rearm_started", "reward": 0.0,
                "platform": platform.name, "complete_at": pending["rearm_complete_at"],
            })
            return
        if phase != "rearming" or now < float(pending.get("rearm_complete_at", now + duration)):
            return
        ammo = self.attack_ammo.setdefault(platform.name, {"fox3": 1, "agm": 1})
        loaded = {"fox3": 0, "agm": 0}
        for weapon in ("fox3", "agm"):
            needed = max(0, 1 - int(ammo.get(weapon, 0)))
            supplied = min(needed, int(self.carrier_ammo_stock.get(weapon, 0)))
            ammo[weapon] = int(ammo.get(weapon, 0)) + supplied
            self.carrier_ammo_stock[weapon] = int(self.carrier_ammo_stock.get(weapon, 0)) - supplied
            loaded[weapon] = supplied
        self.pending_attack_returns.pop(platform.name, None)
        self.attack_controller.reset_expended_weapons(platform.name)
        platform.at_home = True
        group_id = pending.get("group_id")
        group = self.attack_controller.active_groups.get(group_id) if group_id else None
        if group and all(item.at_home or not item.alive for item in group.platforms):
            names = {item.name for item in group.platforms}
            for target_name, owner in list(self.attack_target_reservations.items()):
                if owner in names:
                    self.attack_target_reservations.pop(target_name, None)
            # Do not retire the fixed team.  It retains its original roster
            # and stable team id after return/rearm, including when members
            # were lost during the preceding task.
        platform.task = "PARKED"
        platform.task_status = "IDLE"
        platform.task_assigned = False
        self.last_reward_events.append({
            "type": "attack_rearmed", "reward": 0.0, "platform": platform.name,
            "aam_loaded": loaded["fox3"], "agm_loaded": loaded["agm"],
            "carrier_aam_stock": self.carrier_ammo_stock["fox3"],
            "carrier_agm_stock": self.carrier_ammo_stock["agm"],
        })
    def _maybe_start_automatic_landing(self, platform):
        """Start whole-manifest landing after a ship reaches any island shore."""
        if platform is None or platform.role != "transport":
            return
        if platform.name in self.pending_landing_unloads:
            return
        cargo = self.landing_cargo.get(platform.name, {})
        if not cargo.get("has_army", False) or cargo.get("army_landed", False):
            return
        group = next((
            item for item in self.landing_controller.active_groups.values()
            if any(member.name == platform.name for member in item.platforms)
        ), None)
        if group is None:
            return
        status = self.get_island_status(platform.lat, platform.lon)
        shore_limit = float(self.config.get("scenario", {}).get(
            "landing_shore_arrival_tolerance_m", 200.0
        ))
        if status["on_land"] or status["shore_distance_m"] > shore_limit:
            return
        shore_lat, shore_lon = status["nearest_shore_point"]
        landing_zone = {
            "name": "auto_{0}".format(status["island_name"]),
            "lat": shore_lat,
            "lon": shore_lon,
            "berth_lat": float(platform.lat),
            "berth_lon": float(platform.lon),
            "arrival_tolerance_m": shore_limit,
            "unload_radius_m": shore_limit + float(self.config.get("scenario", {}).get(
                "landing_shore_stop_offset_m", 100.0
            )),
        }
        self._send(self.landing_controller._build_simple_message(platform, "LANDING_HOLD"))
        self._send(self.landing_controller._build_unload_message(platform, landing_zone))
        now = float(self._current_sim_time())
        duration = float(self.config.get("scenario", {}).get("landing_unload_duration_seconds", 900.0))
        self.pending_landing_unloads[platform.name] = {
            "group_id": group.group_id,
            "landing_zone": landing_zone,
            "phase": "unloading",
            "unload_started_at": now,
            "unload_complete_at": now + duration,
        }
        platform.task = "LANDING_UNLOADING"
        platform.task_status = "ASSIGNED"
        platform.task_assigned = True
        platform.speed = 0.0
        self.last_reward_events.append({
            "type": "automatic_landing_started",
            "reward": 0.0,
            "platform": platform.name,
            "island": status["island_name"],
            "shore_point": [shore_lat, shore_lon],
            "complete_at": now + duration,
        })
    def _maybe_confirm_landing_unload(self, platform):
        pending = self.pending_landing_unloads.get(platform.name)
        if not pending:
            return
        zone = pending.get("landing_zone", {})
        landing_lat = float(zone.get("lat", zone.get("center_lat", platform.lat)))
        landing_lon = float(zone.get("lon", zone.get("center_lon", platform.lon)))
        berth_lat = float(zone.get("berth_lat", zone.get("navigation_lat", landing_lat)))
        berth_lon = float(zone.get("berth_lon", zone.get("navigation_lon", landing_lon)))
        arrival_tolerance = float(zone.get("arrival_tolerance_m", 200.0))
        unload_radius = float(zone.get("unload_radius_m", 350.0))
        distance_to_berth, _ = self._distance_and_bearing(platform.lat, platform.lon, berth_lat, berth_lon)
        distance_to_landing, _ = self._distance_and_bearing(platform.lat, platform.lon, landing_lat, landing_lon)
        if (distance_to_berth > arrival_tolerance or distance_to_landing > unload_radius or
                float(platform.speed) > 1.0):
            return
        if self._current_sim_time() >= float(pending.get("unload_complete_at", float("inf"))):
            self._confirm_landing_unload(platform.name)
    def _confirm_landing_unload(self, ship_name):
        pending = self.pending_landing_unloads.pop(ship_name, None)
        if not pending:
            return
        cargo = self.landing_cargo.setdefault(ship_name, {"has_army": True, "army_landed": False})
        manifest_on_ship = sorted(
            name for name, status in self.ground_status.items()
            if status.get("transport") == ship_name and status.get("on_ship", False)
        )
        if not manifest_on_ship:
            cargo["has_army"] = False
            cargo["army_landed"] = True
            return
        landing_zone = pending.get("landing_zone", {})
        for ground_name in manifest_on_ship:
            self._mark_ground_landed_from_transport(
                ship_name, landing_zone, ground_name=ground_name
            )
        cargo["has_army"] = False
        cargo["army_landed"] = True
        platform = self.platforms.get(ship_name)
        if platform:
            platform.task = "LANDING_COMPLETE"
            platform.task_status = "IDLE"
            platform.task_assigned = False
            platform.speed = 0.0
        self.landing_controller.active_groups.pop(pending.get("group_id"), None)
        self.last_reward_events.append({
            "type": "landing_unload_confirmed",
            "reward": 5.0,
            "platform": ship_name,
            "ground_units": manifest_on_ship,
            "remaining_ground_units": 0,
            "landing_complete": True,
        })
    def _select_available_platform(self, side, role, allowed_names=None):
        allowed = set(allowed_names or [])
        for platform in self.platforms.values():
            if allowed and platform.name not in allowed:
                continue
            if (platform.side == side and platform.role == role and platform.alive and
                    platform.platform_id is not None and not self._is_busy(platform)):
                return platform
        return None

    @staticmethod
    def _normalize_recon_area(area):
        radius = float(area.get("radius_m", 0.0))
        width = float(area.get("width_m", radius * 2.0))
        height = float(area.get("height_m", radius * 2.0))
        return {
            "name": area.get("name", ""),
            "lat": float(area.get("lat", area.get("center_lat", 0.0))),
            "lon": float(area.get("lon", area.get("center_lon", 0.0))),
            "alt": float(area.get("alt", area.get("default_alt_m", 9144.0))),
            "width_m": width,
            "height_m": height,
            "radius_m": radius,
            "priority": float(area.get("priority", 0.0)),
            "duration_sec": float(area.get("duration_sec", 60.0)),
        }

    def _recon_completion_threshold(self):
        return min(1.0, max(
            0.1,
            float(self.config.get("scenario", {}).get(
                "recon_coverage_complete_ratio", 0.8
            )),
        ))

    def _recon_freshness_seconds(self):
        return max(60.0, float(self.config.get("scenario", {}).get(
            "recon_area_freshness_seconds", 1800.0
        )))

    @staticmethod
    def _recon_area_name(area):
        return str((area or {}).get("name", ""))

    @staticmethod
    def _recon_valid_cells(area, grid_size=5):
        radius = float((area or {}).get("radius_m", 0.0))
        if radius <= 0.0:
            return set()
        grid_size = max(1, int(grid_size))
        cell_m = radius * 2.0 / float(grid_size)
        valid = set()
        for row in range(grid_size):
            for col in range(grid_size):
                north_m = -radius + (row + 0.5) * cell_m
                east_m = -radius + (col + 0.5) * cell_m
                if math.hypot(north_m, east_m) <= radius:
                    valid.add((row, col))
        return valid

    def _recon_area_coverage_ratio(self, area, grid_size=5):
        normalized = self._normalize_recon_area(area or {})
        valid = self._recon_valid_cells(normalized, grid_size)
        if not valid:
            return 0.0
        covered = self.recon_area_coverage.get(
            self._recon_area_name(normalized), set()
        ) & valid
        return min(1.0, len(covered) / float(len(valid)))

    def _recon_area_age_norm(self, area):
        name = self._recon_area_name(self._normalize_recon_area(area or {}))
        observed_at = self.recon_area_last_observed_time.get(name)
        if observed_at is None:
            return 1.0
        age = max(0.0, float(self._current_sim_time()) - float(observed_at))
        return min(1.0, age / self._recon_freshness_seconds())

    def _recon_area_complete(self, area):
        coverage_complete = (
            self._recon_area_coverage_ratio(area)
            >= self._recon_completion_threshold()
        )
        return coverage_complete and self._recon_area_age_norm(area) < 1.0

    def _prepare_recon_area_cycle(self, area):
        normalized = self._normalize_recon_area(area or {})
        name = self._recon_area_name(normalized)
        if self._recon_area_coverage_ratio(normalized) >= self._recon_completion_threshold() and self._recon_area_age_norm(normalized) >= 1.0:
            self.recon_area_coverage[name] = set()
            self.recon_area_last_observed_time[name] = None

    def request_retreat(self, platform_name):
        """Internal controller hook for tactical retreat.

        RETREAT is intentionally hidden from the commander action space by
        default, but sub-controllers can still call this method.
        """
        platform = self.platforms.get(platform_name)
        if not platform or platform.platform_id is None or not platform.alive:
            self.last_reward_events.append({"type": "retreat_not_ready", "reward": -1.0, "platform": platform_name})
            return False
        msg = {
            "MsgType": "AssignTask",
            "PlatformId": platform.platform_id,
            "PlatformName": platform.name,
            "Task": "RETREAT",
        }
        self._send(msg)
        platform.task = "RETREAT"
        platform.task_status = "ASSIGNED"
        platform.at_home = False
        return True

    def _begin_decision_window(self):
        self._decision_window_id += 1
        self._decision_window_sim_time = float(self._current_sim_time())
        self._decision_window_observer_targets = {}
        self._maintained_this_step = set()
        return self._decision_window_sim_time

    def _decision_window_time(self):
        return float(self._decision_window_sim_time)

    def _drain_messages(self, timeout, until_sim_time=None, until_decision_ready=False):
        # Freeze the time reference before collecting this decision window.
        # Later MoveUpdate messages must not invalidate contacts collected
        # earlier in the same UDP drain.
        self._begin_decision_window()
        if not self.sock:
            return
        end_time = time.time() + timeout
        received_messages = 0
        target_reached = False
        decision_ready_settle_deadline = None
        start_sim_time = self._current_sim_time()
        while time.time() < end_time:
            # A static socket timeout (normally 0.2 s) would make short
            # decision windows, such as 72 ms at 1000x, overrun whenever no
            # UDP packet is immediately available. Never block past the
            # current decision window.
            receive_deadline = decision_ready_settle_deadline or end_time
            remaining = max(0.0, min(end_time, receive_deadline) - time.time())
            if until_decision_ready and decision_ready_settle_deadline is not None and remaining <= 0.0:
                target_reached = True
                break
            self.sock.settimeout(min(self.message_timeout_seconds, remaining))
            try:
                payload, addr = self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            except ConnectionResetError:
                continue
            self.remote_addr = addr
            try:
                msg = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            received_messages += 1
            self._handle_message(msg)
            self._update_red_carrier_detections()
            self._update_red_recon_detections()
            self._update_red_ground_detections()
            if until_decision_ready and self.native_decision_ready and decision_ready_settle_deadline is None:
                decision_ready_settle_deadline = time.time() + 0.15
            if until_sim_time is not None and self._current_sim_time() >= float(until_sim_time):
                target_reached = True
                break
        self.last_drain_metadata = {"start_sim_time": start_sim_time, "end_sim_time": self._current_sim_time(), "until_sim_time": until_sim_time, "until_decision_ready": bool(until_decision_ready), "decision_ready": bool(self.native_decision_ready), "target_reached": target_reached, "received_messages": received_messages, "wall_timeout_seconds": float(timeout)}
        self._check_pending_attack_approaches()

    def _handle_message(self, msg):
        scenario_result = str(
            msg.get("ScenarioResult", msg.get("SCENARIO_RESULT", ""))
        ).strip().upper()
        if scenario_result:
            self.episode_result = scenario_result
        msg_type = msg.get("MsgType", "")
        if msg_type == "DecisionReady":
            self.last_decision_ready_wall_time = time.monotonic()
            ready_time = float(msg.get("WallTime", self._current_sim_time()))
            # Accept the first boundary even when it is T=0, then ignore
            # duplicate retransmissions from the already-consumed boundary.
            if (not self.native_decision_ready_seen) or ready_time > self.native_decision_ready_time + 1.0e-6:
                self.native_decision_ready = True
                self.native_decision_ready_time = ready_time
                self.native_decision_ready_seen = True
                self._native_restart_pending = False
                self._native_restart_boundary_time = None
            elif (self._native_restart_pending
                  and self._native_restart_boundary_time is not None
                  and abs(ready_time - float(self._native_restart_boundary_time)) <= 1.0e-6
                  and time.monotonic() - self._native_restart_last_send >= self._native_restart_resend_seconds):
                # The simulator proved it is still paused at this boundary;
                # retry only then, so a delayed command cannot skip a boundary.
                self._native_restart_last_send = time.monotonic()
                self._send({"MsgType": "SimRestart"})
            return
        if msg_type == "MoveUpdateBatch":
            batch_time = msg.get("WallTime")
            for update in msg.get("Updates", []):
                if not isinstance(update, dict):
                    continue
                item = dict(update)
                item.setdefault("MsgType", "MoveUpdate")
                if batch_time is not None:
                    item.setdefault("WallTime", batch_time)
                self._handle_message(item)
            return
        if msg_type in ("PlatFormAdd", "MoveUpdate"):
            self.last_platform_state_wall_time = time.monotonic()
        if msg_type == "PlatFormAdd":
            name = msg.get("PlatformName", "")
            if name in self.platforms:
                p = self.platforms[name]
                p.platform_id = msg.get("PlatformId")
                p.platform_type = msg.get("PlatformType", p.platform_type)
                loc = msg.get("Location", [])
                if len(loc) >= 3:
                    self._update_platform_motion_from_message(p, msg, loc)
                else:
                    heading = self._message_heading_deg(msg)
                    if heading is not None:
                        p.heading = heading
                    speed = self._message_speed(msg)
                    if speed is not None:
                        p.speed = speed
                # Periodic udpnet snapshots also use PlatFormAdd. Alive/HP
                # must come from combat state, not from the message type.
                self._apply_platform_combat_message(p, msg)
                self._maybe_confirm_recon_return(p)
                self._maybe_confirm_attack_approach(p)
                self._maybe_confirm_attack_rearm(p)
        elif msg_type == "PlatFormDelete":
            p = self._platform_for_message(msg)
            if p is not None:
                if self._is_transport_phase_protected(p):
                    p.alive = True
                    self.last_reward_events.append({
                        "type": "protected_transport_phase_loss_ignored",
                        "reward": 0.0,
                        "platform": p.name,
                        "role": p.role,
                    })
                else:
                    if p.max_hp > 0.0 and p.current_hp > 0.0:
                        self.last_reward_events.append({"type": "premature_platform_delete_ignored", "reward": 0.0, "platform": p.name, "current_hp": float(p.current_hp)})
                        return
                    p.alive = False
                    if p.max_hp > 0.0:
                        p.current_hp = 0.0
                    self._sync_platform_combat_caches(p)
                    if p.side == "red":
                        self.last_reward_events.append({"type": "red_loss", "reward": -5.0, "platform": p.name})
                    elif p.side == "blue":
                        shooter = self.attack_last_shooter_by_target.get(p.name, "")
                        shooter_platform = self.platforms.get(shooter)
                        if (
                            shooter
                            and shooter_platform is not None
                            and shooter_platform.role == "attack_aircraft"
                            and p.role != "attack_aircraft"
                        ) and self.debug_combat_events:
                            print(
                                "[AGM_HIT] shooter={0} target={1} destroyed".format(
                                    shooter, p.name
                                ),
                                flush=True,
                            )
                        self._mark_enemy_track_destroyed(p.name)
        elif msg_type == "CombatUpdate":
            p = self._platform_for_message(msg)
            if p is not None:
                self._apply_platform_combat_message(p, msg)
        elif msg_type == "MoveUpdate":
            loc = msg.get("Location", [])
            p = self._platform_for_message(msg)
            if p is not None and len(loc) >= 3:
                self._apply_platform_combat_message(p, msg)
                self._update_platform_motion_from_message(p, msg, loc)
                self._maybe_confirm_recon_return(p)
                self._maybe_confirm_attack_approach(p)
                self._maybe_confirm_attack_rearm(p)
                if p.role == "transport":
                    self._maybe_confirm_landing_unload(p)
                    self._maybe_start_automatic_landing(p)
        elif msg_type == "TaskAck":
            name = msg.get("PlatformName", "")
            if name in self.platforms:
                p = self.platforms[name]
                ack_task = msg.get("Task", p.task)
                stale_recon_ack = (
                    p.role == "recon_aircraft"
                    and p.task in ("RECON", "RETREAT")
                    and ack_task != p.task
                )
                stale_attack_ack = (
                    p.role == "attack_aircraft"
                    and p.task in ("ATTACK", "RETREAT", "ATTACK_REARMING")
                    and ack_task not in (p.task, "FIRE_AAM", "FIRE_AGM")
                )
                if not stale_recon_ack and not stale_attack_ack:
                    p.task = ack_task
                    p.task_status = "ACK" if msg.get("Success") else "FAILED"
                    if (
                        msg.get("Success")
                        and p.role == "attack_aircraft"
                        and p.name in self.pending_attack_approaches
                    ):
                        self._maybe_confirm_attack_approach(p)
            if msg.get("Success") and str(msg.get("Task", "")) in ("FIRE_AAM", "FIRE_AGM"):
                pending = self.pending_attack_fire_commands.get(name, {})
                target_name = str(pending.get("target_name", msg.get("TargetName", "")))
                if target_name:
                    self.register_target_reward_contributor(
                        target_name, name, "attack")
                # Stop only after AFSIM accepts the launch; an earlier HOLD
                # command can cancel the fire task before the missile leaves.
                self._hold_attack_aircraft_after_fire(name, pending)
            if not msg.get("Success"):
                self._reject_pending_attack_fire(name)
                self.pending_landing_unloads.pop(name, None)
                self.last_reward_events.append({"type": "task_rejected", "reward": -1.0, "message": msg.get("Message", "")})
        elif msg_type == "AttackContactReport":
            aircraft_name = str(
                msg.get("PlatformName")
                or msg.get("ObserverName")
                or msg.get("ReporterName")
                or ""
            )
            detections = msg.get("Detections") or []
            raw_report = {
                "reporter": aircraft_name,
                "targets": [str(det.get("Name", "")) for det in detections],
                "raw": dict(msg),
            }
            self.last_attack_contact_reports.append(raw_report)
            self.last_attack_contact_reports = self.last_attack_contact_reports[-200:]
            if self.debug_attack_contact_reports and detections:
                print(
                    "[ATTACK_CONTACT] reporter=",
                    aircraft_name,
                    "targets=",
                    raw_report["targets"],
                    flush=True,
                )
            if aircraft_name not in self.platforms:
                self.last_reward_events.append({
                    "type": "invalid_attack_contact_reporter",
                    "reward": 0.0,
                    "platform": aircraft_name,
                    "message": dict(msg),
                })
                if self.debug_attack_contact_reports:
                    print("[ATTACK_CONTACT_INVALID]", json.dumps(dict(msg), ensure_ascii=False), flush=True)
                return
            report_time = float(msg.get("ReportTime", self._decision_window_time()))
            contacts = self.attack_local_detections.setdefault(aircraft_name, {})
            reported_names = []
            for det in detections:
                target_name = str(det.get("Name", ""))
                if not target_name:
                    continue
                self.register_target_reward_contributor(
                    target_name, aircraft_name, "discover")
                contact = dict(det)
                contact["ReportTime"] = report_time
                track_age = max(0.0, float(contact.get("TrackAge", 0.0)))
                remaining_ttl = max(0.0, float(self.attack_local_track_ttl_sec) - track_age)
                contact["ExpiresAt"] = report_time + remaining_ttl
                contact["SeenDecisionWindow"] = int(self._decision_window_id)
                contacts[target_name] = contact
                reported_names.append(target_name)
                is_new = target_name not in self.detected_targets
                shared_contact = dict(contact)
                shared_contact["TrackSource"] = contact.get(
                    "TrackSource", "attack_sensor_master_track"
                )
                shared_contact["ObserverName"] = aircraft_name
                shared_contact["known"] = True
                shared_contact["alive"] = bool(contact.get("alive", True))
                if not getattr(self, "_recon_report_blocked", False):
                    self.detected_targets[target_name] = shared_contact
                self._remember_enemy_track(contact, aircraft_name, report_time)
                if is_new:
                    self.last_reward_events.append({
                        "type": "new_attack_detection",
                        "reward": 0.25,
                        "target": target_name,
                        "platform": aircraft_name,
                    })
                target_platform = self.platforms.get(target_name)
                if target_platform is not None:
                    self._apply_platform_combat_message(target_platform, contact)
                    target_platform.detected = True
                    target_platform.lat = float(contact.get("Lat", target_platform.lat))
                    target_platform.lon = float(contact.get("Lon", target_platform.lon))
                    target_platform.alt = float(contact.get("Alt", target_platform.alt))
            if aircraft_name:
                self._mark_enemy_observer_report(aircraft_name, reported_names)
        elif msg_type == "ReconReport":
            if getattr(self, "_recon_report_blocked", False):
                return
            detections = msg.get("Detections") or []
            observer_name = str(msg.get("PlatformName", ""))
            for det in detections:
                name = det.get("Name")
                if not name:
                    continue
                self.register_target_reward_contributor(
                    name, observer_name or det.get("TrackSource", ""), "discover")
                self.last_reward_events.append({
                    "type": "recon_task_detection",
                    "reward": 2.0,
                    "target": name,
                    "observer": observer_name or det.get("TrackSource", ""),
                    "report_time": float(
                        msg.get("ReportTime", self._current_sim_time())
                    ),
                    "track_age": max(0.0, float(det.get("TrackAge", 0.0))),
                })
                if name not in self.detected_targets:
                    self.last_reward_events.append({
                        "type": "new_detection",
                        "reward": 1.0,
                        "target": name,
                        "observer": observer_name or det.get("TrackSource", ""),
                    })
                self.detected_targets[name] = det
                self._remember_enemy_track(
                    det,
                    observer_name or det.get("TrackSource", "red_recon"),
                    msg.get("ReportTime", self._current_sim_time()),
                )
                if name in self.platforms:
                    self._apply_platform_combat_message(self.platforms[name], det)
                    self.platforms[name].detected = True
                    self.platforms[name].lat = float(det.get("Lat", self.platforms[name].lat))
                    self.platforms[name].lon = float(det.get("Lon", self.platforms[name].lon))
                    self.platforms[name].alt = float(det.get("Alt", self.platforms[name].alt))
            if observer_name:
                self._mark_enemy_observer_report(
                    observer_name,
                    (det.get("Name", "") for det in detections),
                )
            if msg.get("Phase") in ["complete", "lost"]:
                name = msg.get("PlatformName", "")
                if name in self.platforms:
                    self.platforms[name].task_status = "IDLE"
                    self.platforms[name].task = "PARKED"
                    self.platforms[name].at_home = msg.get("Phase") == "complete"
        elif msg_type == "AttackResult":
            result = msg.get("Result", "")
            target = msg.get("Target", "")
            name = msg.get("PlatformName", "")
            if result in ["KILL", "SHOT_AT_PRIMARY_TARGET", "SHOT_ENEMY_NO_PRIMARY_WEAPON"]:
                self.register_target_reward_contributor(target, name, "attack")
                self.last_reward_events.append({"type": "attack_result", "reward": 5.0, "target": target, "result": result, "platform": name})
            elif result in ["NO_ENEMY", "NO_COMPATIBLE_WEAPON", "AMMO_EMPTY"]:
                self._reject_pending_attack_fire(name)
                self.last_reward_events.append({"type": "attack_failed", "reward": -1.0, "target": target, "result": result, "platform": name})
            if name in self.platforms:
                platform = self.platforms[name]
                if name not in self.pending_attack_returns:
                    platform.task_status = "ASSIGNED"
                    platform.task = "ATTACK"
                    platform.at_home = False

    def _remember_enemy_track(self, detection, observer_name, report_time=None):
        name = str(detection.get("Name", detection.get("name", "")))
        if not name:
            return None
        platform = self.platforms.get(name)
        if platform is not None and platform.side != "blue":
            return None
        now = float(self._current_sim_time() if report_time is None else report_time)
        memory = dict(self.enemy_track_memory.get(name, {}))
        observers = {
            str(key): dict(value)
            for key, value in dict(memory.get("Observers", {})).items()
        }
        observer = str(observer_name or detection.get("TrackSource", "unknown"))
        observer_state = dict(observers.get(observer, {}))
        last_seen = float(observer_state.get("LastSeen", -1.0))
        was_current = bool(observer_state.get("CurrentlyDetected", False))
        new_encounter = not was_current or last_seen < 0.0 or now - last_seen > 3.0
        observer_state["FirstSeen"] = float(observer_state.get("FirstSeen", now))
        observer_state["LastSeen"] = now
        observer_state["CurrentlyDetected"] = True
        observer_state["EncounterCount"] = int(observer_state.get("EncounterCount", 0)) + (1 if new_encounter else 0)
        observer_state["UpdateCount"] = int(observer_state.get("UpdateCount", 0)) + 1
        observers[observer] = observer_state

        memory.update({
            "Name": name,
            "Type": detection.get("Type", detection.get("type", memory.get("Type", ""))),
            "Lat": float(detection.get("Lat", detection.get("lat", memory.get("Lat", 0.0)))),
            "Lon": float(detection.get("Lon", detection.get("lon", memory.get("Lon", 0.0)))),
            "Alt": float(detection.get("Alt", detection.get("alt", memory.get("Alt", 0.0)))),
            "CurrentHP": float(detection.get("CurrentHP", memory.get("CurrentHP", getattr(platform, "current_hp", 0.0)))),
            "MaxHP": float(detection.get("MaxHP", memory.get("MaxHP", getattr(platform, "max_hp", 0.0)))),
            "alive": bool(detection.get("alive", getattr(platform, "alive", memory.get("alive", True)))),
            "FirstSeen": float(memory.get("FirstSeen", now)),
            "LastSeen": now,
            "LastObserver": observer,
            "TrackSource": detection.get("TrackSource", observer),
            "Observers": observers,
            "EncounterCount": sum(int(value.get("EncounterCount", 0)) for value in observers.values()),
            "UpdateCount": int(memory.get("UpdateCount", 0)) + 1,
            "CurrentlyDetected": any(bool(value.get("CurrentlyDetected", False)) for value in observers.values()),
        })
        for key in ("Range", "Bearing", "TrackAge"):
            if key in detection:
                memory[key] = float(detection[key])
        self.enemy_track_memory[name] = memory
        self._capture_attack_target_snapshot(name, memory, now)
        return memory

    def _capture_attack_target_snapshot(self, target_name, detection, report_time):
        """Freeze values that were observable at this detection time."""
        if target_name not in self.attack_fixed_target_names:
            return
        platform = self.platforms.get(target_name)
        role = (
            platform.role if platform is not None
            else self._configured_blue_target_role(target_name)
        )
        current_hp = float(
            detection.get("CurrentHP", getattr(platform, "current_hp", 0.0))
        )
        max_hp = float(
            detection.get("MaxHP", getattr(platform, "max_hp", 0.0))
        )
        hp_norm = (
            max(0.0, min(1.0, current_hp / max_hp))
            if max_hp > 0.0 else 1.0
        )
        snapshot = {
            "name": target_name,
            "known": True,
            "alive": bool(
                detection.get("alive", getattr(platform, "alive", True))
            ),
            "type": detection.get(
                "Type",
                detection.get(
                    "type",
                    getattr(platform, "platform_type", "") or role,
                ),
            ),
            "lat": float(detection.get("Lat", detection.get("lat", 0.0))),
            "lon": float(detection.get("Lon", detection.get("lon", 0.0))),
            "alt": float(detection.get("Alt", detection.get("alt", 0.0))),
            "hp_norm": hp_norm,
            "last_seen": float(report_time),
            "aam_ammo_norm": 0.0,
            "agm_ammo_norm": 0.0,
            "ground_ammo_norm": 0.0,
            "sam_ammo_norm": 0.0,
        }
        if role == "attack_aircraft":
            ammo = self.attack_ammo.get(target_name, {"fox3": 0, "agm": 0})
            snapshot["aam_ammo_norm"] = min(
                1.0, max(0.0, float(ammo.get("fox3", 0)))
            )
            snapshot["agm_ammo_norm"] = min(
                1.0, max(0.0, float(ammo.get("agm", 0)))
            )
        elif role == "ground_force":
            ammo = self.ground_ammo.get(target_name, {"ground_fire": 0})
            snapshot["ground_ammo_norm"] = min(
                1.0, max(0.0, float(ammo.get("ground_fire", 0)) / 15.0)
            )
        elif role == "sam":
            snapshot["sam_ammo_norm"] = min(
                1.0, max(0.0, float(self.sam_ammo.get(target_name, 0)) / 10.0)
            )
        self.attack_target_snapshots[target_name] = snapshot
    def _initialize_attack_target_priors(self):
        """Seed each fixed blue target with the mission-known blue-base prior."""
        prior = dict(self.config.get("scenario", {}).get("attack_initial_target_intel", {}))
        if not prior or prior.get("lat") is None or prior.get("lon") is None:
            return
        for target_name in self.attack_fixed_target_names:
            role = self._configured_blue_target_role(target_name)
            is_air = role == "attack_aircraft" or "AIR" in str(role).upper()
            self.attack_target_snapshots[target_name] = {
                "name": target_name, "known": True, "alive": True, "type": role,
                "lat": float(prior["lat"]), "lon": float(prior["lon"]),
                "alt": float(prior.get("air_alt_m", 3000.0) if is_air else prior.get("surface_alt_m", 0.0)),
                "hp_norm": 1.0, "last_seen": None, "source": "initial_base_prior",
                "aam_ammo_norm": 0.0, "agm_ammo_norm": 0.0,
                "ground_ammo_norm": 0.0, "sam_ammo_norm": 0.0,
            }
    def _mark_enemy_observer_report(self, observer_name, current_target_names):
        observer = str(observer_name or "")
        current = set(self._decision_window_observer_targets.get(observer, set()))
        current.update(str(name) for name in current_target_names)
        self._decision_window_observer_targets[observer] = current
        for memory in self.enemy_track_memory.values():
            observers = memory.get("Observers", {})
            observer_state = observers.get(observer)
            if observer_state is not None and memory.get("Name", "") not in current:
                observer_state["CurrentlyDetected"] = False
            memory["CurrentlyDetected"] = any(
                bool(value.get("CurrentlyDetected", False))
                for value in observers.values()
            )

    def _enemy_track_memory_records(self):
        return [dict(self.enemy_track_memory[name]) for name in sorted(self.enemy_track_memory)]

    def _mark_enemy_track_destroyed(self, target_name):
        memory = self.enemy_track_memory.get(str(target_name))
        if memory is None:
            return
        memory["alive"] = False
        memory["CurrentHP"] = 0.0
        memory["CurrentlyDetected"] = False
        for observer in memory.get("Observers", {}).values():
            observer["CurrentlyDetected"] = False
        snapshot = self.attack_target_snapshots.get(str(target_name))
        if snapshot is not None:
            snapshot["alive"] = False
            snapshot["hp_norm"] = 0.0

    def _update_red_carrier_detections(self):
        if getattr(self, "_recon_report_blocked", False):
            return
        carrier_names = self.config.get("red", {}).get("carrier", [])
        if not carrier_names:
            return
        carrier = self.platforms.get(carrier_names[0])
        if carrier is None or not carrier.alive or carrier.platform_id is None:
            return

        radar_range_m = 300000.0
        current_target_names = []
        for target in self.platforms.values():
            if (
                target.side != "blue"
                or target.role != "attack_aircraft"
                or not target.alive
                or target.platform_id is None
            ):
                continue
            distance, bearing = self._distance_and_bearing(
                carrier.lat, carrier.lon, target.lat, target.lon
            )
            if distance <= radar_range_m:
                self._record_detected_target_from_recon(
                    carrier, target, distance, bearing, reward_enabled=False
                )
                current_target_names.append(target.name)
        self._mark_enemy_observer_report(carrier.name, current_target_names)

    def _update_red_recon_detections(self):
        if getattr(self, "_recon_report_blocked", False):
            return
        recon_platforms = [
            p for p in self.platforms.values()
            if p.side == "red" and p.role == "recon_aircraft" and p.alive and p.platform_id is not None
        ]
        if not recon_platforms:
            return
        blue_targets = [
            p for p in self.platforms.values()
            if p.side == "blue" and p.alive and p.platform_id is not None and p.role in ["attack_aircraft", "ground_force", "radar", "sam"]
        ]
        for recon in recon_platforms:
            if not self._red_recon_can_detect(recon):
                self._mark_enemy_observer_report(recon.name, ())
                continue
            current_target_names = []
            for target in blue_targets:
                detect_range = self._red_recon_detect_range_m(target)
                if detect_range <= 0.0:
                    continue
                distance, bearing = self._distance_and_bearing(recon.lat, recon.lon, target.lat, target.lon)
                if distance <= detect_range:
                    self._record_detected_target_from_recon(recon, target, distance, bearing)
                    current_target_names.append(target.name)
            self._mark_enemy_observer_report(recon.name, current_target_names)

    @staticmethod
    def _red_recon_detect_range_m(target: PlatformState):
        target_type = str(target.platform_type or target.role).upper()
        if target.role == "attack_aircraft" or "AIR" in target_type or "AIRCRAFT" in target_type or "FIGHTER" in target_type:
            return 100000.0
        if target.role in ["ground_force", "radar", "sam"] or any(marker in target_type for marker in ["GROUND", "RADAR", "SAM"]):
            return 100000.0
        return 0.0

    @staticmethod
    def _red_recon_can_detect(recon: PlatformState):
        if recon.at_home:
            return False
        task = str(recon.task or "").upper()
        status = str(recon.task_status or "").upper()
        return task == "RECON" or status in ["ASSIGNED", "ACK", "RECON_MOVING"]

    def _record_detected_target_from_recon(
        self,
        recon: PlatformState,
        target: PlatformState,
        distance: float,
        bearing: float,
        reward_enabled: bool = True,
    ):
        name = target.name
        if reward_enabled:
            self.last_reward_events.append({
                "type": "recon_task_detection",
                "reward": 2.0,
                "target": name,
                "observer": recon.name,
                "report_time": float(self._current_sim_time()),
                "track_age": 0.0,
            })
        if reward_enabled and name not in self.detected_targets:
            maintained = getattr(self, "_maintained_this_step", set())
            maintained.add(name)
            self._maintained_this_step = maintained
            self.last_reward_events.append({
                "type": "new_detection",
                "reward": 1.0,
                "target": name,
                "observer": recon.name,
            })
        elif reward_enabled:
            maintained = getattr(self, "_maintained_this_step", set())
            if name not in maintained:
                maintained.add(name)
                self._maintained_this_step = maintained
                self.last_reward_events.append({
                    "type": "maintained_detection",
                    "reward": 0.05,
                    "target": name,
                    "observer": recon.name,
                })
        target.detected = True
        record = {
            "Name": name,
            "Type": target.platform_type or target.role,
            "Lat": target.lat,
            "Lon": target.lon,
            "Alt": target.alt,
            "Range": float(distance),
            "Bearing": float(bearing),
            "TrackSource": recon.name,
            "CurrentHP": target.current_hp,
            "MaxHP": target.max_hp,
            "alive": target.alive,
        }
        self.detected_targets[name] = record
        self._remember_enemy_track(record, recon.name)

    def _update_red_ground_detections(self):
        ground_platforms = [
            p for p in self.platforms.values()
            if p.side == "red" and p.role == "ground_force" and p.alive and p.platform_id is not None
        ]
        blue_ground = [
            p for p in self.platforms.values()
            if p.side == "blue" and p.role in ["ground_force", "radar", "sam"] and p.alive and p.platform_id is not None
        ]
        if not ground_platforms or not blue_ground:
            self.ground_detected_targets.clear()
            return
        detect_range = float(self.ground_state_config.get("normalization", {}).get("ground_detect_range_m", 10000.0))
        current_contacts = {}
        for observer in ground_platforms:
            status = self.ground_status.get(observer.name, {"on_ship": False, "landed": True})
            if status.get("on_ship", False) or not status.get("landed", True):
                continue
            for target in blue_ground:
                distance, bearing = self._distance_and_bearing(observer.lat, observer.lon, target.lat, target.lon)
                if distance <= detect_range:
                    previous = current_contacts.get(target.name)
                    if previous is None or distance < float(previous.get("Range", float("inf"))):
                        current_contacts[target.name] = self._ground_detection_record(observer, target, distance, bearing)
        for name, record in current_contacts.items():
            self._remember_enemy_track(record, record.get("TrackSource", "red_ground"))
            if name not in self.ground_detected_targets:
                self.last_reward_events.append({"type": "ground_new_detection", "reward": 0.5, "target": name, "platform": record.get("TrackSource", "")})
        self.ground_detected_targets = current_contacts

    def _ground_detection_record(self, observer: PlatformState, target: PlatformState, distance: float, bearing: float):
        return {
            "Name": target.name,
            "Type": target.platform_type or target.role,
            "Lat": target.lat,
            "Lon": target.lon,
            "Alt": target.alt,
            "Range": float(distance),
            "Bearing": float(bearing),
            "TrackSource": observer.name,
            "DetectedByRole": "red_ground_force",
            "LastSeenStep": int(self.step_count),
            "CurrentHP": target.current_hp,
            "MaxHP": target.max_hp,
        }

    def _update_platform_motion_from_message(self, platform: PlatformState, msg, loc):
        new_lat, new_lon, new_alt = float(loc[0]), float(loc[1]), float(loc[2])
        old_lat, old_lon, old_alt = platform.lat, platform.lon, platform.alt
        old_time = float(getattr(platform, "last_update", 0.0) or 0.0)
        new_time = float(msg.get("WallTime", old_time))
        delta_time = new_time - old_time

        explicit_speed = self._message_speed(msg)
        heading = self._message_heading_deg(msg)
        if delta_time > 0.0 and old_lat != 0.0 and old_lon != 0.0:
            north_m, east_m = self._relative_north_east(old_lat, old_lon, new_lat, new_lon)
            platform.velocity_north_mps = north_m / delta_time
            platform.velocity_east_mps = east_m / delta_time
            platform.velocity_up_mps = (new_alt - old_alt) / delta_time
            measured_speed = math.hypot(platform.velocity_north_mps, platform.velocity_east_mps)
            platform.speed = explicit_speed if explicit_speed is not None else measured_speed
        elif explicit_speed is not None:
            effective_heading = float(platform.heading if heading is None else heading)
            heading_rad = math.radians(effective_heading)
            platform.speed = explicit_speed
            platform.velocity_north_mps = explicit_speed * math.cos(heading_rad)
            platform.velocity_east_mps = explicit_speed * math.sin(heading_rad)
            platform.velocity_up_mps = 0.0
        if heading is not None:
            platform.heading = heading

        platform.lat, platform.lon, platform.alt = new_lat, new_lon, new_alt
        platform.last_update = new_time

    @staticmethod
    def _message_speed(msg):
        for key in ("Speed", "speed", "GroundSpeed", "ground_speed", "GroundSpeedMps", "speed_mps"):
            if key in msg and msg.get(key) is not None:
                return float(msg.get(key))
        velocity = msg.get("Velocity") or msg.get("velocity")
        if isinstance(velocity, (list, tuple)) and len(velocity) >= 2:
            components = [float(value) for value in velocity[:3]]
            return math.sqrt(sum(value * value for value in components))
        return None

    @staticmethod
    def _message_heading_deg(msg):
        if "Heading" in msg and msg.get("Heading") is not None:
            return float(msg.get("Heading"))
        if "heading" in msg and msg.get("heading") is not None:
            return float(msg.get("heading"))
        orientation = msg.get("Orientation") or msg.get("orientation")
        if isinstance(orientation, (list, tuple)) and len(orientation) >= 1:
            return (math.degrees(float(orientation[0])) + 360.0) % 360.0
        return None

    def _is_transport_phase_protected(self, platform: PlatformState):
        if platform.side != "red":
            return False
        if platform.role == "transport":
            cargo = self.landing_cargo.get(platform.name, {})
            return bool(cargo.get("has_army", False)) and not bool(cargo.get("army_landed", False))
        if platform.role == "ground_force":
            status = self.ground_status.get(platform.name, {})
            return bool(status.get("on_ship", False)) and not bool(status.get("landed", False))
        return False

    def _is_busy(self, platform: PlatformState):
        if platform.role in ("recon_aircraft", "attack_aircraft") and platform.task_assigned:
            return True
        return platform.task_status not in ["IDLE", "FAILED"] and platform.task != "PARKED"
    def get_action_mask(self):
        return np.zeros(0, dtype=np.float32)

    def _known_attack_targets_in_area(self, area):
        normalized = self._normalize_recon_area(area or {})
        targets = []
        for name in self.detected_targets:
            info = self._attack_target_info(name)
            if info and self._is_valid_attack_target(info) and self._attack_target_in_area(info, normalized):
                targets.append(info)
        return targets

    def _attack_target_in_area(self, target, area):
        radius_m = max(1.0, float((area or {}).get("radius_m", 0.0)))
        center_lat = float((area or {}).get("lat", (area or {}).get("center_lat", 0.0)))
        center_lon = float((area or {}).get("lon", (area or {}).get("center_lon", 0.0)))
        distance_m, _ = self._distance_and_bearing(
            center_lat,
            center_lon,
            float(target.get("lat", center_lat)),
            float(target.get("lon", center_lon)),
        )
        return distance_m <= radius_m

    def _landing_stage_id(self):
        """Return the three-level landing phase used for priority switching."""
        if float(self._landed_ground_ratio()) > 0.0:
            return 2.0
        if self._is_landing_window_open():
            return 1.0
        return 0.0

    def _is_landing_window_open(self):
        """Return whether combat conditions or the time fallback allow landing."""
        status = self.get_landing_window_status(include_open=False)
        return bool(status["combat_conditions_met"] or status["time_override_met"])

    def get_landing_window_status(self, include_open=True):
        scenario_cfg = self.config.get("scenario", {})
        required_blue_air_kills = int(scenario_cfg.get("landing_blue_air_kills_required", 5))
        max_alive_blue_sams = int(scenario_cfg.get("landing_blue_sam_alive_max", 1))
        force_open_time = float(scenario_cfg.get("landing_force_open_time_seconds", 10800.0))
        destroyed_blue_air = self._destroyed_count("blue", "attack_aircraft")
        alive_blue_sams = self._alive_count("blue", "sam")
        current_time = float(self._current_sim_time())
        combat_conditions_met = (
            destroyed_blue_air >= required_blue_air_kills
            and alive_blue_sams <= max_alive_blue_sams
        )
        time_override_met = current_time >= force_open_time
        status = {
            "destroyed_blue_air": destroyed_blue_air,
            "required_blue_air_kills": required_blue_air_kills,
            "alive_blue_sams": alive_blue_sams,
            "max_alive_blue_sams": max_alive_blue_sams,
            "combat_conditions_met": combat_conditions_met,
            "time_override_met": time_override_met,
            "time_seconds": current_time,
            "force_open_time_seconds": force_open_time,
            "trigger_reason": (
                "combat_conditions" if combat_conditions_met
                else "time_override" if time_override_met
                else "closed"
            ),
        }
        if include_open:
            status["open"] = bool(combat_conditions_met or time_override_met)
        return status
    def _obs_dim(self):
        if self.observation_mode == "commander":
            return len(self.commander_state_fields)
        return len(self._legacy_observation())

    def get_observation(self):
        if self.observation_mode == "commander":
            return self.get_commander_state()
        return self._legacy_observation()

    def _legacy_observation(self):
        features = []
        for name in sorted(self.platforms.keys()):
            p = self.platforms[name]
            features.extend([
                1.0 if p.alive else 0.0,
                self._norm_lat(p.lat),
                self._norm_lon(p.lon),
                self._norm_alt(p.alt),
                1.0 if self._is_busy(p) else 0.0,
                1.0 if p.detected else 0.0,
                self._role_code(p.role),
                1.0 if p.side == "red" else 0.0,
            ])
        for target in self._blue_target_names():
            features.append(1.0 if target in self.detected_targets else 0.0)
        features.extend([
            self._mission_time_norm(),
            len(self.detected_targets) / max(1.0, float(len(self._blue_target_names()))),
            self._alive_ratio("red"),
            self._alive_ratio("blue"),
        ])
        return np.asarray(features, dtype=np.float32)

    def get_commander_summary_state_fields(self):
        """Return the original task-level summary shared with the Critic."""
        return list(self.commander_state_fields)

    def _commander_red_platforms(self):
        return [platform for platform in self.platforms.values() if platform.side == "red"]

    def _commander_blue_contact_capacity(self):
        return sum(1 for platform in self.platforms.values() if platform.side == "blue")

    def get_commander_state_fields(self):
        """Return red-view fields available to the deployable high-level Actor."""
        fields = list(self.commander_state_fields)
        for platform in self._commander_red_platforms():
            fields.extend(
                "own.{0}.{1}".format(platform.name, feature)
                for feature in self.COMMANDER_OWN_ENTITY_FEATURES
            )
        for slot in range(1, self._commander_blue_contact_capacity() + 1):
            fields.extend(
                "known_blue.slot_{0:02d}.{1}".format(slot, feature)
                for feature in self.COMMANDER_KNOWN_CONTACT_FEATURES
            )
        return fields

    def get_commander_summary_state_dict(self):
        values = self._build_commander_state_values()
        return {field: values.get(field, 0.0) for field in self.commander_state_fields}

    def _commander_own_entity_state_values(self, platform):
        critic = self._critic_entity_state_values(platform)
        return {
            "registered": critic["registered"],
            "alive": critic["alive"],
            "indestructible": critic["indestructible"],
            "role_id_norm": min(1.0, max(0.0, critic["role_id"] / 7.0)),
            "lat_norm": self._norm_lat(platform.lat),
            "lon_norm": self._norm_lon(platform.lon),
            "alt_norm": critic["alt_norm"],
            "velocity_east_norm": 0.5 * (critic["velocity_east_norm"] + 1.0),
            "velocity_north_norm": 0.5 * (critic["velocity_north_norm"] + 1.0),
            "velocity_up_norm": 0.5 * (critic["velocity_up_norm"] + 1.0),
            "hp_norm": critic["hp_norm"],
            "fire_cooldown_remaining_norm": critic["fire_cooldown_remaining_norm"],
            "combat_lock_remaining_norm": critic["combat_lock_remaining_norm"],
            "aam_ammo_norm": critic["aam_ammo_norm"],
            "agm_ammo_norm": critic["agm_ammo_norm"],
            "ground_ammo_norm": critic["ground_ammo_norm"],
            "aam_reserve_norm": critic["aam_reserve_norm"],
            "agm_reserve_norm": critic["agm_reserve_norm"],
            "task_id_norm": min(1.0, max(0.0, critic["task_id"] / 9.0)),
            "at_home": critic["at_home"],
            "on_ship": critic["on_ship"],
            "landed": critic["landed"],
            "has_army": critic["has_army"],
            "army_landed": critic["army_landed"],
            "ammo_reserve_unlimited": critic["ammo_reserve_unlimited"],
            "rearm_remaining_norm": critic["rearm_remaining_norm"],
        }

    def _commander_known_blue_records_by_slot(self):
        records = {
            str(name): dict(record)
            for name, record in self.enemy_track_memory.items()
            if str(name)
        }
        for source in (self.detected_targets, self.ground_detected_targets):
            for name, detection in source.items():
                name = str(name)
                if not name:
                    continue
                record = records.setdefault(name, {})
                for key, value in detection.items():
                    record.setdefault(key, value)
                record.setdefault("Name", name)
                record["CurrentlyDetected"] = True

        capacity = self._commander_blue_contact_capacity()
        candidates = []
        for name, record in records.items():
            platform = self.platforms.get(name)
            if platform is not None and platform.side != "blue":
                continue
            first_seen = float(record.get("FirstSeen", record.get("LastSeen", 0.0)))
            candidates.append((first_seen, name, record))
        for _first_seen, name, _record in sorted(candidates):
            if name in self.commander_contact_slots:
                continue
            used = set(self.commander_contact_slots.values())
            free_slot = next((slot for slot in range(1, capacity + 1) if slot not in used), None)
            if free_slot is None:
                break
            self.commander_contact_slots[name] = free_slot

        by_slot = {}
        for _first_seen, name, record in candidates:
            slot = self.commander_contact_slots.get(name)
            if slot is not None and 1 <= slot <= capacity:
                by_slot[slot] = (name, record)
        return by_slot

    def _commander_contact_role_id(self, name, record):
        platform = self.platforms.get(name)
        if platform is not None:
            return self.CRITIC_ROLE_IDS.get(platform.role, 0)
        target_type = str(record.get("Type", record.get("type", ""))).upper()
        if "SAM" in target_type:
            return self.CRITIC_ROLE_IDS["sam"]
        if "RADAR" in target_type:
            return self.CRITIC_ROLE_IDS["radar"]
        if "GROUND" in target_type or "FORCE" in target_type:
            return self.CRITIC_ROLE_IDS["ground_force"]
        if "AIR" in target_type or "FIGHTER" in target_type:
            return self.CRITIC_ROLE_IDS["attack_aircraft"]
        if "BASE" in target_type:
            return self.CRITIC_ROLE_IDS["base"]
        return 0

    def _commander_known_contact_state_values(self, name, record):
        now = float(self._current_sim_time())
        last_seen = float(record.get("LastSeen", now))
        max_track_age = float(
            self.commander_state_config.get("normalization", {}).get(
                "max_track_age_seconds", 3600.0
            )
        )
        current_hp = record.get("CurrentHP")
        max_hp = record.get("MaxHP")
        hp_known = current_hp is not None and max_hp is not None and float(max_hp) > 0.0
        hp_norm = (
            min(1.0, max(0.0, float(current_hp) / float(max_hp)))
            if hp_known else 0.0
        )
        return {
            "known": 1.0,
            "currently_detected": 1.0 if bool(record.get("CurrentlyDetected", False)) else 0.0,
            "role_id_norm": min(1.0, max(0.0, self._commander_contact_role_id(name, record) / 7.0)),
            "lat_norm": self._norm_lat(float(record.get("Lat", record.get("lat", 0.0)))),
            "lon_norm": self._norm_lon(float(record.get("Lon", record.get("lon", 0.0)))),
            "alt_norm": self._norm_alt(float(record.get("Alt", record.get("alt", 0.0))), 12000.0),
            "hp_known": 1.0 if hp_known else 0.0,
            "hp_norm": hp_norm,
            "alive_last_known": 1.0 if bool(record.get("alive", True)) else 0.0,
            "track_age_norm": min(1.0, max(0.0, now - last_seen) / max(1.0, max_track_age)),
        }

    def get_commander_state_dict(self):
        values = self.get_commander_summary_state_dict()
        for platform in self._commander_red_platforms():
            prefix = "own.{0}.".format(platform.name)
            entity = self._commander_own_entity_state_values(platform)
            for feature in self.COMMANDER_OWN_ENTITY_FEATURES:
                values[prefix + feature] = entity[feature]

        contacts = self._commander_known_blue_records_by_slot()
        for slot in range(1, self._commander_blue_contact_capacity() + 1):
            prefix = "known_blue.slot_{0:02d}.".format(slot)
            contact = contacts.get(slot)
            contact_values = (
                self._commander_known_contact_state_values(*contact)
                if contact is not None else {}
            )
            for feature in self.COMMANDER_KNOWN_CONTACT_FEATURES:
                values[prefix + feature] = contact_values.get(feature, 0.0)
        return values

    def get_commander_state(self):
        values = self.get_commander_state_dict()
        return np.asarray(
            [values.get(field, 0.0) for field in self.get_commander_state_fields()],
            dtype=np.float32,
        )

    def get_critic_global_state_fields(self):
        fields = ["global." + field for field in self.get_commander_summary_state_fields()]
        for platform_name in self.platforms:
            fields.extend(
                "entity.{0}.{1}".format(platform_name, feature)
                for feature in self.CRITIC_ENTITY_FEATURES
            )
        return fields

    def get_critic_global_state_dict(self):
        values = {
            "global." + field: value
            for field, value in self.get_commander_summary_state_dict().items()
        }
        for platform in self.platforms.values():
            prefix = "entity.{0}.".format(platform.name)
            entity_values = self._critic_entity_state_values(platform)
            for feature in self.CRITIC_ENTITY_FEATURES:
                values[prefix + feature] = entity_values.get(feature, 0.0)
        return values

    def get_critic_global_state(self):
        values = self.get_critic_global_state_dict()
        return np.asarray(
            [values[field] for field in self.get_critic_global_state_fields()],
            dtype=np.float32,
        )

    def get_critic_entity_state_trace(self, platform_name=None):
        """Return physical/cache values and their final 27-field state values.

        This is a diagnostics-only view of the exact values consumed by
        ``_critic_entity_state_values``.  Physical motion fields have already
        been decoded from UDP messages but have not yet been normalized.
        Task, ammunition and cargo fields come from their named
        authoritative environment caches.
        """
        if platform_name is None:
            platforms = list(self.platforms.values())
        else:
            platform = self.platforms.get(str(platform_name))
            if platform is None:
                raise KeyError("unknown platform: {0}".format(platform_name))
            platforms = [platform]
        now = float(self._current_sim_time())
        traces = {}
        for platform in platforms:
            ground_status = self.ground_status.get(platform.name, {})
            cargo = self.landing_cargo.get(platform.name, {})
            attack_ammo = self.attack_ammo.get(
                platform.name, {"fox3": 0, "agm": 0}
            )
            ground_ammo = self.ground_ammo.get(
                platform.name, {"ground_fire": 0}
            )
            east_m, north_m, east_scale_m, north_scale_m = (
                self._critic_local_position_components(platform.lat, platform.lon)
            )
            fire_remaining = max(0.0, float(platform.fire_cooldown_until) - now)
            lock_remaining = max(0.0, float(platform.combat_lock_until) - now)
            rearm_complete_at = self._attack_rearm_complete_at(platform)
            rearm_remaining = max(0.0, rearm_complete_at - now)
            is_red_carrier = platform.side == "red" and platform.role == "carrier"
            is_blue_base = platform.side == "blue" and platform.role == "base"
            ammo_reserve_unlimited = is_blue_base
            aam_reserve = (
                int(self.carrier_ammo_stock.get("fox3", 0))
                if is_red_carrier else ("infinite" if is_blue_base else None)
            )
            agm_reserve = (
                int(self.carrier_ammo_stock.get("agm", 0))
                if is_red_carrier else ("infinite" if is_blue_base else None)
            )
            initial_carrier_stock = self._initial_carrier_ammo_stock()
            raw = {
                "platform_id": platform.platform_id,
                "alive_bool": bool(platform.alive),
                "indestructible_bool": bool(platform.indestructible),
                "side_label": str(platform.side),
                "role_label": str(platform.role),
                "lat_deg": float(platform.lat),
                "lon_deg": float(platform.lon),
                "alt_m": float(platform.alt),
                "east_m": float(east_m),
                "north_m": float(north_m),
                "east_scale_m": float(east_scale_m),
                "north_scale_m": float(north_scale_m),
                "heading_deg": float(platform.heading),
                "speed_mps": float(platform.speed),
                "velocity_east_mps": float(platform.velocity_east_mps),
                "velocity_north_mps": float(platform.velocity_north_mps),
                "velocity_up_mps": float(platform.velocity_up_mps),
                "last_update_sim_time_sec": float(platform.last_update),
                "current_hp": float(platform.current_hp),
                "max_hp": float(platform.max_hp),
                "sim_time_sec": now,
                "fire_cooldown_until_sec": float(platform.fire_cooldown_until),
                "fire_cooldown_remaining_sec": fire_remaining,
                "combat_lock_until_sec": float(platform.combat_lock_until),
                "combat_lock_remaining_sec": lock_remaining,
                "rearm_complete_at_sec": rearm_complete_at,
                "rearm_remaining_sec": rearm_remaining,
                "aam_rounds": int(attack_ammo.get("fox3", 0)),
                "agm_rounds": int(attack_ammo.get("agm", 0)),
                "ground_rounds": int(ground_ammo.get("ground_fire", 0)),
                "aam_reserve_rounds": aam_reserve,
                "agm_reserve_rounds": agm_reserve,
                "initial_aam_reserve_rounds": (
                    int(initial_carrier_stock["fox3"])
                    if is_red_carrier else ("infinite" if is_blue_base else None)
                ),
                "initial_agm_reserve_rounds": (
                    int(initial_carrier_stock["agm"])
                    if is_red_carrier else ("infinite" if is_blue_base else None)
                ),
                "ammo_reserve_unlimited_bool": bool(ammo_reserve_unlimited),
                "task_label": str(platform.task or ""),
                "task_status_label": str(platform.task_status or ""),
                "at_home_bool": bool(platform.at_home),
                "at_home_applicable_bool": bool(
                    platform.side == "red"
                    and platform.role in ("recon_aircraft", "attack_aircraft")
                ),
                "on_ship_bool": bool(ground_status.get("on_ship", False)),
                "landed_bool": bool(ground_status.get("landed", False)),
                "has_army_bool": bool(cargo.get("has_army", False)),
                "army_landed_bool": bool(cargo.get("army_landed", False)),
            }
            traces[platform.name] = {
                "raw": raw,
                "state": self._critic_entity_state_values(platform),
                "sources": {
                    "identity_motion_hp": "platform_state_from_udp_or_scenario_initial_value",
                    "attack_ammo": "attack_ammo_cache_updated_by_udp_and_fire_commands",
                    "ground_ammo": "ground_ammo_cache_updated_by_fire_commands",
                    "ammo_reserve": (
                        "red_carrier_stock_or_blue_base_unlimited_rearm_logic"
                    ),
                    "task": "platform_task_and_task_status",
                    "landing": "ground_status_and_landing_cargo_caches",
                },
            }
        return traces

    def _critic_entity_state_values(self, platform):
        aam_ammo = 0.0
        agm_ammo = 0.0
        ground_ammo = 0.0
        if platform.role == "attack_aircraft":
            ammo = self.attack_ammo.get(platform.name, {"fox3": 0, "agm": 0})
            aam_ammo = min(1.0, max(0.0, float(ammo.get("fox3", 0))))
            agm_ammo = min(1.0, max(0.0, float(ammo.get("agm", 0))))
        elif platform.role == "ground_force":
            ammo = self.ground_ammo.get(platform.name, {"ground_fire": 0})
            ground_ammo = min(1.0, max(0.0, float(ammo.get("ground_fire", 0)) / 15.0))

        ground_status = self.ground_status.get(platform.name, {})
        cargo = self.landing_cargo.get(platform.name, {})
        east_norm, north_norm = self._critic_local_position_norm(platform.lat, platform.lon)
        velocity_scale = 500.0 / 3.6
        aam_reserve = 0.0
        agm_reserve = 0.0
        ammo_reserve_unlimited = 0.0
        rearm_remaining = 0.0
        if platform.role == "attack_aircraft":
            rearm_complete_at = self._attack_rearm_complete_at(platform)
            rearm_duration = float(
                self.config.get("scenario", {}).get(
                    "attack_rearm_duration_seconds", 600.0
                )
            )
            rearm_remaining = self._combat_remaining_norm(
                rearm_complete_at, rearm_duration
            )
        if platform.side == "red" and platform.role == "carrier":
            initial_stock = self._initial_carrier_ammo_stock()
            aam_reserve = min(
                1.0,
                max(
                    0.0,
                    float(self.carrier_ammo_stock.get("fox3", 0))
                    / float(max(1, initial_stock["fox3"])),
                ),
            )
            agm_reserve = min(
                1.0,
                max(
                    0.0,
                    float(self.carrier_ammo_stock.get("agm", 0))
                    / float(max(1, initial_stock["agm"])),
                ),
            )
        elif platform.side == "blue" and platform.role == "base":
            # Blue fighters reload both weapons at blue_base without reading
            # or decrementing a base stock counter in the AFSIM scenario.
            aam_reserve = 1.0
            agm_reserve = 1.0
            ammo_reserve_unlimited = 1.0
        fire_cooldown_duration = 600.0 if (
            platform.side == "blue" and platform.role == "sam"
        ) else 300.0

        return {
            "registered": 1.0 if platform.platform_id is not None else 0.0,
            "alive": 1.0 if platform.alive else 0.0,
            "indestructible": 1.0 if platform.indestructible else 0.0,
            "side_id": 0.0 if platform.side == "red" else 1.0,
            "role_id": float(self.CRITIC_ROLE_IDS.get(platform.role, 0)),
            "east_norm": east_norm,
            "north_norm": north_norm,
            "alt_norm": self._norm_alt(platform.alt, 12000.0),
            "velocity_east_norm": min(1.0, max(-1.0, float(platform.velocity_east_mps) / velocity_scale)),
            "velocity_north_norm": min(1.0, max(-1.0, float(platform.velocity_north_mps) / velocity_scale)),
            "velocity_up_norm": min(1.0, max(-1.0, float(platform.velocity_up_mps) / velocity_scale)),
            "hp_norm": 0.0 if (
                platform.indestructible or platform.max_hp <= 0.0
            ) else self._hp_norm(platform),
            "fire_cooldown_remaining_norm": self._combat_remaining_norm(platform.fire_cooldown_until, fire_cooldown_duration),
            "combat_lock_remaining_norm": self._combat_remaining_norm(platform.combat_lock_until, 300.0),
            "aam_ammo_norm": aam_ammo,
            "agm_ammo_norm": agm_ammo,
            "ground_ammo_norm": ground_ammo,
            "aam_reserve_norm": aam_reserve,
            "agm_reserve_norm": agm_reserve,
            "task_id": float(self._critic_task_id(platform)),
            "at_home": 1.0 if (
                platform.side == "red"
                and platform.role in ("recon_aircraft", "attack_aircraft")
                and platform.at_home
            ) else 0.0,
            "on_ship": 1.0 if ground_status.get("on_ship", False) else 0.0,
            "landed": 1.0 if ground_status.get("landed", False) else 0.0,
            "has_army": 1.0 if cargo.get("has_army", False) else 0.0,
            "army_landed": 1.0 if cargo.get("army_landed", False) else 0.0,
            "ammo_reserve_unlimited": ammo_reserve_unlimited,
            "rearm_remaining_norm": rearm_remaining,
        }

    def _attack_rearm_complete_at(self, platform):
        if platform.side == "red":
            pending = self.pending_attack_returns.get(platform.name, {})
            if str(pending.get("phase", "")) == "rearming":
                return float(pending.get("rearm_complete_at", 0.0))
        return float(getattr(platform, "rearm_complete_at", 0.0) or 0.0)

    def _critic_target_detected_by_red(self, platform_name):
        if platform_name in self.detected_targets or platform_name in self.ground_detected_targets:
            return True
        memory = self.enemy_track_memory.get(platform_name, {})
        return bool(memory.get("CurrentlyDetected", False))

    def _critic_local_position_norm(self, lat, lon):
        east_m, north_m, east_scale, north_scale = (
            self._critic_local_position_components(lat, lon)
        )
        return (
            min(1.0, max(-1.0, east_m / east_scale)),
            min(1.0, max(-1.0, north_m / north_scale)),
        )

    def _critic_local_position_components(self, lat, lon):
        lat_min = float(self.bounds.get("lat_min", 23.5))
        lat_max = float(self.bounds.get("lat_max", 25.8))
        lon_min = float(self.bounds.get("lon_min", 118.8))
        lon_max = float(self.bounds.get("lon_max", 122.2))
        origin_lat = (lat_min + lat_max) / 2.0
        origin_lon = (lon_min + lon_max) / 2.0
        cos_lat = max(0.1, math.cos(math.radians(origin_lat)))
        north_m = (float(lat) - origin_lat) * 111320.0
        east_m = (float(lon) - origin_lon) * 111320.0 * cos_lat
        north_scale = max(1.0, (lat_max - lat_min) * 111320.0 / 2.0)
        east_scale = max(1.0, (lon_max - lon_min) * 111320.0 * cos_lat / 2.0)
        return east_m, north_m, east_scale, north_scale

    def _critic_task_id(self, platform):
        task = "{0} {1}".format(platform.task or "", platform.task_status or "").upper()
        if not platform.alive or "DESTROY" in task or "DEFEAT" in task:
            return self.CRITIC_TASK_IDS["destroyed"]
        if "REARM" in task or "SERVICE" in task:
            return self.CRITIC_TASK_IDS["service"]
        if "RETURN" in task or "RTB" in task:
            return self.CRITIC_TASK_IDS["return"]
        if "UNLOAD" in task:
            return self.CRITIC_TASK_IDS["unloading"]
        if platform.role == "attack_aircraft" and (
            "ATTACK" in task
            or "FIRE_" in task
            or "INTERCEPT" in task
            or "CAP_PATROL" in task
        ):
            return self.CRITIC_TASK_IDS["attack"]
        if "GROUND" in task or "CAPTURE" in task:
            return self.CRITIC_TASK_IDS["ground"]
        if "LANDING" in task or "TRANSPORT" in task:
            return self.CRITIC_TASK_IDS["landing"]
        if "ATTACK" in task or "FIRE_" in task:
            return self.CRITIC_TASK_IDS["attack"]
        if "RECON" in task or "SEARCH" in task:
            return self.CRITIC_TASK_IDS["recon"]
        if "MOVE" in task or "MOVING" in task:
            return self.CRITIC_TASK_IDS["moving"]
        physical_speed = max(
            abs(float(platform.speed)),
            math.hypot(
                float(platform.velocity_east_mps),
                float(platform.velocity_north_mps),
            ),
            abs(float(platform.velocity_up_mps)),
        )
        if physical_speed > 1.0:
            return self.CRITIC_TASK_IDS["moving"]
        return self.CRITIC_TASK_IDS["idle"]

    def _build_commander_state_values(self):
        total_blue_targets = max(1, len(self._blue_target_names()))
        blue_air_total = self._role_total("blue", "attack_aircraft")
        blue_sam_total = self._role_total("blue", "sam")
        blue_radar_total = self._role_total("blue", "radar")
        blue_ground_total = self._role_total("blue", "ground_force")
        red_recon_total = self._role_total("red", "recon_aircraft")
        red_attack_total = self._role_total("red", "attack_aircraft")
        red_transport_total = self._role_total("red", "transport")
        red_ground_total = self._role_total("red", "ground_force")
        known_air = self._known_target_count(["AIR", "AIRCRAFT", "FIGHTER"])
        known_sam = self._known_target_count(["SAM"])
        known_radar = self._known_target_count(["RADAR"])
        known_ground = self._known_target_count(["GROUND", "FORCE"])
        scenario_cfg = self.config.get("scenario", {})
        required_blue_air_kills = max(1, int(scenario_cfg.get("landing_blue_air_kills_required", 5)))
        max_alive_blue_sams = int(scenario_cfg.get("landing_blue_sam_alive_max", 1))
        alive_blue_sams = self._alive_count("blue", "sam")
        max_commander_distance = float(self.commander_state_config.get("normalization", {}).get("max_distance_m", 300000.0))
        values = {
            "progress": self._mission_time_norm(),
            "known_target_ratio": min(1.0, len(self.detected_targets) / float(total_blue_targets)),
            "known_blue_air_ratio": min(1.0, known_air / float(max(1, blue_air_total))),
            "known_blue_sam_ratio": min(1.0, known_sam / float(max(1, blue_sam_total))),
            "known_blue_radar_ratio": min(1.0, known_radar / float(max(1, blue_radar_total))),
            "known_blue_ground_ratio": min(1.0, known_ground / float(max(1, blue_ground_total))),
            "recon_coverage_ratio": self._commander_recon_coverage_ratio(),
            "last_recon_gain_norm": self._last_recon_gain_norm(),
            "red_recon_available_ratio": self._available_ratio("red", "recon_aircraft"),
            "red_attack_available_ratio": self._available_ratio("red", "attack_aircraft"),
            "red_transport_available_ratio": self._available_ratio("red", "transport"),
            "red_ground_available_ratio": self._ground_available_ratio(),
            "red_landed_ground_ratio": self._landed_ground_ratio(),
            "blue_air_destroyed_ratio": self._destroyed_ratio("blue", "attack_aircraft"),
            "blue_sam_destroyed_ratio": self._destroyed_ratio("blue", "sam"),
            "blue_radar_destroyed_ratio": self._destroyed_ratio("blue", "radar"),
            "blue_ground_destroyed_ratio": self._destroyed_ratio("blue", "ground_force"),
            "landing_window_open": 1.0 if self._is_landing_window_open() else 0.0,
            "landing_stage_id": self._landing_stage_id(),
            "recon_in_progress": 1.0 if self._task_in_progress("RECON") else 0.0,
            "attack_in_progress": 1.0 if self._task_in_progress("ATTACK") else 0.0,
            "landing_in_progress": 1.0 if self._task_in_progress("LANDING") else 0.0,
            "ground_in_progress": 1.0 if self._task_in_progress("GROUND") else 0.0,
            "active_recon_group_count_norm": self._active_group_count_norm(self.recon_controller, red_recon_total),
            "active_attack_group_count_norm": self._active_group_count_norm(self.attack_controller, red_attack_total),
            "active_landing_group_count_norm": self._active_group_count_norm(self.landing_controller, red_transport_total),
            "active_ground_group_count_norm": self._active_group_count_norm(self.ground_controller, red_ground_total),
            "ground_available": 1.0 if any(self._is_landed_ground_available(p) and p.alive and not self._is_busy(p) and p.platform_id is not None for p in self.platforms.values() if p.side == "red" and p.role == "ground_force") else 0.0,
            "capture_condition": 1.0 if self._any_capture_condition() else 0.0,
            "red_ground_alive_ratio": self._role_alive_ratio("red", "ground_force"),
            "red_attack_alive_ratio": self._role_alive_ratio("red", "attack_aircraft"),
            "red_recon_alive_ratio": self._role_alive_ratio("red", "recon_aircraft"),
            "red_transport_alive_ratio": self._role_alive_ratio("red", "transport"),
            "red_transport_with_army_ratio": self._transport_with_army_ratio(),
            "red_recon_hp_ratio": self._role_hp_ratio("red", "recon_aircraft"),
            "red_attack_hp_ratio": self._role_hp_ratio("red", "attack_aircraft"),
            "red_ground_hp_ratio": self._role_hp_ratio("red", "ground_force"),
            "known_blue_combat_hp_ratio": self._known_blue_hp_ratio(),
            "known_blue_air_hp_ratio": self._known_role_hp_ratio("attack_aircraft"),
            "known_blue_sam_hp_ratio": self._known_role_hp_ratio("sam"),
            "known_blue_radar_hp_ratio": self._known_role_hp_ratio("radar"),
            "known_blue_ground_hp_ratio": self._known_role_hp_ratio("ground_force"),
            "landing_blue_air_kill_progress": min(1.0, self._destroyed_count("blue", "attack_aircraft") / float(required_blue_air_kills)),
            "landing_blue_sam_suppression": 1.0 if alive_blue_sams <= max_alive_blue_sams else max(0.0, 1.0 - ((alive_blue_sams - max_alive_blue_sams) / float(max(1, blue_sam_total)))),
            "nearest_known_sam_to_landing_norm": self._nearest_known_target_to_points_norm(["SAM"], self.config.get("landing_zones", []), max_commander_distance),
            "nearest_red_ground_to_objective_norm": self._nearest_red_ground_to_points_norm(self.config.get("ground_objectives", []), max_commander_distance),
        }
        values.update(self._spatial_grid_values(grid_size=3, max_per_cell=5))
        values.update(self._last_command_state_values())
        for slot, zone in enumerate(self.config.get("landing_zones", [])[:3], start=1):
            prefix = "landing_zone_{0}_".format(slot)
            values[prefix + "exists"] = 1.0
            values[prefix + "nearest_transport_distance_norm"] = self._nearest_transport_to_point_norm(zone, max_commander_distance)
            values[prefix + "nearest_known_sam_distance_norm"] = self._nearest_known_target_to_point_norm(["SAM"], zone, max_commander_distance)
            values[prefix + "nearest_known_blue_ground_distance_norm"] = self._nearest_known_target_to_point_norm(["GROUND", "FORCE"], zone, max_commander_distance)
        for slot in range(len(self.config.get("landing_zones", [])) + 1, 4):
            prefix = "landing_zone_{0}_".format(slot)
            values[prefix + "exists"] = 0.0
            values[prefix + "nearest_transport_distance_norm"] = 1.0
            values[prefix + "nearest_known_sam_distance_norm"] = 1.0
            values[prefix + "nearest_known_blue_ground_distance_norm"] = 1.0
        return values


    def _agent_id_norm(self, platform):
        names = self._role_identity_names(platform.role)
        if not names or platform.name not in names:
            return 0.0
        return (names.index(platform.name) + 1) / float(len(names))

    def _group_slot_norm(self, platform, group):
        platforms = list(getattr(group, "platforms", []) or [])
        names = [p.name for p in platforms]
        if not names or platform.name not in names:
            return 0.0
        return (names.index(platform.name) + 1) / float(len(names))

    def _recon_team_id_norm(self, group):
        teams = list(getattr(self.recon_controller, "fixed_recon_teams", []) or [])
        if not teams or group is None:
            return 0.0
        member_names = tuple(
            platform.name for platform in (getattr(group, "platforms", []) or [])
        )
        for index, team in enumerate(teams, start=1):
            if tuple(team) == member_names:
                return index / float(len(teams))
        return 0.0
    def _attack_team_id_norm(self, group):
        teams = list(getattr(self.attack_controller, "fixed_attack_teams", []) or [])
        if not teams or group is None:
            return 0.0
        fixed_team_id = str(getattr(group, "fixed_team_id", ""))
        prefix = "attack_fixed_team_"
        if fixed_team_id.startswith(prefix):
            try:
                team_index = int(fixed_team_id[len(prefix):])
            except ValueError:
                team_index = 0
            if 1 <= team_index <= len(teams):
                return team_index / float(len(teams))
        member_names = tuple(
            platform.name for platform in (getattr(group, "platforms", []) or [])
        )
        for index, team in enumerate(teams, start=1):
            if tuple(team) == member_names:
                return index / float(len(teams))
        return 0.0

    def _role_identity_names(self, role):
        red = self.config.get("red", {})
        if role == "recon_aircraft":
            return list(red.get("commandable_recon_aircraft", red.get("recon_aircraft", [])))
        if role == "attack_aircraft":
            return list(red.get("commandable_attack_aircraft", red.get("attack_aircraft", [])))
        if role == "transport":
            return list(red.get("commandable_transports", red.get("transports", [])))
        if role == "ground_force":
            return list(red.get("commandable_ground_forces", red.get("ground_forces", [])))
        return [p.name for p in self.platforms.values() if p.side == "red" and p.role == role]

    def _role_total(self, side, role):
        return sum(1 for p in self.platforms.values() if p.side == side and p.role == role)

    def _alive_count(self, side, role):
        return sum(
            1 for p in self.platforms.values()
            if p.side == side and p.role == role and p.alive
        )

    def _destroyed_count(self, side, role):
        return sum(
            1 for p in self.platforms.values()
            if p.side == side and p.role == role and not p.alive
        )

    def _role_alive_ratio(self, side, role):
        total = self._role_total(side, role)
        if total <= 0:
            return 0.0
        return self._alive_count(side, role) / float(total)

    def _role_hp_ratio(self, side, role):
        units = [p for p in self.platforms.values() if p.side == side and p.role == role and p.max_hp > 0.0]
        maximum = sum(p.max_hp for p in units)
        if maximum <= 0.0:
            return 0.0
        return max(0.0, min(1.0, sum(max(0.0, p.current_hp) for p in units) / maximum))

    def _known_blue_hp_ratio(self):
        units = [p for p in self.platforms.values() if p.side == "blue" and p.max_hp > 0.0 and p.name in self.detected_targets]
        maximum = sum(p.max_hp for p in units)
        if maximum <= 0.0:
            return 0.0
        return max(0.0, min(1.0, sum(max(0.0, p.current_hp) for p in units) / maximum))

    def _known_role_hp_ratio(self, role):
        units = [p for p in self.platforms.values() if p.side == "blue" and p.role == role and p.max_hp > 0.0 and p.name in self.detected_targets]
        maximum = sum(p.max_hp for p in units)
        if maximum <= 0.0:
            return 0.0
        return max(0.0, min(1.0, sum(max(0.0, p.current_hp) for p in units) / maximum))

    def _destroyed_ratio(self, side, role):
        total = self._role_total(side, role)
        if total <= 0:
            return 0.0
        return self._destroyed_count(side, role) / float(total)

    def _available_ratio(self, side, role):
        total = self._role_total(side, role)
        if total <= 0:
            return 0.0
        count = sum(
            1 for p in self.platforms.values()
            if p.side == side and p.role == role and p.alive and p.platform_id is not None and not self._is_busy(p)
        )
        return count / float(total)

    def _ground_available_ratio(self):
        total = self._role_total("red", "ground_force")
        if total <= 0:
            return 0.0
        count = sum(
            1 for p in self.platforms.values()
            if p.side == "red" and p.role == "ground_force" and p.alive and p.platform_id is not None
            and self._is_landed_ground_available(p) and not self._is_busy(p)
        )
        return count / float(total)

    def _active_group_count_norm(self, controller, max_count):
        groups = getattr(controller, "active_groups", {})
        active = 0
        for group in groups.values():
            platforms = getattr(group, "platforms", [])
            if any(platform.alive and self._is_busy(platform) for platform in platforms):
                active += 1
        return min(1.0, active / float(max(1, max_count)))

    def _commander_recon_coverage_ratio(self):
        return 0.0

    def _last_recon_gain_norm(self):
        count = sum(1 for event in self.last_reward_events if event.get("type") == "new_detection")
        return min(1.0, count / float(max(1, len(self._blue_target_names()))))

    def _known_target_count(self, type_markers):
        markers = [marker.upper() for marker in type_markers]
        count = 0
        for det in self.detected_targets.values():
            target_type = str(det.get("Type", det.get("type", ""))).upper()
            if any(marker in target_type for marker in markers):
                count += 1
        return count

    def _landed_ground_ratio(self):
        total = max(1, len(self.config.get("red", {}).get("ground_forces", [])))
        landed = 0
        for status in self.ground_status.values():
            if status.get("landed", False) and not status.get("on_ship", False):
                landed += 1
        return min(1.0, landed / float(total))

    def _transport_with_army_ratio(self):
        total = max(1, len(self.config.get("red", {}).get("transports", [])))
        count = 0
        for cargo in self.landing_cargo.values():
            if cargo.get("has_army", True):
                count += 1
        return min(1.0, count / float(total))

    def _task_in_progress(self, task_prefix):
        prefix = str(task_prefix).upper()
        for p in self.platforms.values():
            if p.task and str(p.task).upper().startswith(prefix) and self._is_busy(p):
                return True
        return False

    def _any_capture_condition(self):
        for obj in self.config.get("ground_objectives", []):
            lat = float(obj.get("lat", 0.0))
            lon = float(obj.get("lon", 0.0))
            radius = float(obj.get("radius_m", 5000.0))
            blue_count, red_count = self._objective_presence(lat, lon, radius)
            if red_count > 0 and blue_count == 0:
                return True
        return False

    def _nearest_known_target_to_points_norm(self, type_markers, points, max_distance):
        markers = [marker.upper() for marker in type_markers]
        best = float("inf")
        for det in self.detected_targets.values():
            target_type = str(det.get("Type", det.get("type", ""))).upper()
            if not any(marker in target_type for marker in markers):
                continue
            lat = det.get("Lat", det.get("lat"))
            lon = det.get("Lon", det.get("lon"))
            if lat is None or lon is None:
                continue
            for point in points:
                p_lat = point.get("lat", point.get("center_lat"))
                p_lon = point.get("lon", point.get("center_lon"))
                if p_lat is None or p_lon is None:
                    continue
                distance, _ = self._distance_and_bearing(float(lat), float(lon), float(p_lat), float(p_lon))
                best = min(best, distance)
        return self._norm_distance(best, max_distance)

    def _nearest_known_target_to_point_norm(self, type_markers, point, max_distance):
        markers = [marker.upper() for marker in type_markers]
        p_lat = point.get("lat", point.get("center_lat"))
        p_lon = point.get("lon", point.get("center_lon"))
        if p_lat is None or p_lon is None:
            return 1.0
        best = float("inf")
        for det in self.detected_targets.values():
            target_type = str(det.get("Type", det.get("type", ""))).upper()
            if not any(marker in target_type for marker in markers):
                continue
            lat = det.get("Lat", det.get("lat"))
            lon = det.get("Lon", det.get("lon"))
            if lat is None or lon is None:
                continue
            distance, _ = self._distance_and_bearing(float(lat), float(lon), float(p_lat), float(p_lon))
            best = min(best, distance)
        return self._norm_distance(best, max_distance)

    def _nearest_transport_to_point_norm(self, point, max_distance):
        p_lat = point.get("lat", point.get("center_lat"))
        p_lon = point.get("lon", point.get("center_lon"))
        if p_lat is None or p_lon is None:
            return 1.0
        best = float("inf")
        for platform in self.platforms.values():
            if platform.side != "red" or platform.role != "transport" or not platform.alive:
                continue
            if platform.platform_id is None or self._is_busy(platform):
                continue
            distance, _ = self._distance_and_bearing(platform.lat, platform.lon, float(p_lat), float(p_lon))
            best = min(best, distance)
        return self._norm_distance(best, max_distance)
    def _spatial_grid_values(self, grid_size=3, max_per_cell=5):
        lo_lat = float(self.bounds.get("lat_min", 23.5))
        hi_lat = float(self.bounds.get("lat_max", 25.8))
        lo_lon = float(self.bounds.get("lon_min", 118.8))
        hi_lon = float(self.bounds.get("lon_max", 122.2))
        lat_span = max(1e-6, hi_lat - lo_lat)
        lon_span = max(1e-6, hi_lon - lo_lon)

        red = [[0] * grid_size for _ in range(grid_size)]
        blue = [[0] * grid_size for _ in range(grid_size)]

        for p in self.platforms.values():
            if not p.alive or p.platform_id is None:
                continue
            r = min(grid_size - 1, max(0, int((p.lat - lo_lat) / lat_span * grid_size)))
            c = min(grid_size - 1, max(0, int((p.lon - lo_lon) / lon_span * grid_size)))
            if p.side == "red" and p.role in ("recon_aircraft", "attack_aircraft", "ground_force"):
                red[r][c] += 1
            elif p.side == "blue" and p.name in self.detected_targets:
                blue[r][c] += 1

        values = {}
        for r in range(grid_size):
            for c in range(grid_size):
                values["grid_{0}_{1}_red".format(r, c)] = min(1.0, red[r][c] / float(max_per_cell))
                values["grid_{0}_{1}_blue".format(r, c)] = min(1.0, blue[r][c] / float(max_per_cell))
        return values

    def _nearest_red_ground_to_points_norm(self, points, max_distance):
        best = float("inf")
        for platform in self.platforms.values():
            if platform.side != "red" or platform.role != "ground_force" or not platform.alive:
                continue
            status = self.ground_status.get(platform.name, {"on_ship": False, "landed": True})
            if status.get("on_ship", False) or not status.get("landed", False):
                continue
            for point in points:
                p_lat = point.get("lat", point.get("center_lat"))
                p_lon = point.get("lon", point.get("center_lon"))
                if p_lat is None or p_lon is None:
                    continue
                distance, _ = self._distance_and_bearing(platform.lat, platform.lon, float(p_lat), float(p_lon))
                best = min(best, distance)
        return self._norm_distance(best, max_distance)

    def _role_code(self, role):
        roles = ["carrier", "recon_aircraft", "attack_aircraft", "transport", "ground_force", "base", "radar", "sam"]
        if role not in roles:
            return 0.0
        return (roles.index(role) + 1) / float(len(roles))

    def _norm_lat(self, lat):
        lo = float(self.bounds.get("lat_min", 23.0))
        hi = float(self.bounds.get("lat_max", 26.0))
        return min(1.0, max(0.0, (lat - lo) / max(1e-6, hi - lo)))

    def _norm_lon(self, lon):
        lo = float(self.bounds.get("lon_min", 119.0))
        hi = float(self.bounds.get("lon_max", 123.0))
        return min(1.0, max(0.0, (lon - lo) / max(1e-6, hi - lo)))

    @staticmethod
    def _norm_alt(alt, max_alt=12000.0):
        return min(1.0, max(0.0, float(alt) / max(1.0, float(max_alt))))

    def _alive_ratio(self, side):
        plats = [p for p in self.platforms.values() if p.side == side]
        if not plats:
            return 0.0
        return sum(1 for p in plats if p.alive) / float(len(plats))

    def get_agent_observation_fields(self, agent_type):
        return list(self.agent_observation_fields.get(agent_type, []))

    def get_agent_observation(self, agent_type):
        blackboard = self.get_blackboard()
        fields = self.get_agent_observation_fields(agent_type)
        return {field: self._resolve_field(blackboard, field) for field in fields}

    def get_blackboard(self):
        return {
            "time": {
                "step_count": self.step_count,
                "progress": self._mission_time_norm(),
            },
            "red": {
                "recon": self._unit_group("red", "recon_aircraft"),
                "attack": self._unit_group("red", "attack_aircraft"),
                "transport": self._unit_group("red", "transport"),
                "ground": self._unit_group("red", "ground_force"),
            },
            "own": {
                "recon_aircraft": self._unit_group("red", "recon_aircraft"),
                "attack_aircraft": self._unit_group("red", "attack_aircraft"),
                "transports": self._unit_group("red", "transport"),
                "ground_forces": self._unit_group("red", "ground_force"),
            },
            "known_blue": self._known_blue_summary(),
            "home": {
                "carrier": self._first_unit_position("red", "carrier"),
            },
            "objective": {
                "recon_areas": self.recon_areas,
                "landing_zones": self.config.get("landing_zones", []),
                "ground_objectives": self.config.get("ground_objectives", []),
                "capture_progress": 0.0,
            },
            "task": dict(self.task_flags),
        }

    def _unit_group(self, side, role):
        units = [self._platform_to_dict(p) for p in self.platforms.values() if p.side == side and p.role == role]
        available = [u for u in units if u["alive"] and not u["busy"] and u["platform_id"] is not None]
        return {
            "count": len(units),
            "available_count": len(available),
            "units": units,
        }

    def _platform_to_dict(self, platform: PlatformState):
        return {
            "name": platform.name,
            "role": platform.role,
            "side": platform.side,
            "platform_id": platform.platform_id,
            "type": platform.platform_type,
            "alive": platform.alive,
            "busy": self._is_busy(platform),
            "task": platform.task,
            "task_status": platform.task_status,
            "at_home": platform.at_home,
            "detected": platform.detected,
            "position": {
                "lat": platform.lat,
                "lon": platform.lon,
                "alt": platform.alt,
            },
        }

    def _first_unit_position(self, side, role):
        for platform in self.platforms.values():
            if platform.side == side and platform.role == role:
                return self._platform_to_dict(platform)["position"]
        return None

    def _known_blue_summary(self):
        groups = {
            "aircraft": [],
            "ground": [],
            "radar": [],
            "sam": [],
            "targets": [],
        }
        for name, det in self.detected_targets.items():
            target_type = det.get("Type", "")
            item = {
                "name": name,
                "type": target_type,
                "position": {
                    "lat": det.get("Lat"),
                    "lon": det.get("Lon"),
                    "alt": det.get("Alt"),
                },
                "range": det.get("Range"),
                "track_source": det.get("TrackSource"),
            }
            groups["targets"].append(item)
            upper_type = target_type.upper()
            if "SAM" in upper_type:
                groups["sam"].append(item)
            elif "RADAR" in upper_type:
                groups["radar"].append(item)
            elif "GROUND" in upper_type or "FORCE" in upper_type:
                groups["ground"].append(item)
            elif "AIR" in upper_type or "FIGHTER" in upper_type or "AIRCRAFT" in upper_type:
                groups["aircraft"].append(item)
        summary = {
            key: {
                "count": len(value),
                "units": value,
            }
            for key, value in groups.items()
        }
        summary["memory_count"] = len(self.enemy_track_memory)
        summary["memory_units"] = self._enemy_track_memory_records()
        return summary

    @staticmethod
    def _resolve_field(data, field):
        current = data
        for part in field.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current



    def get_ground_action_table(self):
        return [
            {
                "id": int(action.get("id", 0)),
                "name": action.get("name", ""),
                "description": action.get("description", ""),
                "afsim_task": action.get("afsim_task", ""),
            }
            for action in sorted(self.ground_action_config.get("actions", []), key=lambda item: int(item.get("id", 0)))
        ]

    def get_ground_state_fields(self):
        return list(self.ground_state_fields)

    def get_ground_task_state(self, group_id):
        self._update_red_ground_detections()
        group = self.ground_controller.active_groups.get(group_id)
        if not group:
            return None
        units = {}
        target_slots_by_unit = {}
        complete_count = 0
        for platform in group.platforms:
            values = self._build_ground_unit_obs(platform, group)
            target_slots = self._ground_target_slots(group, platform)
            target_slots_by_unit[platform.name] = target_slots
            if values.get("task_complete", 0.0) > 0.0:
                complete_count += 1
            units[platform.name] = {
                "fields": list(self.ground_state_fields),
                "obs": [values.get(field, 0.0) for field in self.ground_state_fields],
                "obs_by_name": {field: values.get(field, 0.0) for field in self.ground_state_fields},
                "action_mask": self._build_ground_action_mask(platform, group, values),
                "target_slots": target_slots,
            }
        return {
            "task": {
                "group_id": group.group_id,
                "objective": group.objective.get("name", ""),
                "objective_lat": group.objective.get("lat", 0.0),
                "objective_lon": group.objective.get("lon", 0.0),
                "objective_radius_m": group.objective.get("radius_m", 5000.0),
                "last_action_id": group.last_action_id,
                "assigned_units": [platform.name for platform in group.platforms],
                "complete_ratio": complete_count / float(max(1, len(group.platforms))),
                "target_slots_by_unit": target_slots_by_unit,
            },
            "units": units,
            "action_table": self.get_ground_action_table(),
        }

    def _build_ground_unit_obs(self, platform, group):
        norm_cfg = self.ground_state_config.get("normalization", {})
        max_distance = float(norm_cfg.get("max_distance_m", 200000.0))
        weapon_range = float(norm_cfg.get("ground_weapon_range_m", 5000.0))
        objective_radius = float(group.objective.get("radius_m", norm_cfg.get("objective_radius_m", 5000.0)))
        friendly_radius = float(norm_cfg.get("friendly_near_radius_m", 3000.0))
        max_elapsed_steps = float(norm_cfg.get("max_elapsed_steps", 360.0))
        obj_lat = float(group.objective.get("lat", platform.lat))
        obj_lon = float(group.objective.get("lon", platform.lon))
        dist_obj, bearing_obj = self._distance_and_bearing(platform.lat, platform.lon, obj_lat, obj_lon)
        status = self.ground_status.get(platform.name, {"on_ship": False, "landed": True})
        blue_ground = self._nearest_ground_fire_target(platform) or {"distance_m": float("inf"), "bearing_deg": 0.0, "name": "", "alive": False, "lat": None, "lon": None, "type": ""}
        blue_air = self._nearest_detected_target(platform, ["AIR", "AIRCRAFT", "FIGHTER"])
        sam = self._nearest_detected_target(platform, ["SAM"])
        near_friend = 0
        for friend in self.platforms.values():
            if friend.name == platform.name or friend.side != "red" or friend.role != "ground_force" or not friend.alive:
                continue
            friend_status = self.ground_status.get(friend.name, {"on_ship": False, "landed": True})
            if friend_status.get("landed", False) and not friend_status.get("on_ship", False):
                d, _ = self._distance_and_bearing_to_platform(platform, friend)
                if d <= friendly_radius:
                    near_friend += 1
        blue_in_obj, red_in_obj = self._objective_presence(obj_lat, obj_lon, objective_radius)
        elapsed_steps = max(0, self.step_count - int(getattr(group, "start_step", self.step_count)))
        ammo = self.ground_ammo.get(platform.name, {"ground_fire": 15}).get("ground_fire", 15)
        max_speed = float(norm_cfg.get("max_speed_mps", 10.0))
        last_action_den = max(1.0, float(max(self.ground_controller.action_specs.keys() or [1])))
        values = {
            "alive": 1.0 if platform.alive else 0.0,
            "hp_norm": self._hp_norm(platform),
            "damaged": 1.0 if 0.0 < platform.current_hp < platform.max_hp else 0.0,
            "fire_cooldown_remaining_norm": self._combat_remaining_norm(platform.fire_cooldown_until, 300.0),
            "combat_lock_remaining_norm": self._combat_remaining_norm(platform.combat_lock_until, 300.0),
            "agent_id_norm": self._agent_id_norm(platform),
            "group_slot_norm": self._group_slot_norm(platform, group),
            "lat_norm": self._norm_lat(platform.lat),
            "lon_norm": self._norm_lon(platform.lon),
            "is_busy": 1.0 if self._is_busy(platform) else 0.0,
            "task_assigned": 1.0,
            "is_stationary": 1.0 if float(platform.speed) <= 1.0 else 0.0,
            "speed_norm": min(1.0, max(0.0, float(platform.speed) / max(1.0, max_speed))),
            "heading_sin": self._bearing_sin(getattr(platform, "heading", 0.0)),
            "heading_cos": self._bearing_cos(getattr(platform, "heading", 0.0)),
            "on_ship": 1.0 if status.get("on_ship", False) else 0.0,
            "landed": 1.0 if status.get("landed", False) else 0.0,
            "distance_to_objective_norm": self._norm_distance(dist_obj, max_distance),
            "bearing_to_objective_norm": float(bearing_obj) / 360.0,
            "inside_objective": 1.0 if dist_obj <= objective_radius else 0.0,
            "nearest_blue_ground_distance_norm": self._norm_distance(blue_ground["distance_m"], max_distance),
            "nearest_blue_ground_bearing_norm": float(blue_ground["bearing_deg"]) / 360.0,
            "nearest_blue_ground_alive": 1.0 if blue_ground.get("alive", False) else 0.0,
            "nearest_blue_ground_hp_norm": float(blue_ground.get("hp_norm", 0.0)),
            "inside_ground_weapon_range": 1.0 if blue_ground["distance_m"] <= weapon_range else 0.0,
            "ground_ammo_norm": min(1.0, max(0.0, float(ammo) / 15.0)),
            "nearest_blue_air_distance_norm": self._norm_distance(blue_air["distance_m"], max_distance),
            "nearest_blue_sam_distance_norm": self._norm_distance(sam["distance_m"], max_distance),
            "near_friendly_ground_count_norm": min(1.0, near_friend / 5.0),
            "known_blue_ground_count_norm": self._known_target_count_norm(["GROUND", "FORCE"], 20.0),
            "known_blue_radar_count_norm": self._known_target_count_norm(["RADAR"], 3.0),
            "known_blue_sam_count_norm": self._known_target_count_norm(["SAM"], 5.0),
            "objective_blue_presence_norm": min(1.0, blue_in_obj / 20.0),
            "objective_red_presence_norm": min(1.0, red_in_obj / 10.0),
            "capture_progress_norm": 1.0 if red_in_obj > 0 and blue_in_obj == 0 else 0.0,
            "task_complete": 1.0 if dist_obj <= objective_radius and blue_in_obj == 0 else 0.0,
            "last_action_id_norm": min(1.0, max(0.0, float(getattr(group, "last_action_id", 0)) / last_action_den)),
        }
        target_slots = self._ground_target_slots(group, platform)
        for slot in range(self._ground_target_slot_count()):
            prefix = "target_slot_{0}_".format(slot + 1)
            target = target_slots[slot] if slot < len(target_slots) else None
            role = str(target.get("role", "")) if target else ""
            distance = float(target.get("distance_m", float("inf"))) if target else float("inf")
            bearing = float(target.get("bearing_deg", 0.0)) if target else 0.0
            values.update({
                prefix + "exists": 1.0 if target else 0.0,
                prefix + "known": 1.0 if target and target.get("known", False) else 0.0,
                prefix + "distance_norm": self._norm_distance(distance, max_distance),
                prefix + "bearing_sin": self._bearing_sin(bearing),
                prefix + "bearing_cos": self._bearing_cos(bearing),
                prefix + "hp_norm": float(target.get("hp_norm", 0.0)) if target else 0.0,
                prefix + "is_ground_force": 1.0 if role == "ground_force" else 0.0,
                prefix + "is_radar": 1.0 if role == "radar" else 0.0,
                prefix + "is_sam": 1.0 if role == "sam" else 0.0,
                prefix + "in_weapon_range": 1.0 if target and distance <= weapon_range else 0.0,
            })
        return values

    def _build_ground_action_mask(self, platform, group, values):
        mask = []
        alive = values.get("alive", 0.0) > 0.0
        landed = values.get("landed", 0.0) > 0.0 and values.get("on_ship", 1.0) <= 0.0
        has_ammo = values.get("ground_ammo_norm", 0.0) > 0.0
        fire_ready = values.get("fire_cooldown_remaining_norm", 0.0) <= 0.0
        movement_locked = values.get("combat_lock_remaining_norm", 0.0) > 0.0
        weapon_range = float(self.ground_state_config.get("normalization", {}).get("ground_weapon_range_m", 5000.0))
        target_slots = self._ground_target_slots(group, platform)
        inside_objective = values.get("inside_objective", 0.0) > 0.0
        blue_present = values.get("objective_blue_presence_norm", 0.0) > 0.0
        for action_id, spec in sorted(self.ground_controller.action_specs.items()):
            name = spec.get("name", "")
            afsim_task = spec.get("afsim_task", "")
            available = 0.0
            if not alive:
                available = 1.0 if name == "HOLD" else 0.0
            elif name == "HOLD":
                available = 1.0
            elif not landed:
                available = 0.0
            elif afsim_task == "GROUND_TARGET_SLOT":
                slot = int(spec.get("target_slot", -1))
                target = target_slots[slot] if 0 <= slot < len(target_slots) else None
                reserved_by = self.attack_target_reservations.get(target.get("name", "")) if target else None
                target_available = bool(
                    target
                    and target.get("known", False)
                    and target.get("alive", False)
                    and (not reserved_by or reserved_by == platform.name)
                )
                available = 1.0 if has_ammo and fire_ready and target_available else 0.0
            elif afsim_task == "GROUND_CAPTURE":
                available = 1.0 if not movement_locked and inside_objective and not blue_present else 0.0
            elif afsim_task == "GROUND_MOVE_POINT":
                available = 0.0 if movement_locked else 1.0
            mask.append(available)
        return mask

    def _clear_expired_ground_target_reservations(self):
        now = float(self._current_sim_time())
        for target_name, reservation in list(self.ground_target_reservations.items()):
            target = self.platforms.get(target_name)
            expired = now >= float(reservation.get("expires_at", 0.0))
            target_gone = target is None or not target.alive
            if not expired and not target_gone:
                continue
            if self.attack_target_reservations.get(target_name) == reservation.get("owner"):
                self.attack_target_reservations.pop(target_name, None)
            self.ground_target_reservations.pop(target_name, None)

    def _nearest_ground_fire_target(self, platform):
        if platform is None:
            return None
        self._clear_expired_ground_target_reservations()
        self._update_red_ground_detections()
        best = None
        best_distance = float("inf")
        for name, det in self.ground_detected_targets.items():
            candidate = self.platforms.get(name)
            if candidate is None or candidate.side != "blue" or candidate.role not in ["ground_force", "radar", "sam"] or not candidate.alive:
                continue
            if self.attack_target_reservations.get(name):
                continue
            distance, bearing = self._distance_and_bearing_to_platform(platform, candidate)
            if distance < best_distance:
                best_distance = distance
                best = {
                    "name": candidate.name,
                    "lat": candidate.lat,
                    "lon": candidate.lon,
                    "alt": candidate.alt,
                    "type": candidate.platform_type or candidate.role,
                    "distance_m": distance,
                    "bearing_deg": bearing,
                    "alive": candidate.alive,
                    "hp_norm": self._hp_norm(candidate),
                    "track_source": det.get("TrackSource", ""),
                }
        return best

    def _ground_target_slot_count(self):
        return len(self.ground_fixed_target_names)

    def _ground_target_slots(self, group, platform):
        """Return permanent enemy slots using the latest shared reconnaissance track.

        Navigation coordinates are deliberately read from enemy_track_memory, never
        from live blue-platform state.  A current local ground observation refreshes
        that shared memory, so local and recon reports use the same track pipeline.
        """
        if group is None or platform is None:
            return []
        self._clear_expired_ground_target_reservations()
        self._update_red_ground_detections()
        slots = []
        for slot, name in enumerate(self.ground_fixed_target_names):
            candidate = self.platforms.get(name)
            memory = self.enemy_track_memory.get(name)
            compatible = bool(candidate and candidate.side == "blue" and candidate.role in ("ground_force", "radar", "sam"))
            known = bool(compatible and memory)
            lat = float(memory.get("Lat", 0.0)) if known else 0.0
            lon = float(memory.get("Lon", 0.0)) if known else 0.0
            alt = float(memory.get("Alt", 0.0)) if known else 0.0
            distance, bearing = (self._distance_and_bearing(platform.lat, platform.lon, lat, lon) if known else (float("inf"), 0.0))
            slots.append({"slot": slot, "name": name, "lat": lat, "lon": lon, "alt": alt,
                          "type": (candidate.platform_type or candidate.role) if candidate else "",
                          "role": candidate.role if candidate else "", "distance_m": distance,
                          "bearing_deg": bearing, "known": known,
                          "alive": bool(candidate and candidate.alive),
                          "hp_norm": self._hp_norm(candidate) if known else 0.0,
                          "track_source": memory.get("TrackSource", "") if memory else "",
                          "last_seen": memory.get("LastSeen") if memory else None})
        return slots

    def _objective_presence(self, lat, lon, radius_m):
        blue_count = 0
        red_count = 0
        for platform in self.platforms.values():
            if platform.role != "ground_force" or not platform.alive:
                continue
            if platform.side == "red":
                status = self.ground_status.get(platform.name, {"on_ship": False, "landed": True})
                if status.get("on_ship", False) or not status.get("landed", False):
                    continue
            d, _ = self._distance_and_bearing(platform.lat, platform.lon, lat, lon)
            if d <= radius_m:
                if platform.side == "blue":
                    blue_count += 1
                elif platform.side == "red":
                    red_count += 1
        return blue_count, red_count

    def get_landing_action_table(self):
        return [
            {
                "id": index,
                "name": name,
                "description": "Continuous transport movement component in [-1, 1].",
                "action_type": "continuous",
            }
            for index, name in enumerate((
                "move_east_norm", "move_north_norm", "surface_vertical_reserved"
            ))
        ]
        return [
            {
                "id": int(action.get("id", 0)),
                "name": action.get("name", ""),
                "description": action.get("description", ""),
                "afsim_task": action.get("afsim_task", ""),
            }
            for action in sorted(self.landing_action_config.get("actions", []), key=lambda item: int(item.get("id", 0)))
        ]

    def get_landing_state_fields(self):
        return list(self.landing_state_fields)

    def get_landing_task_state(self, group_id):
        group = self.landing_controller.active_groups.get(group_id)
        if not group:
            return None
        ships = {}
        complete_count = 0
        for platform in group.platforms:
            values = self._build_landing_ship_obs(platform, group)
            if values.get("landing_complete", 0.0) > 0.0:
                complete_count += 1
            ships[platform.name] = {
                "fields": list(self.landing_state_fields),
                "obs": [values.get(field, 0.0) for field in self.landing_state_fields],
                "obs_by_name": {field: values.get(field, 0.0) for field in self.landing_state_fields},
                "action_mask": self._build_landing_action_mask(platform, group, values),
            }
        return {
            "task": {
                "group_id": group.group_id,
                "landing_zone": group.landing_zone.get("name", ""),
                "landing_lat": group.landing_zone.get("lat", 0.0),
                "landing_lon": group.landing_zone.get("lon", 0.0),
                "navigation_lat": group.landing_zone.get("navigation_lat", group.landing_zone.get("berth_lat", group.landing_zone.get("lat", 0.0))),
                "navigation_lon": group.landing_zone.get("navigation_lon", group.landing_zone.get("berth_lon", group.landing_zone.get("lon", 0.0))),
                "berth_lat": group.landing_zone.get("berth_lat", group.landing_zone.get("lat", 0.0)),
                "berth_lon": group.landing_zone.get("berth_lon", group.landing_zone.get("lon", 0.0)),
                "arrival_tolerance_m": group.landing_zone.get("arrival_tolerance_m", 200.0),
                "unload_radius_m": group.landing_zone.get("unload_radius_m", 350.0),
                "landing_radius_m": group.landing_zone.get("unload_radius_m", 350.0),
                "last_action_id": group.last_action_id,
                "assigned_ships": [platform.name for platform in group.platforms],
                "complete_ratio": complete_count / float(max(1, len(group.platforms))),
            },
            "ships": ships,
            "action_table": self.get_landing_action_table(),
        }

    def _build_landing_ship_obs(self, platform, group):
        norm_cfg = self.landing_state_config.get("normalization", {})
        max_distance = float(norm_cfg.get("max_distance_m", 300000.0))
        max_elapsed_steps = float(norm_cfg.get("max_elapsed_steps", 240.0))
        landing_zone = group.landing_zone
        landing_lat = float(landing_zone.get("lat", landing_zone.get("center_lat", platform.lat)))
        landing_lon = float(landing_zone.get("lon", landing_zone.get("center_lon", platform.lon)))
        berth_lat = float(landing_zone.get("navigation_lat", landing_zone.get("berth_lat", landing_lat)))
        berth_lon = float(landing_zone.get("navigation_lon", landing_zone.get("berth_lon", landing_lon)))
        arrival_tolerance = float(landing_zone.get("arrival_tolerance_m", 200.0))
        unload_radius = float(landing_zone.get("unload_radius_m", 350.0))
        dist_berth, bearing_berth = self._distance_and_bearing(platform.lat, platform.lon, berth_lat, berth_lon)
        dist_landing, _ = self._distance_and_bearing(platform.lat, platform.lon, landing_lat, landing_lon)
        cargo = self.landing_cargo.get(platform.name, {"has_army": True, "army_landed": False})
        unload = self.pending_landing_unloads.get(platform.name, {})
        unloading = str(unload.get("phase", "")) == "unloading"
        unload_duration = float(self.config.get("scenario", {}).get("landing_unload_duration_seconds", 900.0))
        unload_remaining = max(0.0, float(unload.get("unload_complete_at", 0.0)) - self._current_sim_time())
        transport_map = self.config.get("red", {}).get("ground_transport_map", {})
        manifest_total = sum(1 for transport in transport_map.values() if transport == platform.name)
        manifest_on_ship = sum(
            1 for name, status in self.ground_status.items()
            if status.get("transport") == platform.name and status.get("on_ship", False)
        )
        blue_ground = self._nearest_detected_target(platform, ["GROUND", "FORCE"])
        sam = self._nearest_detected_target(platform, ["SAM"])
        blue_air = self._nearest_detected_target(platform, ["AIR", "AIRCRAFT", "FIGHTER"])
        elapsed_steps = max(0, self.step_count - int(getattr(group, "start_step", self.step_count)))
        sea_path = self._landing_sea_path_metrics(platform, landing_zone)
        last_action_den = max(1.0, float(max(self.landing_controller.action_specs.keys() or [1])))
        values = {
            "alive": 1.0 if platform.alive else 0.0,
            "agent_id_norm": self._agent_id_norm(platform),
            "group_slot_norm": self._group_slot_norm(platform, group),
            "lat_norm": self._norm_lat(platform.lat),
            "lon_norm": self._norm_lon(platform.lon),
            "is_busy": 1.0 if self._is_busy(platform) else 0.0,
            "task_assigned": 1.0,
            "is_stationary": 1.0 if float(platform.speed) <= 1.0 else 0.0,
            "speed_norm": min(1.0, max(0.0, float(platform.speed) / max(1.0, self.landing_controller.max_speed_mps))),
            "command_speed_norm": min(1.0, max(0.0, float(group.command_speed_mps) / max(1.0, self.landing_controller.max_speed_mps))),
            "heading_sin": self._bearing_sin(getattr(platform, "heading", 0.0)),
            "heading_cos": self._bearing_cos(getattr(platform, "heading", 0.0)),
            "last_action_id_norm": min(1.0, max(0.0, float(getattr(group, "last_action_id", 0)) / last_action_den)),
            "has_army": 1.0 if cargo.get("has_army", True) else 0.0,
            "army_landed": 1.0 if cargo.get("army_landed", False) else 0.0,
            "unloading": 1.0 if unloading else 0.0,
            "unload_remaining_norm": min(1.0, unload_remaining / max(1.0, unload_duration)),
            "cargo_unit_count_norm": min(1.0, manifest_on_ship / float(max(1, manifest_total))),
            "distance_to_landing_zone_norm": self._norm_distance(dist_berth, max_distance),
            "bearing_to_landing_zone_norm": float(bearing_berth) / 360.0,
            "sea_path_distance_norm": self._norm_distance(sea_path["distance_m"], max_distance) if sea_path["reachable"] else 1.0,
            "sea_path_reachable": 1.0 if sea_path["reachable"] else 0.0,
            "at_landing_zone": 1.0 if dist_berth <= arrival_tolerance and dist_landing <= unload_radius else 0.0,
            "nearest_blue_ground_distance_norm": self._norm_distance(blue_ground["distance_m"], max_distance),
            "nearest_blue_ground_bearing_norm": float(blue_ground["bearing_deg"]) / 360.0,
            "nearest_blue_sam_distance_norm": self._norm_distance(sam["distance_m"], max_distance),
            "nearest_blue_sam_bearing_norm": float(sam["bearing_deg"]) / 360.0,
            "nearest_blue_air_distance_norm": self._norm_distance(blue_air["distance_m"], max_distance),
            "nearest_blue_air_bearing_norm": float(blue_air["bearing_deg"]) / 360.0,
            "known_blue_ground_count_norm": self._known_target_count_norm(["GROUND", "FORCE"], 20.0),
            "known_blue_sam_count_norm": self._known_target_count_norm(["SAM"], 5.0),
            "available_attack_aircraft_count_norm": self._available_count_norm("red", "attack_aircraft", 10.0),
            "landing_complete": 1.0 if cargo.get("army_landed", False) else 0.0,
        }
        for slot, zone in enumerate(self.config.get("landing_zones", [])[:3], start=1):
            prefix = "landing_zone_slot_{0}_".format(slot)
            values.update(self._landing_zone_slot_obs(platform, group, zone, prefix, max_distance))
        for slot in range(len(self.config.get("landing_zones", [])) + 1, 4):
            values.update(self._empty_landing_zone_slot_obs("landing_zone_slot_{0}_".format(slot)))
        return values

    def _landing_zone_slot_obs(self, platform, group, zone, prefix, max_distance):
        z_lat = float(zone.get("lat", zone.get("center_lat", platform.lat)))
        z_lon = float(zone.get("lon", zone.get("center_lon", platform.lon)))
        distance, bearing = self._distance_and_bearing(platform.lat, platform.lon, z_lat, z_lon)
        selected = zone.get("name", "") == group.landing_zone.get("name", "")
        return {
            prefix + "exists": 1.0,
            prefix + "selected": 1.0 if selected else 0.0,
            prefix + "distance_norm": self._norm_distance(distance, max_distance),
            prefix + "bearing_norm": float(bearing) / 360.0,
            prefix + "nearest_known_sam_distance_norm": self._nearest_known_target_to_point_norm(["SAM"], zone, max_distance),
            prefix + "nearest_known_blue_ground_distance_norm": self._nearest_known_target_to_point_norm(["GROUND", "FORCE"], zone, max_distance),
        }

    @staticmethod
    def _empty_landing_zone_slot_obs(prefix):
        return {
            prefix + "exists": 0.0,
            prefix + "selected": 0.0,
            prefix + "distance_norm": 1.0,
            prefix + "bearing_norm": 0.0,
            prefix + "nearest_known_sam_distance_norm": 1.0,
            prefix + "nearest_known_blue_ground_distance_norm": 1.0,
        }

    def _build_landing_action_mask(self, platform, group, values):
        available = (
            values.get("alive", 0.0) > 0.0
            and values.get("has_army", 0.0) > 0.0
            and values.get("landing_complete", 0.0) <= 0.0
            and values.get("unloading", 0.0) <= 0.0
            and self._is_landing_window_open()
        )
        return [1.0 if available else 0.0] * self.landing_controller.continuous_action_dim

        mask = []
        alive = values.get("alive", 0.0) > 0.0
        has_army = values.get("has_army", 0.0) > 0.0
        at_lz = values.get("at_landing_zone", 0.0) > 0.0
        stationary = values.get("is_stationary", 0.0) > 0.0
        complete = values.get("landing_complete", 0.0) > 0.0
        unloading = values.get("unloading", 0.0) > 0.0
        landing_window_open = self._is_landing_window_open()
        for action_id, spec in sorted(self.landing_controller.action_specs.items()):
            name = spec.get("name", "")
            afsim_task = spec.get("afsim_task", "")
            available = 0.0
            if not alive:
                available = 1.0 if name == "HOLD" else 0.0
            elif unloading:
                available = 1.0 if name == "HOLD" else 0.0
            elif name == "HOLD":
                available = 1.0
            elif not landing_window_open:
                available = 0.0
            elif afsim_task == "LANDING_UNLOAD":
                available = 1.0 if has_army and at_lz and stationary and not complete else 0.0
            elif afsim_task == "LANDING_RETURN_WAIT":
                # Legacy controller support only. RETURN_WAIT_AREA is no longer exposed
                # to the learned bottom policy because it bypasses path learning.
                available = 0.0
            elif afsim_task == "LANDING_MOVE_POINT":
                sea_segment_ok = self._landing_move_segment_allowed(platform, group, spec)
                if name == "SPEED_UP":
                    speed_ok = group.command_speed_mps < self.landing_controller.max_speed_mps - 0.1
                elif name == "SPEED_DOWN":
                    speed_ok = group.command_speed_mps > self.landing_controller.min_speed_mps + 0.1
                else:
                    speed_ok = True
                available = 1.0 if has_army and not complete and sea_segment_ok and speed_ok else 0.0
            mask.append(available)
        return mask

    def _landing_navigation_bounds(self):
        scenario_cfg = self.config.get("scenario", {})
        return scenario_cfg.get("landing_navigation_bounds", self.bounds)

    def _landing_point_in_navigation_bounds(self, point):
        bounds = self._landing_navigation_bounds()
        lat, lon = float(point[0]), float(point[1])
        return (
            float(bounds.get("lat_min", -90.0)) <= lat <= float(bounds.get("lat_max", 90.0))
            and float(bounds.get("lon_min", -180.0)) <= lon <= float(bounds.get("lon_max", 180.0))
        )

    def _landing_distance_field(self, landing_zone):
        berth = (
            float(landing_zone.get("navigation_lat", landing_zone.get("berth_lat", landing_zone.get("lat", 0.0)))),
            float(landing_zone.get("navigation_lon", landing_zone.get("berth_lon", landing_zone.get("lon", 0.0)))),
        )
        key = (round(berth[0], 7), round(berth[1], 7))
        field = self._landing_distance_fields.get(key)
        if field is None:
            polygons = [
                self._polygon_points(item)
                for item in self.config.get("scenario", {}).get("landing_land_polygons", [])
            ]
            bounds = self._landing_navigation_bounds()
            field = SeaGridDistanceField(
                bounds,
                polygons,
                berth,
                cell_m=float(bounds.get("grid_cell_m", 2000.0)),
            )
            self._landing_distance_fields[key] = field
        return field

    def _landing_sea_path_metrics(self, platform, landing_zone):
        if not landing_zone or "navigation_lat" not in landing_zone:
            return {"distance_m": 0.0, "reachable": True}
        return self._landing_distance_field(landing_zone).metrics(platform.lat, platform.lon)

    def _landing_move_segment_allowed(self, platform, group, action_spec):
        target = self._landing_move_target(platform, group, action_spec)
        if target is None:
            return False
        start = (float(platform.lat), float(platform.lon))
        end = (float(target[0]), float(target[1]))
        if not self._landing_point_in_navigation_bounds(end):
            return False
        for polygon in self.config.get("scenario", {}).get("landing_land_polygons", []):
            points = self._polygon_points(polygon)
            if len(points) < 3:
                continue
            if self._point_in_polygon(end, points):
                return False
            if self._segment_intersects_polygon(start, end, points):
                return False
        return True

    def _landing_move_target(self, platform, group, action_spec):
        if action_spec.get("to_landing_zone"):
            zone = group.landing_zone
            target_lat = float(zone.get("navigation_lat", zone.get("berth_lat", zone.get("lat", zone.get("center_lat", platform.lat)))))
            target_lon = float(zone.get("navigation_lon", zone.get("berth_lon", zone.get("lon", zone.get("center_lon", platform.lon)))))
            north_m, east_m = self._relative_north_east(platform.lat, platform.lon, target_lat, target_lon)
            distance_m = math.hypot(north_m, east_m)
            step_m = max(1000.0, float(action_spec.get("move_step_m", 30000.0)))
            if distance_m <= step_m or distance_m <= 1.0:
                return target_lat, target_lon
            scale = step_m / distance_m
            return self._offset_lat_lon(platform.lat, platform.lon, north_m * scale, east_m * scale)
        if action_spec.get("afsim_task") == "LANDING_RETURN_WAIT":
            zone = group.landing_zone
            return (float(zone.get("wait_lat", platform.lat)), float(zone.get("wait_lon", platform.lon)))
        if action_spec.get("afsim_task") != "LANDING_MOVE_POINT":
            return None
        north_m = float(action_spec.get("north_m", 0.0))
        east_m = float(action_spec.get("east_m", 0.0))
        if "speed_delta_mps" in action_spec and abs(north_m) + abs(east_m) <= 1.0:
            heading_rad = math.radians(float(getattr(platform, "heading", 0.0)))
            forward_m = float(action_spec.get("forward_m", 2000.0))
            north_m = math.cos(heading_rad) * forward_m
            east_m = math.sin(heading_rad) * forward_m
        zone = group.landing_zone
        berth_lat = float(zone.get("navigation_lat", zone.get("berth_lat", zone.get("lat", platform.lat))))
        berth_lon = float(zone.get("navigation_lon", zone.get("berth_lon", zone.get("lon", platform.lon))))
        distance_to_berth, _ = self._distance_and_bearing(platform.lat, platform.lon, berth_lat, berth_lon)
        if distance_to_berth <= float(zone.get("near_maneuver_distance_m", 2000.0)):
            action_step = math.hypot(north_m, east_m)
            near_step = float(zone.get("near_step_m", 250.0))
            if action_step > near_step > 0.0:
                scale = near_step / action_step
                north_m *= scale
                east_m *= scale
        return self._offset_lat_lon(platform.lat, platform.lon, north_m, east_m)

    @staticmethod
    def _polygon_points(polygon):
        raw_points = polygon.get("points", polygon) if isinstance(polygon, dict) else polygon
        points = []
        for point in raw_points or []:
            if isinstance(point, dict):
                points.append((float(point.get("lat", 0.0)), float(point.get("lon", 0.0))))
            else:
                points.append((float(point[0]), float(point[1])))
        return points

    def _annotated_island_polygons(self):
        """Return named land polygons designated as explorable islands.

        The scenario may explicitly list ``island_annotations`` with a
        ``land_polygon`` name.  Until that list is supplied, every configured
        landing land polygon remains an island annotation for compatibility.
        """
        land_polygons = {
            str(item.get("name", "")): self._polygon_points(item)
            for item in self.config.get("scenario", {}).get("landing_land_polygons", [])
            if isinstance(item, dict)
        }
        annotations = self.config.get("scenario", {}).get("island_annotations", [])
        if not annotations:
            return [(name, points) for name, points in land_polygons.items() if len(points) >= 3]
        result = []
        for annotation in annotations:
            polygon_name = str(annotation.get("land_polygon", annotation.get("name", "")))
            points = land_polygons.get(polygon_name, [])
            if len(points) >= 3:
                result.append((str(annotation.get("name", polygon_name)), points))
        return result

    @staticmethod
    def _nearest_point_on_island_segment(lat, lon, start, end):
        """Nearest shoreline point using a local east/north metre projection."""
        mean_lat = math.radians(float(lat))
        north_a = (float(start[0]) - float(lat)) * 111320.0
        east_a = (float(start[1]) - float(lon)) * 111320.0 * math.cos(mean_lat)
        north_b = (float(end[0]) - float(lat)) * 111320.0
        east_b = (float(end[1]) - float(lon)) * 111320.0 * math.cos(mean_lat)
        delta_north = north_b - north_a
        delta_east = east_b - east_a
        length_sq = delta_north * delta_north + delta_east * delta_east
        if length_sq <= 1.0e-9:
            fraction = 0.0
        else:
            fraction = max(0.0, min(1.0, -(
                north_a * delta_north + east_a * delta_east
            ) / length_sq))
        nearest_north = north_a + fraction * delta_north
        nearest_east = east_a + fraction * delta_east
        shoreline_lat = float(lat) + nearest_north / 111320.0
        cos_lat = max(0.1, math.cos(mean_lat))
        shoreline_lon = float(lon) + nearest_east / (111320.0 * cos_lat)
        return shoreline_lat, shoreline_lon, math.hypot(nearest_north, nearest_east)

    def get_island_status(self, lat, lon):
        """Classify a point against the annotated island shoreline.

        ``on_land`` is true only for points inside a labelled land polygon.
        ``shore_distance_m`` and ``nearest_shore_point`` are defined both at
        sea and on land, which lets later transport logic detect and snap a
        sea-to-land crossing without using a pre-authored landing zone.
        """
        point = (float(lat), float(lon))
        nearest = {"island_name": "", "shore_distance_m": float("inf"), "nearest_shore_point": None}
        on_land = False
        island_name = ""
        for name, polygon in self._annotated_island_polygons():
            if self._point_in_polygon(point, polygon):
                on_land = True
                island_name = name
            for index, start in enumerate(polygon):
                end = polygon[(index + 1) % len(polygon)]
                shore_lat, shore_lon, distance = self._nearest_point_on_island_segment(
                    point[0], point[1], start, end
                )
                if distance < nearest["shore_distance_m"]:
                    nearest = {
                        "island_name": name,
                        "shore_distance_m": float(distance),
                        "nearest_shore_point": (shore_lat, shore_lon),
                    }
        if on_land:
            nearest["island_name"] = island_name
        nearest["on_land"] = on_land
        return nearest

    def is_on_annotated_island(self, lat, lon):
        return bool(self.get_island_status(lat, lon)["on_land"])

    @staticmethod
    def _segment_intersection_fraction(start, end, edge_start, edge_end):
        """Return the first-line fraction at a proper 2-D segment intersection."""
        x1, y1 = float(start[1]), float(start[0])
        x2, y2 = float(end[1]), float(end[0])
        x3, y3 = float(edge_start[1]), float(edge_start[0])
        x4, y4 = float(edge_end[1]), float(edge_end[0])
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denominator) <= 1.0e-12:
            return None
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator
        if -1.0e-9 <= t <= 1.0 + 1.0e-9 and -1.0e-9 <= u <= 1.0 + 1.0e-9:
            return min(1.0, max(0.0, t))
        return None

    def _first_island_shore_intersection(self, start, end):
        best = None
        for island_name, polygon in self._annotated_island_polygons():
            for index, edge_start in enumerate(polygon):
                edge_end = polygon[(index + 1) % len(polygon)]
                fraction = self._segment_intersection_fraction(start, end, edge_start, edge_end)
                if fraction is None or (best is not None and fraction >= best[0]):
                    continue
                lat = float(start[0]) + fraction * (float(end[0]) - float(start[0]))
                lon = float(start[1]) + fraction * (float(end[1]) - float(start[1]))
                best = (fraction, island_name, (lat, lon))
        return best

    def _clip_transport_move_to_shore(self, start_lat, start_lon, target_lat, target_lon):
        """Keep a ship at the sea-side of its first attempted island crossing."""
        start = (float(start_lat), float(start_lon))
        end = (float(target_lat), float(target_lon))
        crossing = self._first_island_shore_intersection(start, end)
        if crossing is None or self.is_on_annotated_island(*start):
            return end[0], end[1], None
        fraction, _island_name, shore = crossing
        distance, _bearing = self._distance_and_bearing(*start, *end)
        offset_m = float(self.config.get("scenario", {}).get(
            "landing_shore_stop_offset_m", 100.0
        ))
        sea_fraction = max(0.0, fraction - offset_m / max(1.0, distance))
        berth_lat = start[0] + sea_fraction * (end[0] - start[0])
        berth_lon = start[1] + sea_fraction * (end[1] - start[1])
        return berth_lat, berth_lon, shore

    @staticmethod
    def _point_in_polygon(point, polygon):
        y, x = float(point[0]), float(point[1])
        inside = False
        j = len(polygon) - 1
        for i in range(len(polygon)):
            yi, xi = float(polygon[i][0]), float(polygon[i][1])
            yj, xj = float(polygon[j][0]), float(polygon[j][1])
            denominator = yj - yi
            if ((yi > y) != (yj > y)) and abs(denominator) > 1.0e-12:
                boundary_x = (xj - xi) * (y - yi) / denominator + xi
                if x < boundary_x:
                    inside = not inside
            j = i
        return inside

    @classmethod
    def _segment_intersects_polygon(cls, start, end, polygon):
        for i in range(len(polygon)):
            a = polygon[i]
            b = polygon[(i + 1) % len(polygon)]
            if cls._segments_intersect(start, end, a, b):
                return True
        return False

    @staticmethod
    def _segments_intersect(a, b, c, d):
        ax, ay = float(a[1]), float(a[0])
        bx, by = float(b[1]), float(b[0])
        cx, cy = float(c[1]), float(c[0])
        dx, dy = float(d[1]), float(d[0])

        def orient(px, py, qx, qy, rx, ry):
            return (qy - py) * (rx - qx) - (qx - px) * (ry - qy)

        def on_segment(px, py, qx, qy, rx, ry):
            eps = 1.0e-12
            return (
                min(px, rx) - eps <= qx <= max(px, rx) + eps
                and min(py, ry) - eps <= qy <= max(py, ry) + eps
            )

        o1 = orient(ax, ay, bx, by, cx, cy)
        o2 = orient(ax, ay, bx, by, dx, dy)
        o3 = orient(cx, cy, dx, dy, ax, ay)
        o4 = orient(cx, cy, dx, dy, bx, by)
        eps = 1.0e-12
        if o1 * o2 < -eps and o3 * o4 < -eps:
            return True
        if abs(o1) <= eps and on_segment(ax, ay, cx, cy, bx, by):
            return True
        if abs(o2) <= eps and on_segment(ax, ay, dx, dy, bx, by):
            return True
        if abs(o3) <= eps and on_segment(cx, cy, ax, ay, dx, dy):
            return True
        if abs(o4) <= eps and on_segment(cx, cy, bx, by, dx, dy):
            return True
        return False

    def _known_target_count_norm(self, type_markers, max_count):
        count = 0
        markers = [marker.upper() for marker in type_markers]
        for det in self.detected_targets.values():
            target_type = str(det.get("Type", det.get("type", ""))).upper()
            if any(marker in target_type for marker in markers):
                count += 1
        return min(1.0, count / float(max(1.0, max_count)))

    def _available_count_norm(self, side, role, max_count):
        count = 0
        for platform in self.platforms.values():
            if platform.side == side and platform.role == role and platform.alive and platform.platform_id is not None and not self._is_busy(platform):
                count += 1
        return min(1.0, count / float(max(1.0, max_count)))

    def get_recon_action_table(self):
        fields = list(self.recon_action_config.get("continuous_action_fields", (
            "move_east_norm", "move_north_norm", "altitude_delta_norm"
        )))
        return [
            {
                "id": index,
                "name": name,
                "description": "Continuous reconnaissance movement component in [-1, 1].",
                "action_type": "continuous",
            }
            for index, name in enumerate(fields)
        ]

    def get_attack_action_table(self):
        return [
            {
                "id": int(action.get("id", 0)),
                "name": action.get("name", ""),
                "description": action.get("description", ""),
                "afsim_task": action.get("afsim_task", ""),
                "weapon": action.get("weapon", ""),
                "target_domain": action.get("target_domain", ""),
                "target_slot": action.get("target_slot", None),
                "target_name": action.get("target_name", ""),
            }
            for action in sorted(self.attack_action_config.get("actions", []), key=lambda item: int(item.get("id", 0)))
        ]

    def get_attack_state_fields(self):
        return list(self.attack_state_fields)

    def get_attack_task_state(self, group_id):
        group = self.attack_controller.active_groups.get(group_id)
        if not group:
            return None
        leader = self.attack_controller.ensure_group_leader(group)
        updated_target = self._attack_target_info(group.target_name) if group.target_name else None
        if updated_target:
            group.target.update(updated_target)
        target_slots = self._attack_target_slots(group)
        aircraft = {}
        target_slots_by_aircraft = {}
        for platform in group.platforms:
            platform_target_slots = self._attack_target_slots(group, platform)
            target_slots_by_aircraft[platform.name] = platform_target_slots
            values = self._build_attack_aircraft_obs(platform, group)
            aircraft[platform.name] = {
                "fields": list(self.attack_state_fields),
                "obs": [values.get(field, 0.0) for field in self.attack_state_fields],
                "obs_by_name": {field: values.get(field, 0.0) for field in self.attack_state_fields},
                "action_mask": self._build_attack_action_mask(platform, group, values),
                "target_slots": platform_target_slots,
            }
        return {
            "team": {
                "group_id": group.group_id,
                "team_id": group.fixed_team_id,
                "target_name": group.target_name,
                "target_type": group.target.get("type", ""),
                "target_known": group.target.get("known", False),
                "target_alive": group.target.get("alive", True),
                "last_action_id": group.last_action_id,
                "members": [platform.name for platform in group.platforms],
                "leader_name": leader.name if leader is not None else "",
                "formation_spacing_by_aircraft": dict(group.formation_spacing_by_aircraft),
                "command_speed_mps": float(group.command_speed_mps),
                "target_slots": target_slots,
                "target_slots_by_aircraft": target_slots_by_aircraft,
                "enemy_track_memory_count": len(self.enemy_track_memory),
                "enemy_track_memory": self._enemy_track_memory_records(),
                "target_reservations": dict(getattr(group, "target_reservations", {})),
                "global_target_reservations": dict(getattr(self, "attack_target_reservations", {})),
            },
            "aircraft": aircraft,
            "action_table": self.get_attack_action_table(),
        }

    def _build_attack_aircraft_obs(self, platform, group):
        norm_cfg = self.attack_state_config.get("normalization", {})
        max_distance = float(norm_cfg.get("max_distance_m", 300000.0))
        max_alt = float(norm_cfg.get("max_alt_m", 6000.0))
        max_speed = float(norm_cfg.get("max_speed_mps", 500.0 / 3.6))
        max_weapon_range = float(norm_cfg.get("max_weapon_range_m", 60000.0))
        max_elapsed_steps = float(norm_cfg.get("max_elapsed_steps", 120.0))
        max_formation_spacing = float(norm_cfg.get("max_formation_spacing_m", 15000.0))
        max_group_size = float(norm_cfg.get("max_group_size", 3.0))
        leader = self.attack_controller.ensure_group_leader(group)
        is_leader = leader is not None and platform.name == leader.name
        if leader is None or is_leader:
            dist_leader, bearing_leader = 0.0, 0.0
        else:
            dist_leader, bearing_leader = self._distance_and_bearing(
                platform.lat, platform.lon, leader.lat, leader.lon
            )
        alive_members = [member for member in group.platforms if member.alive]
        alive_group_size = len(alive_members)
        formation_aam_total = sum(
            max(0, int(self.attack_ammo.get(member.name, {}).get("fox3", 0)))
            for member in alive_members
        )
        formation_agm_total = sum(
            max(0, int(self.attack_ammo.get(member.name, {}).get("agm", 0)))
            for member in alive_members
        )
        formation_capacity = float(max(1, len(group.platforms)))
        formation_aam_total_norm = min(1.0, formation_aam_total / formation_capacity)
        formation_agm_total_norm = min(1.0, formation_agm_total / formation_capacity)
        carrier = self._first_platform("red", "carrier")
        dist_carrier, bearing_carrier = self._distance_and_bearing_to_platform(platform, carrier)
        sam = self._nearest_detected_target(platform, ["SAM"], use_slant_range=True)
        radar = self._nearest_detected_target(platform, ["RADAR"], use_slant_range=True)
        blue_air = self._nearest_detected_target(platform, ["AIR", "AIRCRAFT", "FIGHTER"], use_slant_range=True)
        ammo = self.attack_ammo.get(platform.name, {"fox3": 1, "agm": 1})
        elapsed_steps = max(0, self.step_count - int(getattr(group, "start_step", self.step_count)))
        primary_alive = bool(group.target.get("alive", True)) if group.target_name else True
        service = self.pending_attack_returns.get(platform.name, {})
        service_phase = str(service.get("phase", ""))
        rearm_duration = float(self.config.get("scenario", {}).get("attack_rearm_duration_seconds", 600.0))
        rearm_remaining = max(0.0, float(service.get("rearm_complete_at", 0.0)) - self._current_sim_time())
        last_action_den = max(1.0, float(max(self.attack_controller.action_specs.keys() or [1])))
        values = {
            "alive": 1.0 if platform.alive else 0.0,
            "hp_norm": self._hp_norm(platform),
            "team_id_norm": self._attack_team_id_norm(group),
            "lat_norm": self._norm_lat(platform.lat),
            "lon_norm": self._norm_lon(platform.lon),
            "alt_norm": min(1.0, max(0.0, float(platform.alt) / max(1.0, max_alt))),
            "heading_sin": self._bearing_sin(getattr(platform, "heading", 0.0)),
            "heading_cos": self._bearing_cos(getattr(platform, "heading", 0.0)),
            "speed_norm": min(1.0, max(0.0, float(platform.speed) / max(1.0, max_speed))),
            "at_home": 1.0 if platform.at_home else 0.0,
            "is_busy": 1.0 if self._is_busy(platform) else 0.0,
            "task_assigned": 1.0 if platform.task_assigned else 0.0,
            "is_leader": 1.0 if is_leader else 0.0,
            "distance_to_leader_norm": self._norm_distance(dist_leader, max_formation_spacing),
            "alive_group_size_norm": min(1.0, max(0.0, alive_group_size / max(1.0, max_group_size))),
            "formation_aam_total_norm": formation_aam_total_norm,
            "formation_agm_total_norm": formation_agm_total_norm,
            "returning_to_carrier": 1.0 if service_phase == "returning" else 0.0,
            "rearming": 1.0 if service_phase == "rearming" else 0.0,
            "rearm_remaining_norm": min(1.0, rearm_remaining / max(1.0, rearm_duration)),
            "carrier_aam_stock_norm": min(1.0, max(0.0, float(self.carrier_ammo_stock.get("fox3", 0)) / 30.0)),
            "carrier_agm_stock_norm": min(1.0, max(0.0, float(self.carrier_ammo_stock.get("agm", 0)) / 50.0)),

            "is_stationary": 1.0 if float(platform.speed) <= 1.0 else 0.0,
            "fire_command_pending": 1.0 if self._active_pending_attack_fire(platform.name) else 0.0,
            "last_action_id_norm": min(1.0, max(0.0, float(getattr(group, "last_action_id", 0)) / last_action_den)),
            "aam_count_norm": min(1.0, max(0.0, float(ammo.get("fox3", 0)))),
            "agm_count_norm": min(1.0, max(0.0, float(ammo.get("agm", 0)))),
            "nearest_known_sam_distance_norm": self._norm_distance(sam["distance_m"], max_distance),
            "nearest_known_sam_bearing_sin": self._bearing_sin(sam["bearing_deg"]),
            "nearest_known_sam_bearing_cos": self._bearing_cos(sam["bearing_deg"]),
            "nearest_known_radar_distance_norm": self._norm_distance(radar["distance_m"], max_distance),
            "nearest_known_radar_bearing_sin": self._bearing_sin(radar["bearing_deg"]),
            "nearest_known_radar_bearing_cos": self._bearing_cos(radar["bearing_deg"]),
            "nearest_known_blue_air_distance_norm": self._norm_distance(blue_air["distance_m"], max_distance),
            "nearest_known_blue_air_bearing_sin": self._bearing_sin(blue_air["bearing_deg"]),
            "nearest_known_blue_air_bearing_cos": self._bearing_cos(blue_air["bearing_deg"]),
            "distance_to_carrier_norm": self._norm_distance(dist_carrier, max_distance),
            "bearing_to_carrier_sin": self._bearing_sin(bearing_carrier),
            "bearing_to_carrier_cos": self._bearing_cos(bearing_carrier),
            "has_fired_any_weapon": 1.0 if self.attack_controller.has_fired_any_weapon(group.group_id, platform.name) else 0.0,
            "has_fired_at_primary_target": 1.0 if group.target_name and self.attack_controller.has_fired_at_primary(group.group_id, platform.name) else 0.0,
            "target_defeated": 1.0 if group.target_name and not primary_alive else 0.0,
        }
        target_slots = self._attack_target_slots(group, platform)
        slot_count = self._attack_target_slot_count()
        for slot in range(slot_count):
            prefix = "target_slot_{0}_".format(slot + 1)
            if slot < len(target_slots):
                values.update(self._attack_target_slot_obs(platform, group, target_slots[slot], prefix, ammo, max_distance, max_alt, max_weapon_range))
            else:
                values.update(self._empty_attack_target_slot_obs(prefix))

        # ---- Friendly unit slots ----
        red_cfg = self.config.get("red", {})
        bounds = self.config.get("scenario", {}).get("bounds", {})
        _add_friendly_carrier(values, self.platforms, red_cfg, self.carrier_ammo_stock, bounds)
        _add_friendly_recon(values, self.platforms, red_cfg, max_speed, bounds)
        _add_friendly_attack(values, self.platforms, red_cfg, group, self.attack_ammo, max_speed, bounds, own_name=platform.name)
        _add_friendly_ground(values, self.platforms, red_cfg, self.ground_ammo, max_speed, bounds)

        return values

    def _target_in_attack_launch_range(self, platform, target, slant_distance=None):
        """Return whether the selected weapon may launch at this target."""
        weapon = self.attack_controller._compatible_weapon(target)
        if weapon == "agm":
            # AGM launch geometry is specified as horizontal ground range.
            # At 3000 m altitude, a 1 km horizontal range corresponds to
            # about 3.16 km slant range.
            ground_range = float(
                self.attack_state_config.get("normalization", {}).get(
                    "agm_horizontal_launch_range_m", 45000.0
                )
            )
            horizontal, _ = self._distance_and_bearing(
                float(platform.lat), float(platform.lon),
                float(target.get("lat", platform.lat)),
                float(target.get("lon", platform.lon)),
            )
            return horizontal <= ground_range
        if slant_distance is None:
            slant_distance, _ = self._slant_distance_and_bearing(
                platform.lat, platform.lon, platform.alt,
                float(target.get("lat", platform.lat)),
                float(target.get("lon", platform.lon)),
                float(target.get("alt", 0.0)),
            )
        return float(slant_distance) <= self._attack_weapon_range(target)

    def _attack_weapon_range(self, target, fallback=926000.0):
        weapon = self.attack_controller._compatible_weapon(target)
        norm_cfg = self.attack_state_config.get("normalization", {})
        key = "aam_weapon_range_m" if weapon == "fox3" else "agm_weapon_range_m"
        return float(norm_cfg.get(key, fallback))

    def _attack_target_slot_obs(self, platform, group, target, prefix, ammo, max_distance, max_alt, max_weapon_range):
        del platform, group, ammo, max_distance, max_weapon_range
        if not target.get("known", False):
            return self._empty_attack_target_slot_obs(prefix)
        target_alt = float(target.get("alt", 0.0))
        return {
            prefix + "lat_norm": self._norm_lat(float(target.get("lat", 0.0))),
            prefix + "lon_norm": self._norm_lon(float(target.get("lon", 0.0))),
            prefix + "alt_norm": min(1.0, max(0.0, target_alt / max(1.0, max_alt))),
            prefix + "hp_norm": float(target.get("hp_norm", 0.0)),
            prefix + "aam_ammo_norm": float(target.get("aam_ammo_norm", 0.0)),
            prefix + "agm_ammo_norm": float(target.get("agm_ammo_norm", 0.0)),
            prefix + "ground_ammo_norm": float(target.get("ground_ammo_norm", 0.0)),
            prefix + "sam_ammo_norm": float(target.get("sam_ammo_norm", 0.0)),
        }

    @staticmethod
    def _empty_attack_target_slot_obs(prefix):
        return {
            prefix + "lat_norm": 0.0,
            prefix + "lon_norm": 0.0,
            prefix + "alt_norm": 0.0,
            prefix + "hp_norm": 0.0,
            prefix + "aam_ammo_norm": 0.0,
            prefix + "agm_ammo_norm": 0.0,
            prefix + "ground_ammo_norm": 0.0,
            prefix + "sam_ammo_norm": 0.0,
        }
    def _ground_attack_unlocked(self):
        """Allow attacks on blue ground forces only after key air defenses fall."""
        prerequisite_targets = (
            "blue_radar_1",
            "blue_sam_1",
            "blue_sam_2",
            "blue_sam_4",
        )
        return all(
            target is not None and not target.alive
            for target in (self.platforms.get(name) for name in prerequisite_targets)
        )

    def _build_attack_action_mask(self, platform, group, values):
        del values
        ammo = self.attack_ammo.get(platform.name, {"fox3": 1, "agm": 1})
        target_slots = self._attack_action_target_slots()
        service_phase = str(
            self.pending_attack_returns.get(platform.name, {}).get("phase", "")
        )
        specs = sorted(self.attack_controller.action_specs.items())
        if service_phase in ("returning", "rearming"):
            return [
                1.0 if spec.get("name") == "RETURN_HOME" else 0.0
                for _, spec in specs
            ]
        if self._active_pending_attack_fire(platform.name):
            return [
                1.0 if spec.get("name") == "HOLD" else 0.0
                for _, spec in specs
            ]
        if (
            int(ammo.get("fox3", 0)) <= 0
            and int(ammo.get("agm", 0)) <= 0
        ):
            return [
                1.0 if spec.get("name") == "RETURN_HOME" else 0.0
                for _, spec in specs
            ]

        leader = self.attack_controller.ensure_group_leader(group)
        is_leader = leader is not None and platform.name == leader.name
        mask = []
        for _, spec in specs:
            name = str(spec.get("name", ""))
            task = str(spec.get("afsim_task", ""))
            if not platform.alive:
                available = 1.0 if name == "HOLD" else 0.0
            elif name == "HOLD":
                # HOLD is reserved for the pending-fire acknowledgement window
                # handled above. A live aircraft in normal combat must choose
                # an attack, rejoin, or return action instead of stopping.
                available = 0.0
            elif name == "RETURN_HOME":
                # Unlock return only after this aircraft has expended at least
                # one of its two initial weapons during the current sortie.
                # Rearming restores both counts to one and locks it again.
                available = 1.0 if (
                    int(ammo.get("fox3", 0)) < 1
                    or int(ammo.get("agm", 0)) < 1
                ) else 0.0
            elif task == "ATTACK_REJOIN_FORMATION":
                available = 1.0 if not is_leader else 0.0
            elif task == "ATTACK_TARGET_SLOT":
                slot = int(spec.get("target_slot", -1))
                if not self._is_attack_slot_available(
                    platform, group, target_slots, slot, ammo, 0.0
                ):
                    available = 0.0
                else:
                    target = target_slots[slot]
                    if (
                        self._attack_target_category(target) == "ground"
                        and not self._ground_attack_unlocked()
                    ):
                        available = 0.0
                    elif not is_leader:
                        target_lat = float(target.get("lat", 0.0))
                        target_lon = float(target.get("lon", 0.0))
                        target_alt = float(target.get("alt", 0.0))
                        weapon_range = self._attack_weapon_range(target)
                        dist, _ = self._slant_distance_and_bearing(
                            platform.lat, platform.lon, float(platform.alt),
                            target_lat, target_lon, target_alt,
                        )
                        # Followers may independently approach a target; range is required only to fire.
                        available = 1.0
                    else:
                        available = 1.0
            else:
                available = 0.0
            mask.append(available)
        # With normal-combat HOLD masked, an aircraft can have no valid target
        # while RETURN_HOME is still locked because neither weapon was fired.
        # Keep the discrete distribution valid by making return the sole safe
        # fallback; never re-enable an arbitrary stop command.
        if platform.alive and not any(mask):
            for index, (_, spec) in enumerate(specs):
                if spec.get("name") == "RETURN_HOME":
                    mask[index] = 1.0
                    break
        return mask

    def _attack_target_reserved_by(self, group, target_name):
        if not target_name:
            return None
        reserved_by = getattr(self, "attack_target_reservations", {}).get(target_name)
        if reserved_by:
            return reserved_by
        return getattr(group, "target_reservations", {}).get(target_name)

    def _is_attack_slot_available(self, platform, group, target_slots, slot, ammo, max_weapon_range):
        del group, max_weapon_range
        if slot < 0 or slot >= len(target_slots):
            return False
        target = target_slots[slot]
        if not target.get("alive", False):
            return False
        weapon = self.attack_controller._compatible_weapon(target)
        return int(ammo.get(weapon, 0)) > 0
    def _attack_target_slots(self, group, platform=None):
        """Return one permanent, identity-stable slot per configured target.

        Slot identity never depends on distance, observer, visibility, or task.
        Reconnaissance and attack-sensor reports only change whether the fixed
        slot contains a currently known track; they never move a target to a
        different action index.
        """
        del group, platform
        return [
            dict(self.attack_target_snapshots.get(target_name, {
                "name": target_name,
                "known": False,
                "alive": False,
                "type": self._configured_blue_target_role(target_name),
            }))
            for target_name in self.attack_fixed_target_names
        ]

    def _attack_action_target_slots(self):
        """Return real fixed identities for action execution and masks only."""
        return [
            self._fixed_attack_action_target_info(target_name)
            for target_name in self.attack_fixed_target_names
        ]

    def _fixed_attack_action_target_info(self, target_name):
        platform = self.platforms.get(target_name)
        snapshot = self.attack_target_snapshots.get(target_name)
        if snapshot is not None:
            info = dict(snapshot)
            # Liveness is combat state; coordinates remain prior/latest intel.
            if platform is not None:
                info["alive"] = bool(platform.alive)
            return info
        return {
            "name": target_name,
            "known": False,
            "alive": bool(platform.alive) if platform is not None else False,
            "type": self._configured_blue_target_role(target_name),
        }

    def _fixed_attack_target_info(self, target_name):
        info = self._attack_target_info(target_name)
        if info:
            return info
        return {
            "name": target_name,
            "known": False,
            "alive": False,
            "type": "",
        }
    def _attack_target_category(self, target):
        target_type = str(target.get("type", "")).upper()
        if self._target_is_air(target_type):
            return "air"
        if "SAM" in target_type:
            return "sam"
        if "RADAR" in target_type:
            return "radar"
        return "ground"

    def _attack_local_target_distance(self, platform, detection):
        distance, _ = self._slant_distance_and_bearing(
            platform.lat,
            platform.lon,
            platform.alt,
            float(detection.get("Lat", detection.get("lat", platform.lat))),
            float(detection.get("Lon", detection.get("lon", platform.lon))),
            float(detection.get("Alt", detection.get("alt", 0.0))),
        )
        return float(distance)

    def _attack_local_target_info(self, aircraft_name, target_name):
        contacts = self.attack_local_detections.get(aircraft_name, {})
        det = contacts.get(target_name)
        if not det:
            return None
        reference_time = self._decision_window_time()
        report_time = float(det.get("ReportTime", reference_time))
        reported_track_age = max(0.0, float(det.get("TrackAge", 0.0)))
        current_track_age = reported_track_age + max(0.0, reference_time - report_time)
        if current_track_age > float(self.attack_local_track_ttl_sec):
            contacts.pop(target_name, None)
            return None
        platform = self.platforms.get(target_name)
        return {
            "name": target_name,
            "known": True,
            "local_only": target_name not in self.detected_targets,
            "track_source": det.get("TrackSource", "attack_sensor_master_track"),
            "track_age": current_track_age,
            "alive": platform.alive if platform is not None else bool(det.get("alive", True)),
            "lat": float(det.get("Lat", det.get("lat", 0.0))),
            "lon": float(det.get("Lon", det.get("lon", 0.0))),
            "alt": float(det.get("Alt", det.get("alt", 0.0))),
            "type": det.get("Type", det.get("type", "")),
            "hp_norm": self._hp_norm(platform),
        }

    def _attack_target_slot_count(self):
        slots = [int(spec.get("target_slot", -1)) for spec in self.attack_controller.action_specs.values() if spec.get("afsim_task") == "ATTACK_TARGET_SLOT"]
        if slots:
            return max(slots) + 1
        return int(self.attack_action_config.get("target_slot_count", 0))

    def _is_valid_attack_target(self, target):
        name = target.get("name", "")
        platform = self.platforms.get(name)
        if platform and platform.side != "blue":
            return False
        return bool(target.get("known", False)) and bool(target.get("alive", True))
    def _attack_target_info(self, target_name):
        det = self.detected_targets.get(target_name)
        if det:
            return {
                "name": target_name,
                "known": True,
                "alive": self.platforms.get(target_name, PlatformState(target_name, "", "")).alive if target_name in self.platforms else True,
                "lat": float(det.get("Lat", det.get("lat", 0.0))),
                "lon": float(det.get("Lon", det.get("lon", 0.0))),
                "alt": float(det.get("Alt", det.get("alt", 0.0))),
                "type": det.get("Type", det.get("type", "")),
                "hp_norm": self._hp_norm(self.platforms.get(target_name)),
            }
        platform = self.platforms.get(target_name)
        if platform and platform.detected:
            return {
                "name": target_name,
                "known": True,
                "alive": platform.alive,
                "lat": platform.lat,
                "lon": platform.lon,
                "alt": platform.alt,
                "type": platform.platform_type or platform.role,
                "hp_norm": self._hp_norm(platform),
            }
        if platform:
            return {
                "name": target_name,
                "known": False,
                "alive": platform.alive,
                "lat": platform.lat,
                "lon": platform.lon,
                "alt": platform.alt,
                "type": platform.platform_type or platform.role,
                "hp_norm": self._hp_norm(platform),
            }
        return None


    @staticmethod
    def _target_is_air(target_type):
        upper = str(target_type).upper()
        return "AIR" in upper or "FIGHTER" in upper or "AIRCRAFT" in upper

    @staticmethod
    def _target_type_code(target_type):
        upper = str(target_type).upper()
        if "SAM" in upper:
            return 0.8
        if "RADAR" in upper:
            return 0.6
        if "GROUND" in upper or "FORCE" in upper or "LAND" in upper:
            return 0.4
        if "AIR" in upper or "FIGHTER" in upper or "AIRCRAFT" in upper:
            return 0.2
        return 0.0

    def get_recon_task_state(self, group_id):
        group = self.recon_controller.active_groups.get(group_id)
        if not group:
            return None
        leader = self.recon_controller.ensure_group_leader(group)
        aircraft = {}
        for platform in group.platforms:
            values = self._build_recon_aircraft_obs(platform, group)
            aircraft[platform.name] = {
                "fields": list(self.recon_state_fields),
                "obs": [values.get(field, 0.0) for field in self.recon_state_fields],
                "obs_by_name": {field: values.get(field, 0.0) for field in self.recon_state_fields},
                "action_mask": self._build_recon_action_mask(platform, group),
            }
        return {
            "team": {
                "group_id": group.group_id,
                "team_id": group.fixed_team_id,
                "last_action_id": group.last_action_id,
                "members": [platform.name for platform in group.platforms],
                "leader_name": leader.name if leader is not None else "",
                "leader_alive": bool(leader is not None),
                "formation_spacing_by_aircraft": dict(group.formation_spacing_by_aircraft),
            },
            "aircraft": aircraft,
            "action_table": self.get_recon_action_table(),
        }

    def _build_recon_aircraft_obs(self, platform, group):
        norm_cfg = self.recon_state_config.get("normalization", {})
        max_speed = float(norm_cfg.get("max_speed_mps", 138.888889))
        max_alt = float(norm_cfg.get("max_alt_m", 10000.0))
        max_group_size = float(norm_cfg.get("max_group_size", 3.0))
        leader = self.recon_controller.ensure_group_leader(group)
        is_leader = leader is not None and platform.name == leader.name
        if leader is None or is_leader:
            dist_leader, _bearing_leader = 0.0, 0.0
        else:
            dist_leader, _bearing_leader = self._distance_and_bearing(
                platform.lat, platform.lon, leader.lat, leader.lon
            )
        alive_group_size = sum(1 for member in group.platforms if member.alive)
        returning_to_carrier = str(platform.task).upper() == "RETREAT"

        values = {
            "alive": 1.0 if platform.alive else 0.0,
            "lat_norm": self._norm_lat(platform.lat),
            "lon_norm": self._norm_lon(platform.lon),
            "alt_norm": self._norm_alt(platform.alt, max_alt),
            "heading_sin": self._bearing_sin(getattr(platform, "heading", 0.0)),
            "heading_cos": self._bearing_cos(getattr(platform, "heading", 0.0)),
            "speed_norm": min(1.0, max(0.0, float(getattr(platform, "speed", 0.0)) / max(1.0, max_speed))),
            "at_home": 1.0 if platform.at_home else 0.0,
            "hp_norm": self._hp_norm(platform),
            "returning_to_carrier": 1.0 if returning_to_carrier else 0.0,
            "is_stationary": 1.0 if float(platform.speed) <= 1.0 else 0.0,

            "team_id_norm": self._recon_team_id_norm(group),
            "is_leader": 1.0 if is_leader else 0.0,
            "distance_to_leader_norm": self._norm_distance(dist_leader, float(norm_cfg.get("max_formation_spacing_m", 15000.0))),
            "alive_group_size_norm": min(1.0, max(0.0, alive_group_size / max(1.0, max_group_size))),
        }

        # Friendly unit slots
        red_cfg = self.config.get("red", {})
        bounds = self.config.get("scenario", {}).get("bounds", {})
        _add_friendly_carrier(values, self.platforms, red_cfg, self.carrier_ammo_stock, bounds)
        _add_friendly_recon(values, self.platforms, red_cfg, max_speed, bounds, own_name=platform.name)
        _add_friendly_attack(values, self.platforms, red_cfg, group, self.attack_ammo, max_speed, bounds, own_name="")
        _add_friendly_ground(values, self.platforms, red_cfg, self.ground_ammo, max_speed, bounds)

        # Enemy target slots (same snapshots as attack)
        target_slots = self._attack_target_slots(group, platform)
        slot_count = self._attack_target_slot_count()
        for slot in range(slot_count):
            prefix = "target_slot_{0}_".format(slot + 1)
            if slot < len(target_slots):
                values.update(self._attack_target_slot_obs(platform, group, target_slots[slot], prefix, {}, float(norm_cfg.get("max_distance_m", 250000.0)), max_alt, float(norm_cfg.get("aam_weapon_range_m", 40000))))
            else:
                values.update(self._empty_attack_target_slot_obs(prefix))
        return values
    def _recon_cell_for_position(self, group, lat, lon):
        area = group.area
        radius = float(area.get("radius_m", 0.0))
        if radius <= 0.0:
            return None
        north_m, east_m = self._relative_north_east(area.get("lat", 0.0), area.get("lon", 0.0), lat, lon)
        if math.hypot(north_m, east_m) > radius:
            return None
        grid_size = max(1, int(getattr(group, "coverage_grid_size", 5)))
        side_m = radius * 2.0
        row = int((north_m + radius) / side_m * grid_size)
        col = int((east_m + radius) / side_m * grid_size)
        row = min(grid_size - 1, max(0, row))
        col = min(grid_size - 1, max(0, col))
        if (row, col) not in self._recon_valid_cells(area, grid_size):
            return None
        return row, col

    def _nearest_uncovered_recon_cell(self, platform, group):
        area = group.area
        radius = float(area.get("radius_m", 0.0))
        if radius <= 0.0:
            return {"distance_m": float("inf"), "bearing_deg": 0.0, "lat": None, "lon": None}
        grid_size = max(1, int(getattr(group, "coverage_grid_size", 5)))
        area_name = self._recon_area_name(area)
        covered = set(self.recon_area_coverage.get(area_name, set()))
        valid_cells = self._recon_valid_cells(area, grid_size)
        cell_m = (radius * 2.0) / float(grid_size)
        best = {"distance_m": float("inf"), "bearing_deg": 0.0, "lat": None, "lon": None}
        for row in range(grid_size):
            for col in range(grid_size):
                if (row, col) not in valid_cells:
                    continue
                if (row, col) in covered:
                    continue
                north_m = -radius + (row + 0.5) * cell_m
                east_m = -radius + (col + 0.5) * cell_m
                lat, lon = self._offset_lat_lon(area.get("lat", 0.0), area.get("lon", 0.0), north_m, east_m)
                distance, bearing = self._distance_and_bearing(platform.lat, platform.lon, lat, lon)
                if distance < best["distance_m"]:
                    best = {"distance_m": distance, "bearing_deg": bearing, "lat": lat, "lon": lon}
        return best
    def _build_recon_action_mask(self, platform, group):
        # Continuous Box actions do not use categorical availability masks.
        # Keep a fixed three-value validity vector for diagnostics and generic
        # rollout plumbing; the official continuous actor ignores it.
        available = 1.0 if platform.alive else 0.0
        return [available] * int(self.recon_controller.continuous_action_dim)

    def _recon_group_in_transit(self, group, area_lat, area_lon, area_radius):
        for member in group.platforms:
            if not member.alive:
                continue
            distance, _ = self._distance_and_bearing(member.lat, member.lon, area_lat, area_lon)
            if distance > area_radius:
                return True
        return False

    def _recon_action_keeps_formation(self, platform, group, spec):
        if len(group.platforms) <= 1:
            return True
        projected = self.recon_controller._area_for_aircraft_action(group.area, spec, platform, None)
        candidate_lat = float(projected.get("lat", platform.lat))
        candidate_lon = float(projected.get("lon", platform.lon))
        max_radius = float(self.recon_controller.formation_search_radius_m)
        formation_index = group.platforms.index(platform)
        peers = group.platforms[1:] if formation_index == 0 else group.platforms[:1]
        for peer in peers:
            if not peer.alive:
                continue
            current_distance, _ = self._distance_and_bearing(platform.lat, platform.lon, peer.lat, peer.lon)
            candidate_distance, _ = self._distance_and_bearing(candidate_lat, candidate_lon, peer.lat, peer.lon)
            if candidate_distance > max_radius and candidate_distance >= current_distance:
                return False
        return True

    def _first_platform(self, side, role):
        for platform in self.platforms.values():
            if platform.side == side and platform.role == role:
                return platform
        return None

    def _nearest_platform(self, platform, candidates):
        nearest = None
        nearest_distance = float("inf")
        for candidate in candidates:
            distance, _ = self._distance_and_bearing_to_platform(platform, candidate)
            if distance < nearest_distance:
                nearest = candidate
                nearest_distance = distance
        return nearest

    def _nearest_detected_target(self, platform, type_markers, use_slant_range=False):
        best = {"distance_m": float("inf"), "bearing_deg": 0.0, "name": "", "lat": None, "lon": None, "alt": None, "type": ""}
        for name, det in self.detected_targets.items():
            target_type = str(det.get("Type", "")).upper()
            if not any(marker in target_type for marker in type_markers):
                continue
            lat = det.get("Lat", det.get("lat"))
            lon = det.get("Lon", det.get("lon"))
            if lat is None or lon is None:
                continue
            alt = float(det.get("Alt", det.get("alt", 0.0)))
            if use_slant_range:
                distance, bearing = self._slant_distance_and_bearing(
                    platform.lat,
                    platform.lon,
                    platform.alt,
                    float(lat),
                    float(lon),
                    alt,
                )
            else:
                distance, bearing = self._distance_and_bearing(platform.lat, platform.lon, float(lat), float(lon))
            if distance < best["distance_m"]:
                best = {"distance_m": distance, "bearing_deg": bearing, "name": name, "lat": float(lat), "lon": float(lon), "alt": alt, "type": target_type}
        return best

    def _nearest_recon_threat(self, platform):
        if platform is None:
            return None
        threat = self._nearest_detected_target(platform, ["SAM", "RADAR"])
        if threat["lat"] is None or math.isinf(threat["distance_m"]):
            return None
        return threat

    def _update_recon_coverage(self, group):
        area = group.area
        radius = float(area.get("radius_m", 0.0))
        if radius <= 0.0:
            return
        grid_size = max(1, int(getattr(group, "coverage_grid_size", 5)))
        area_name = self._recon_area_name(area)
        global_covered = self.recon_area_coverage.setdefault(area_name, set())
        valid_cells = self._recon_valid_cells(area, grid_size)
        now = float(self._current_sim_time())
        for platform in group.platforms:
            if not platform.alive:
                continue
            cell = self._recon_cell_for_position(
                group, platform.lat, platform.lon
            )
            if cell is None or cell not in valid_cells:
                continue
            self.recon_area_last_observed_time[area_name] = now
            if float(getattr(group, "survey_started_at", -1.0)) < 0.0:
                group.survey_started_at = now
            if cell not in global_covered:
                global_covered.add(cell)
                credits = getattr(group, "coverage_cells_by_aircraft", None)
                if credits is None:
                    credits = {}
                    group.coverage_cells_by_aircraft = credits
                credits.setdefault(platform.name, set()).add(cell)
        group.covered_cells = set(global_covered)

    def _recon_coverage_ratio(self, group):
        grid_size = max(1, int(getattr(group, "coverage_grid_size", 5)))
        return self._recon_area_coverage_ratio(group.area, grid_size)

    def _recon_aircraft_coverage_credit_ratio(self, group, aircraft_name):
        grid_size = max(1, int(getattr(group, "coverage_grid_size", 5)))
        valid_cells = self._recon_valid_cells(group.area, grid_size)
        if not valid_cells:
            return 0.0
        credited = getattr(group, "coverage_cells_by_aircraft", {}).get(aircraft_name, set())
        return min(1.0, len(set(credited) & valid_cells) / float(len(valid_cells)))

    def _distance_and_bearing_to_platform(self, platform, target):
        if target is None:
            return float("inf"), 0.0
        return self._distance_and_bearing(platform.lat, platform.lon, target.lat, target.lon)

    @staticmethod
    def _offset_lat_lon(lat, lon, north_m, east_m):
        lat_offset = float(north_m) / 111320.0
        cos_lat = max(0.1, math.cos(math.radians(float(lat))))
        lon_offset = float(east_m) / (111320.0 * cos_lat)
        return float(lat) + lat_offset, float(lon) + lon_offset
    @staticmethod
    def _relative_north_east(origin_lat, origin_lon, target_lat, target_lon):
        north_m = (float(target_lat) - float(origin_lat)) * 111320.0
        cos_lat = max(0.1, math.cos(math.radians(float(origin_lat))))
        east_m = (float(target_lon) - float(origin_lon)) * 111320.0 * cos_lat
        return north_m, east_m

    @staticmethod
    def _distance_and_bearing(lat1, lon1, lat2, lon2):
        lat1_rad = math.radians(float(lat1))
        lat2_rad = math.radians(float(lat2))
        dlat = lat2_rad - lat1_rad
        dlon = math.radians(float(lon2) - float(lon1))
        a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2.0) ** 2
        distance = 6371000.0 * 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
        y = math.sin(dlon) * math.cos(lat2_rad)
        x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
        bearing = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
        return distance, bearing

    @classmethod
    def _slant_distance_and_bearing(cls, lat1, lon1, alt1, lat2, lon2, alt2):
        horizontal_distance, bearing = cls._distance_and_bearing(lat1, lon1, lat2, lon2)
        altitude_delta = float(alt2) - float(alt1)
        return math.hypot(horizontal_distance, altitude_delta), bearing

    @staticmethod
    def _norm_distance(distance, max_distance):
        if math.isinf(distance):
            return 1.0
        return min(1.0, max(0.0, float(distance) / max(1.0, float(max_distance))))

    @staticmethod
    def _bearing_sin(bearing_deg):
        return math.sin(math.radians(float(bearing_deg)))

    @staticmethod
    def _bearing_cos(bearing_deg):
        return math.cos(math.radians(float(bearing_deg)))
    def set_negative_rewards_enabled(self, enabled):
        """Globally enable or suppress every negative reward component."""
        self.negative_rewards_enabled = bool(enabled)
        self.reward_manager.set_negative_rewards_enabled(self.negative_rewards_enabled)

    def apply_negative_reward_switch(self, value):
        value = float(value)
        if value < 0.0 and not self.negative_rewards_enabled:
            return 0.0
        return value

    def compute_reward(self):
        reward = self.reward_manager.compute(self)
        self.last_reward_details = list(self.reward_manager.last_details)
        return reward

    def platform_state_age_seconds(self):
        """Wall-clock age of the newest PlatformState/MoveUpdate heartbeat."""
        last = float(getattr(self, "last_platform_state_wall_time", 0.0) or 0.0)
        if last <= 0.0:
            return float("inf")
        return max(0.0, time.monotonic() - last)

    def decision_ready_age_seconds(self):
        """Wall-clock age of the native pause control heartbeat."""
        last = float(getattr(self, "last_decision_ready_wall_time", 0.0) or 0.0)
        if last <= 0.0:
            return float("inf")
        return max(0.0, time.monotonic() - last)

    def _current_sim_time(self):
        return max(
            [float(p.last_update) for p in self.platforms.values() if p.last_update > 0.0]
            or [float(self.step_count) * float(getattr(self, "decision_sim_seconds", self.config.get("scenario", {}).get("decision_sim_seconds", self.decision_seconds)))]
        )

    def _done_rule_enabled(self, name):
        rule = self.done_rules_config.get("rules", {}).get(name, {})
        return bool(rule.get("enabled", False))

    def _fixed_horizon_rule(self):
        return self.done_rules_config.get("rules", {}).get("fixed_horizon", {})

    def _mission_duration_seconds(self):
        return max(1.0, float(self._fixed_horizon_rule().get("simulation_seconds", 21600.0)))

    def _mission_time_norm(self):
        return min(1.0, max(0.0, self._current_sim_time() / self._mission_duration_seconds()))

    def _final_objective_score(self):
        rule = self._fixed_horizon_rule()
        center_name = str(rule.get("score_center_platform", "blue_base"))
        center = self.platforms.get(center_name)
        if center is not None and (abs(float(center.lat)) > 1.0e-9 or abs(float(center.lon)) > 1.0e-9):
            center_lat, center_lon = float(center.lat), float(center.lon)
        else:
            objective = next(
                (item for item in self.config.get("ground_objectives", []) if item.get("name") == center_name),
                self.config.get("ground_objectives", [{}])[0],
            )
            center_lat = float(objective.get("lat", 0.0))
            center_lon = float(objective.get("lon", 0.0))

        radius_m = float(rule.get("score_radius_m", 1000.0))
        score_side = str(rule.get("score_side", "red"))
        score_role = str(rule.get("score_role", "ground_force"))
        require_landed = bool(rule.get("require_landed", True))
        units = []
        raw_score = 0.0
        for platform in self.platforms.values():
            if platform.side != score_side or platform.role != score_role:
                continue
            if not platform.alive or platform.current_hp <= 0.0:
                continue
            status = self.ground_status.get(platform.name, {"on_ship": False, "landed": True})
            if require_landed and (status.get("on_ship", False) or not status.get("landed", False)):
                continue
            distance_m, _ = self._distance_and_bearing(platform.lat, platform.lon, center_lat, center_lon)
            if distance_m > radius_m:
                continue
            hp = max(0.0, float(platform.current_hp))
            raw_score += hp
            units.append({"name": platform.name, "current_hp": hp, "distance_m": float(distance_m)})

        configured_max = float(rule.get("maximum_score_hp", 0.0))
        if configured_max <= 0.0:
            configured_max = sum(
                max(0.0, float(platform.max_hp))
                for platform in self.platforms.values()
                if platform.side == score_side and platform.role == score_role
            )
        normalized = raw_score / max(1.0, configured_max)
        return {
            "raw": float(raw_score),
            "normalized": min(1.0, max(0.0, float(normalized))),
            "unit_count": len(units),
            "units": units,
            "radius_m": radius_m,
            "center": center_name,
        }

    def _settle_final_score(self, reason):
        if self.final_score_settled:
            return
        score = self._final_objective_score()
        self.final_score_raw = float(score["raw"])
        self.final_score_norm = float(score["normalized"])
        self.final_score_unit_count = int(score["unit_count"])
        self.final_score_units = list(score["units"])
        self.final_score_sim_time = float(self._current_sim_time())
        self.final_score_settled = True
        reward_per_hp = float(self._fixed_horizon_rule().get("terminal_reward_per_hp", 1.0))
        self.last_reward_events.append({
            "type": "final_objective_hp_score",
            "reason": str(reason),
            "score_raw": self.final_score_raw,
            "score_norm": self.final_score_norm,
            "unit_count": self.final_score_unit_count,
            "units": list(self.final_score_units),
            "reward": self.final_score_raw * reward_per_hp,
        })

    def is_done(self):
        result = str(self.episode_result or "").upper()
        red_ground_alive = any(
            p.alive and p.current_hp > 0.0
            for p in self.platforms.values()
            if p.side == "red" and p.role == "ground_force"
        )
        if (
            self._done_rule_enabled("red_ground_defeated")
            and (result == "BLUE_DEFENSE_SUCCESS" or not red_ground_alive)
        ):
            self._settle_final_score("red_ground_defeated")
            self.episode_result = "BLUE_DEFENSE_SUCCESS"
            self.episode_done_reason = "red_ground_defeated"
            return True

        horizon_reached = self._current_sim_time() >= self._mission_duration_seconds()
        if self._done_rule_enabled("fixed_horizon") and (
            horizon_reached or result == "FIXED_HORIZON_COMPLETE"
        ):
            self._settle_final_score("fixed_horizon")
            self.episode_result = "FIXED_HORIZON_COMPLETE"
            self.episode_done_reason = "fixed_horizon"
            return True

        if (self._done_rule_enabled("timeout") or not self.live_mode) and self.step_count >= self.max_steps:
            self._settle_final_score("timeout")
            self.episode_result = "SAFETY_TIMEOUT"
            self.episode_done_reason = "timeout"
            return True

        self.episode_done_reason = "none"
        return False

