import json
import os
from copy import deepcopy


class RewardManager(object):
    """Central reward calculator for the AFSIM island environment.

    It combines explicit env events with state deltas. RL code should use
    env.step(...), while debugging code can inspect env.last_reward_details.
    """

    def __init__(self, rules_path):
        self.rules_path = rules_path
        self.rules = self._load_rules(rules_path)
        self.negative_rewards_enabled = bool(
            self.rules.get("negative_rewards_enabled", False))
        self.previous_snapshot = None
        self.last_details = []

    @staticmethod
    def _merge_rules(base, fragment):
        for key, value in fragment.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                RewardManager._merge_rules(base[key], value)
            else:
                base[key] = deepcopy(value)
        return base

    @classmethod
    def _load_rules(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        fragments = manifest.pop("fragments", None)
        if not fragments:
            return manifest
        merged = dict(manifest)
        base_dir = os.path.dirname(os.path.abspath(path))
        for relative_path in fragments:
            fragment_path = os.path.join(base_dir, str(relative_path))
            with open(fragment_path, "r", encoding="utf-8") as f:
                cls._merge_rules(merged, json.load(f))
        return merged
    def set_negative_rewards_enabled(self, enabled):
        self.negative_rewards_enabled = bool(enabled)

    def _gate_value(self, value):
        value = float(value)
        return value if self.negative_rewards_enabled or value >= 0.0 else 0.0

    def _detail_value(self, value):
        raw_value = float(value)
        return raw_value, self._gate_value(raw_value), raw_value < 0.0 and not self.negative_rewards_enabled


    def reset(self, env):
        self.previous_snapshot = self.snapshot(env)
        self.last_details = []

    def compute(self, env):
        current = self.snapshot(env)
        details = []
        total = 0.0

        event_reward = self._compute_event_rewards(env, details)
        total += event_reward

        if self.previous_snapshot is not None:
            delta_reward = self._compute_delta_rewards(self.previous_snapshot, current, details)
            total += delta_reward

        self.previous_snapshot = deepcopy(current)
        total = self._clip(total)
        self.last_details = details
        return total

    def snapshot(self, env):
        return {
            "sim_time": float(env._current_sim_time()),
            "detected_targets": set(env.detected_targets.keys()),
            "alive": self._alive_counts(env),
            "destroyed": self._destroyed_counts(env),
            "hp": self._hp_totals(env),
            # Reward only a combat-qualified landing transition. The scenario
            # time override still opens landing operationally, but must not
            # grant success shaping to a deadline-missed episode.
            "landing_window_open": bool(
                env.get_landing_window_status(include_open=False).get(
                    "combat_conditions_met", False
                )
            ),
            "landed_ground_count": self._landed_ground_count(env),
            "unloaded_transport_count": self._unloaded_transport_count(env),
            "capture_condition": self._capture_condition(env),
        }

    def _compute_event_rewards(self, env, details):
        reward = 0.0
        event_rewards = self.rules.get("event_rewards", {})
        attack_result_rewards = self.rules.get("attack_result_rewards", {})
        for event in env.last_reward_events:
            event_type = event.get("type", "")
            value = event_rewards.get(event_type)
            if event_type == "attack_result" and event.get("result") in attack_result_rewards:
                value = attack_result_rewards[event.get("result")]
            # Recon reports are observational telemetry, not repeatable team
            # reward events. Older Windows rules omitted explicit zero entries,
            # so do not use legacy packet rewards for these event types.
            if value is None and event_type in (
                    "recon_task_detection", "maintained_detection"):
                value = 0.0
            if value is None:
                value = float(event.get("reward", 0.0))
            combat_value = self._combat_event_value(event)
            value = float(combat_value) if combat_value is not None else float(value)
            if combat_value is None and event_type in ("damage_dealt", "damage_received"):
                amount = max(0.0, float(event.get("amount", 0.0)))
                max_hp = float(event.get("max_hp", 0.0))
                # Legacy non-combat events retain proportional scaling.
                damage_scale = amount / max_hp if max_hp > 0.0 else amount
                value *= damage_scale
            raw_value, value, suppressed = self._detail_value(value)
            if raw_value != 0.0:
                detail = {"source": "event", "type": event_type,
                          "value": value, "event": dict(event)}
                if suppressed:
                    detail.update({"raw_value": raw_value,
                                   "negative_reward_suppressed": True})
                details.append(detail)
            reward += value
        return reward

    def _combat_event_value(self, event):
        """Return fixed combat reward for a blue target, or None for non-combat events."""
        event_type = str(event.get("type", ""))
        if event_type not in ("damage_dealt", "target_destroyed"):
            return None
        if str(event.get("side", "")) != "blue":
            return None
        role = str(event.get("role", ""))
        by_role = self.rules.get("combat_positive_rewards", {}).get("by_target_role", {})
        role_rule = by_role.get(role)
        if role_rule is None:
            return 0.0
        if event_type == "damage_dealt":
            amount = max(0.0, float(event.get("amount", 0.0)))
            return amount * float(role_rule.get("damage_per_hp", 0.0))
        return float(role_rule.get("destroyed", 0.0))
    def _compute_delta_rewards(self, previous, current, details):
        reward = 0.0
        delta_rules = self.rules.get("delta_rewards", {})
        elapsed_seconds = max(0.0, float(current.get("sim_time", 0.0)) - float(previous.get("sim_time", 0.0)))
        reward += self._add_scaled_detail(
            details,
            "delta",
            "simulation_time_cost",
            delta_rules.get("simulation_time_cost_per_minute", 0.0),
            elapsed_seconds / 60.0,
            elapsed_seconds=elapsed_seconds,
        )

        new_detections = len(current["detected_targets"] - previous["detected_targets"])
        reward += self._add_detail(details, "delta", "new_detected_target", delta_rules.get("new_detected_target", 0.0), new_detections)

        reward += self._destroyed_delta(details, previous, current, "blue", "attack_aircraft", "blue_aircraft_destroyed")
        reward += self._destroyed_delta(details, previous, current, "blue", "sam", "blue_sam_destroyed")
        reward += self._destroyed_delta(details, previous, current, "blue", "radar", "blue_radar_destroyed")
        reward += self._destroyed_delta(details, previous, current, "blue", "ground_force", "blue_ground_destroyed")
        reward += self._destroyed_delta(details, previous, current, "red", "recon_aircraft", "red_aircraft_lost")
        reward += self._destroyed_delta(details, previous, current, "red", "attack_aircraft", "red_aircraft_lost")
        reward += self._destroyed_delta(details, previous, current, "red", "transport", "red_transport_lost")
        reward += self._destroyed_delta(details, previous, current, "red", "ground_force", "red_ground_lost")

        if not previous["landing_window_open"] and current["landing_window_open"]:
            reward += self._add_detail(details, "delta", "landing_window_opened", delta_rules.get("landing_window_opened", 0.0), 1)

        landed_delta = current["landed_ground_count"] - previous["landed_ground_count"]
        reward += self._add_detail(details, "delta", "ground_unit_landed", delta_rules.get("ground_unit_landed", 0.0), landed_delta)

        unloaded_delta = current["unloaded_transport_count"] - previous["unloaded_transport_count"]
        reward += self._add_detail(details, "delta", "transport_unloaded", delta_rules.get("transport_unloaded", 0.0), unloaded_delta)

        if not previous["capture_condition"] and current["capture_condition"]:
            reward += self._add_detail(details, "delta", "capture_condition_reached", delta_rules.get("capture_condition_reached", 0.0), 1)

        return reward

    def _destroyed_delta(self, details, previous, current, side, role, reward_key):
        prev = previous["destroyed"].get(side, {}).get(role, 0)
        curr = current["destroyed"].get(side, {}).get(role, 0)
        delta = curr - prev
        value = self.rules.get("delta_rewards", {}).get(reward_key, 0.0)
        return self._add_detail(details, "delta", reward_key, value, delta)

    def _add_detail(self, details, source, reward_type, unit_value, count):
        count = int(count)
        if count <= 0:
            return 0.0
        raw_value = float(unit_value) * count
        raw_value, value, suppressed = self._detail_value(raw_value)
        if raw_value != 0.0:
            detail = {"source": source, "type": reward_type, "count": count,
                      "unit_value": float(unit_value), "value": value}
            if suppressed:
                detail.update({"raw_value": raw_value, "negative_reward_suppressed": True})
            details.append(detail)
        return value

    def _add_scaled_detail(self, details, source, reward_type, unit_value, scale, **metadata):
        scale = max(0.0, float(scale))
        raw_value = float(unit_value) * scale
        raw_value, value, suppressed = self._detail_value(raw_value)
        if raw_value != 0.0:
            detail = {
                "source": source,
                "type": reward_type,
                "scale": scale,
                "unit_value": float(unit_value),
                "value": value,
            }
            detail.update(metadata)
            details.append(detail)
            if suppressed:
                detail.update({"raw_value": raw_value,
                               "negative_reward_suppressed": True})
        return value

    def _clip(self, reward):
        limits = self.rules.get("limits", {})
        reward = max(float(limits.get("min_reward", -1.0e9)), min(float(limits.get("max_reward", 1.0e9)), float(reward)))
        return reward

    @staticmethod
    def _hp_totals(env):
        totals = {}
        for platform in env.platforms.values():
            if platform.max_hp <= 0.0:
                continue
            role = totals.setdefault(platform.side, {}).setdefault(platform.role, {"current": 0.0, "maximum": 0.0})
            role["current"] += max(0.0, float(platform.current_hp))
            role["maximum"] += max(0.0, float(platform.max_hp))
        return totals

    @staticmethod
    def _alive_counts(env):
        counts = {}
        for platform in env.platforms.values():
            counts.setdefault(platform.side, {}).setdefault(platform.role, 0)
            if platform.alive:
                counts[platform.side][platform.role] += 1
        return counts

    @staticmethod
    def _destroyed_counts(env):
        counts = {}
        for platform in env.platforms.values():
            counts.setdefault(platform.side, {}).setdefault(platform.role, 0)
            if not platform.alive:
                counts[platform.side][platform.role] += 1
        return counts

    @staticmethod
    def _landed_ground_count(env):
        count = 0
        for status in env.ground_status.values():
            if status.get("landed", False) and not status.get("on_ship", False):
                count += 1
        return count

    @staticmethod
    def _unloaded_transport_count(env):
        count = 0
        for cargo in env.landing_cargo.values():
            if cargo.get("army_landed", False) and not cargo.get("has_army", True):
                count += 1
        return count

    @staticmethod
    def _capture_condition(env):
        for obj in env.config.get("ground_objectives", []):
            lat = float(obj.get("lat", 0.0))
            lon = float(obj.get("lon", 0.0))
            radius = float(obj.get("radius_m", 5000.0))
            blue_count, red_count = env._objective_presence(lat, lon, radius)
            if red_count > 0 and blue_count == 0:
                return True
        return False
