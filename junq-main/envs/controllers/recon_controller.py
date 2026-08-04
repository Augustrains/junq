import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class ReconAssignment:
    platform: object
    message: dict
    action_name: str


@dataclass
class ReconGroup:
    group_id: str
    area: dict
    platforms: List[object]
    fixed_team_id: str = ""
    last_action_id: int = 0
    leader_action_id: int = 0
    start_sim_time: float = 0.0
    survey_started_at: float = -1.0
    start_step: int = 0
    coverage_grid_size: int = 5
    covered_cells: set = field(default_factory=set)
    coverage_cells_by_aircraft: Dict[str, set] = field(default_factory=dict)
    leader_name: str = ""
    leader_target_area: dict = field(default_factory=dict)
    command_speed_mps: float = 100.0
    command_altitude_m: float = 3000.0
    formation_spacing_by_aircraft: Dict[str, float] = field(default_factory=dict)
    leader_command_version: int = 0
    follower_command_versions: Dict[str, int] = field(default_factory=dict)
    last_continuous_action: Tuple[float, float, float] = (0.0, 1.0, 0.0)


class ReconController(object):
    """Bottom controller for always-on reconnaissance teams."""

    def __init__(self, red_config, action_config=None):
        action_config = action_config or {}
        self.recon_aircraft = list(red_config.get("recon_aircraft", []))
        self.commandable_recon_aircraft = list(
            red_config.get("commandable_recon_aircraft", self.recon_aircraft)
        )
        configured_teams = red_config.get("fixed_recon_teams", [])
        self.fixed_recon_teams = [
            tuple(str(name) for name in team if str(name))
            for team in configured_teams if team
        ]
        flattened = [name for team in self.fixed_recon_teams for name in team]
        if len(flattened) != len(set(flattened)):
            raise ValueError("fixed_recon_teams contains duplicate aircraft")
        unknown = sorted(set(flattened) - set(self.commandable_recon_aircraft))
        if unknown:
            raise ValueError(
                "fixed_recon_teams contains non-commandable aircraft: {0}".format(unknown)
            )
        self.active_groups: Dict[str, ReconGroup] = {}
        self.next_group_index = 1
        self.min_altitude_m = float(action_config.get("min_altitude_m", 1000.0))
        self.max_altitude_m = float(action_config.get("max_altitude_m", 10000.0))
        self.default_altitude_m = float(action_config.get("default_altitude_m", 9144.0))
        self.initial_task_altitude_m = min(
            self.max_altitude_m,
            max(self.min_altitude_m, float(action_config.get("initial_task_altitude_m", 3000.0))),
        )
        self.max_speed_mps = float(action_config.get("max_speed_mps", 500.0 / 3.6))
        # Reconnaissance speed is an actuator constant, not a learned action.
        self.min_speed_mps = self.max_speed_mps
        self.default_speed_mps = self.max_speed_mps
        self.continuous_action = str(action_config.get("action_type", "")).lower() == "continuous"
        self.continuous_action_dim = int(action_config.get("action_dim", 3))
        # `horizontal_step_m` is the maximum distance reachable in one
        # environment decision, not a long-horizon waypoint distance.
        self.horizontal_step_m = float(action_config.get("horizontal_step_m", 10000.0))
        self.decision_sim_seconds = float(action_config.get("decision_sim_seconds", 72.0))
        self.altitude_step_m = float(action_config.get("altitude_step_m", 1000.0))
        self.direction_deadband = float(action_config.get("direction_deadband", 0.05))
        self.formation_spacing_m = float(action_config.get("formation_spacing_m", 2500.0))
        self.min_formation_spacing_m = float(action_config.get("min_formation_spacing_m", self.formation_spacing_m))
        self.max_formation_spacing_m = float(action_config.get("max_formation_spacing_m", 15000.0))
        self.formation_spacing_step_m = float(action_config.get("formation_spacing_step_m", 2500.0))
        self.formation_search_radius_m = float(action_config.get("formation_search_radius_m", 15000.0))
        self._leader_rng = random.Random(int(action_config.get("leader_election_seed", 0)))
        configured_actions = action_config.get("actions", [])
        self.action_specs = {
            int(action["id"]): dict(action)
            for action in configured_actions
        }
        if not self.action_specs and not self.continuous_action:
            self.action_specs = self._default_action_specs()
        self.path_actions = {
            action_id: spec["name"]
            for action_id, spec in self.action_specs.items()
        }

    def initialize_teams(self, platforms: Dict[str, object]) -> List[ReconGroup]:
        """Create stable coordination contexts without assigning a mission."""
        groups = []
        for team_index, member_names in enumerate(self.fixed_recon_teams):
            group_id = "recon_team_{0}".format(team_index + 1)
            existing = self.active_groups.get(group_id)
            if existing is not None:
                groups.append(existing)
                continue
            members = [platforms[name] for name in member_names if name in platforms]
            if not members:
                continue
            group = ReconGroup(
                group_id=group_id,
                area={},
                platforms=members,
                fixed_team_id=group_id,
                leader_name=members[0].name,
                command_speed_mps=self.max_speed_mps,
                command_altitude_m=self.initial_task_altitude_m,
                formation_spacing_by_aircraft={
                    platform.name: self.formation_spacing_m
                    for platform in members[1:]
                },
                follower_command_versions={
                    platform.name: 0 for platform in members[1:]
                },
            )
            self.active_groups[group_id] = group
            groups.append(group)
        return groups

    def select_fixed_team(
        self,
        platforms: Dict[str, object],
        is_busy: Callable[[object], bool],
        fixed_team_index: int,
    ):
        """Return one exact complete recon formation; never mix teams."""
        if not 0 <= int(fixed_team_index) < len(self.fixed_recon_teams):
            return None
        member_names = self.fixed_recon_teams[int(fixed_team_index)]
        members = [platforms.get(name) for name in member_names]
        if len(members) != len(member_names) or not all(
            self._is_available(platform, is_busy) for platform in members
        ):
            return None
        return "recon_fixed_team_{0}".format(int(fixed_team_index) + 1), members

    def can_start(
        self,
        platforms: Dict[str, object],
        is_busy: Callable[[object], bool],
        fixed_team_index: int,
    ) -> bool:
        return self.select_fixed_team(
            platforms, is_busy, fixed_team_index
        ) is not None

    def start_group(
        self,
        area: dict,
        platforms: Dict[str, object],
        is_busy: Callable[[object], bool],
        normalize_area: Callable[[dict], dict],
        fixed_team_index: int,
    ) -> Tuple[Optional[ReconGroup], List[ReconAssignment], Optional[str]]:
        fixed_team = self.select_fixed_team(
            platforms, is_busy, fixed_team_index
        )
        if fixed_team is None:
            return None, [], "fixed_recon_team_unavailable"
        fixed_team_id, selected = fixed_team
        normalized = normalize_area(area)
        group_id = "recon_group_{0}".format(self.next_group_index)
        self.next_group_index += 1
        group = ReconGroup(
            group_id=group_id,
            area=normalized,
            platforms=selected,
            fixed_team_id=fixed_team_id,
            leader_name=selected[0].name,
            leader_target_area=dict(normalized),
            command_speed_mps=self.max_speed_mps,
            command_altitude_m=self.initial_task_altitude_m,
            formation_spacing_by_aircraft={
                platform.name: self.formation_spacing_m for platform in selected[1:]
            },
            leader_command_version=1,
            follower_command_versions={
                platform.name: 1 for platform in selected[1:]
            },
        )
        self.active_groups[group_id] = group

        assignments = []
        for platform in selected:
            # Rule assignment only activates the task. The first movement is
            # selected by the bottom policy instead of being pre-commanded to
            # the configured area center.
            activation_area = dict(normalized)
            activation_area.update({
                "lat": float(platform.lat),
                "lon": float(platform.lon),
                "alt": group.command_altitude_m,
                "width_m": float(normalized.get("local_search_width_m", 10000.0)),
                "height_m": float(normalized.get("local_search_height_m", 10000.0)),
                "duration_sec": min(
                    30.0, float(normalized.get("duration_sec", 60.0))
                ),
                "command_speed_mps": group.command_speed_mps,
            })
            message = self._build_recon_message(platform, activation_area)
            assignments.append(ReconAssignment(
                platform=platform,
                message=message,
                action_name="RECON:{0}".format(normalized.get("name", "")),
            ))
        leader_activation = dict(normalized)
        leader_activation.update({
            "lat": float(selected[0].lat),
            "lon": float(selected[0].lon),
            "alt": group.command_altitude_m,
            "command_speed_mps": group.command_speed_mps,
        })
        group.leader_target_area = leader_activation
        return group, assignments, None

    def create_aircraft_continuous_action_message(
        self,
        group_id: str,
        aircraft_name: str,
        action,
    ) -> Tuple[Optional[dict], Optional[str]]:
        """Apply a leader's continuous command and keep followers in formation.

        Every aircraft still receives an action from the multi-agent interface.
        The leader's vector defines the shared reconnaissance destination; a
        follower's vector is deliberately ignored and is used as its follow
        acknowledgement for that decision.  This matches the attack-team
        leader/HOLD protocol while preserving the continuous recon interface.
        """
        group = self.active_groups.get(group_id)
        if group is None:
            return None, "recon_team_not_found"
        try:
            values = [float(value) for value in action]
        except (TypeError, ValueError):
            return None, "invalid_recon_continuous_action"
        if len(values) != self.continuous_action_dim or len(values) != 3:
            return None, "invalid_recon_continuous_action_dim"
        if not all(math.isfinite(value) for value in values):
            return None, "nonfinite_recon_continuous_action"
        platform = next((p for p in group.platforms if p.name == aircraft_name), None)
        if platform is None:
            return None, "recon_aircraft_not_in_team"
        if not platform.alive:
            return None, "recon_aircraft_destroyed"

        leader = self.ensure_group_leader(group)
        if leader is None:
            return None, "recon_group_has_no_surviving_aircraft"
        if platform.name != leader.name:
            # Followers do not independently redirect the fixed team.  Send a
            # formation command once per leader command version.
            applied_version = int(group.follower_command_versions.get(platform.name, 0))
            if applied_version >= group.leader_command_version:
                return None, None
            leader_area = dict(group.leader_target_area)
            if "lat" not in leader_area or "lon" not in leader_area:
                return None, None
            leader_area["command_speed_mps"] = group.command_speed_mps
            group.follower_command_versions[platform.name] = group.leader_command_version
            return self._build_recon_message(
                platform,
                leader_area,
                formation_index=self.formation_index(group, platform),
                formation_spacing_m=self.formation_spacing(group, platform.name),
                leader=leader,
            ), None

        east_norm, north_norm, altitude_norm = [
            min(1.0, max(-1.0, value)) for value in values
        ]
        # One action selects a point in the one-step reachable disc.  Inputs
        # outside the unit disc are projected onto its boundary, while the
        # magnitude inside the disc selects the travelled fraction.
        magnitude = math.hypot(east_norm, north_norm)
        if magnitude > 1.0:
            east_norm /= magnitude
            north_norm /= magnitude
        lat, lon = self._offset_lat_lon(
            float(platform.lat), float(platform.lon),
            north_norm * self.horizontal_step_m,
            east_norm * self.horizontal_step_m,
        )
        command_altitude = min(
            self.max_altitude_m,
            max(
                self.min_altitude_m,
                float(platform.alt) + altitude_norm * self.altitude_step_m,
            ),
        )
        command_window = {
            "lat": lat,
            "lon": lon,
            "alt": command_altitude,
            "width_m": 10000.0,
            "height_m": 10000.0,
            "duration_sec": self.decision_sim_seconds,
            "command_speed_mps": self.max_speed_mps,
        }
        self.mark_group_task(group, "RECON")
        group.last_continuous_action = (east_norm, north_norm, altitude_norm)
        group.leader_target_area = dict(command_window)
        group.leader_command_version += 1
        return self._build_recon_message(platform, command_window), None
    def create_path_action_messages(
        self,
        group_id: str,
        action_id: int,
        threat_by_platform: Optional[Dict[str, dict]] = None,
    ) -> Tuple[List[dict], Optional[str]]:
        group = self.active_groups.get(group_id)
        if not group:
            return [], "recon_group_not_found"
        if action_id not in self.action_specs:
            return [], "invalid_recon_path_action"

        leader = self.ensure_group_leader(group)
        if leader is None:
            return [], "recon_group_has_no_surviving_aircraft"
        messages = []
        requested_action = self.action_specs[action_id]
        spacing_action = bool(requested_action.get("formation_spacing_delta_m"))
        hold_id = self.action_id("HOLD")
        leader_action_id = hold_id if spacing_action else action_id
        follower_action_id = action_id if spacing_action else hold_id
        leader_message, error = self.create_aircraft_path_action_message(
            group_id, leader.name, leader_action_id, threat=(threat_by_platform or {}).get(leader.name)
        )
        if error:
            return [], error
        if leader_message:
            messages.append(leader_message)
        for platform in group.platforms:
            if not platform.alive or platform.name == leader.name:
                continue
            message, error = self.create_aircraft_path_action_message(
                group_id, platform.name, follower_action_id, threat=(threat_by_platform or {}).get(platform.name)
            )
            if error:
                return [], error
            if message:
                messages.append(message)
        return messages, None

    def create_aircraft_path_action_message(
        self,
        group_id: str,
        aircraft_name: str,
        action_id: int,
        threat: Optional[dict] = None,
    ) -> Tuple[Optional[dict], Optional[str]]:
        group = self.active_groups.get(group_id)
        if not group:
            return None, "recon_group_not_found"
        if action_id not in self.action_specs:
            return None, "invalid_recon_path_action"

        platform = next((p for p in group.platforms if p.name == aircraft_name), None)
        if platform is None:
            return None, "recon_aircraft_not_in_group"

        leader = self.ensure_group_leader(group)
        if leader is None:
            return None, "recon_group_has_no_surviving_aircraft"
        action = self.action_specs[action_id]
        is_leader = platform.name == leader.name
        if is_leader:
            if action.get("formation_spacing_delta_m"):
                return None, "leader_cannot_adjust_formation_spacing"
            group.last_action_id = action_id
            if action.get("name") == "HOLD":
                return None, None
            if action.get("afsim_task") == "RETREAT":
                self.mark_group_task(group, "RETREAT")
                group.leader_action_id = action_id
                group.leader_command_version += 1
                group.leader_target_area = {
                    "afsim_task": "RETREAT",
                    "command_speed_mps": group.command_speed_mps,
                }
                return self._build_retreat_message(
                    platform, group.command_speed_mps
                ), None
            if "speed_delta_mps" in action:
                area = dict(group.leader_target_area)
                group.command_speed_mps = self._clamp_command_speed(
                    group.command_speed_mps + float(action.get("speed_delta_mps", 0.0))
                )
                group.leader_command_version += 1
                if area.get("afsim_task") == "RETREAT":
                    area["command_speed_mps"] = group.command_speed_mps
                    group.leader_target_area = area
                    return self._build_retreat_message(platform, group.command_speed_mps), None
                if "lat" not in area or "lon" not in area:
                    area = self._area_for_aircraft_action(group.area, {"name": "HOLD"}, platform, threat)
                area["command_speed_mps"] = group.command_speed_mps
                area["alt"] = group.command_altitude_m
                group.leader_target_area = area
                return self._build_recon_message(platform, area), None
            self.mark_group_task(group, "RECON")
            group.leader_action_id = action_id
            group.leader_command_version += 1
            if "altitude_delta_m" in action:
                group.command_altitude_m = min(
                    self.max_altitude_m,
                    max(
                        self.min_altitude_m,
                        group.command_altitude_m + float(action.get("altitude_delta_m", 0.0)),
                    ),
                )
            area = self._area_for_aircraft_action(group.area, action, platform, threat)
            area["command_speed_mps"] = group.command_speed_mps
            area["alt"] = group.command_altitude_m
            group.leader_target_area = dict(area)
            return self._build_recon_message(platform, area), None

        spacing_changed = bool(action.get("formation_spacing_delta_m"))
        if spacing_changed:
            self.adjust_formation_spacing(group, platform.name, float(action["formation_spacing_delta_m"]))
        elif action.get("name") != "HOLD":
            return None, "recon_follower_action_requires_hold_or_spacing"
        group.last_action_id = action_id
        applied_version = int(group.follower_command_versions.get(platform.name, 0))
        if not spacing_changed and applied_version >= group.leader_command_version:
            return None, None
        leader_action = self.action_specs.get(group.leader_action_id, {})
        if leader_action.get("afsim_task") == "RETREAT" or group.leader_target_area.get("afsim_task") == "RETREAT":
            group.follower_command_versions[platform.name] = group.leader_command_version
            return self._build_retreat_message(
                platform, group.command_speed_mps
            ), None
        leader_area = dict(group.leader_target_area or self._area_for_aircraft_action(group.area, {"name": "HOLD"}, leader, threat))
        leader_area["command_speed_mps"] = group.command_speed_mps
        formation_index = self.formation_index(group, platform)
        spacing_m = self.formation_spacing(group, platform.name)
        group.follower_command_versions[platform.name] = group.leader_command_version
        return self._build_recon_message(
            platform,
            leader_area,
            formation_index=formation_index,
            formation_spacing_m=spacing_m,
            leader=leader,
        ), None

    @staticmethod
    def mark_group_task(group: ReconGroup, task: str):
        for member in group.platforms:
            if not member.alive:
                continue
            member.task = task
            member.task_status = "ASSIGNED"
            member.task_assigned = True
            member.at_home = False

    def mark_assigned(self, platform):
        platform.task = "RECON"
        platform.task_status = "ASSIGNED"
        platform.task_assigned = True
        platform.at_home = False

    def mark_group_assigned(self, group: ReconGroup):
        for platform in group.platforms:
            self.mark_assigned(platform)

    def action_id(self, name: str) -> int:
        for action_id, spec in self.action_specs.items():
            if spec.get("name") == name:
                return int(action_id)
        return 0

    def ensure_group_leader(self, group: ReconGroup):
        current = next(
            (platform for platform in group.platforms if platform.name == group.leader_name and platform.alive),
            None,
        )
        if current is not None:
            return current
        candidates = [platform for platform in group.platforms if platform.alive]
        if not candidates:
            group.leader_name = ""
            group.leader_target_area = {}
            return None
        elected = self._leader_rng.choice(candidates)
        group.platforms.remove(elected)
        group.platforms.insert(0, elected)
        group.leader_name = elected.name
        group.leader_action_id = self.action_id("HOLD")
        group.leader_command_version += 1
        group.formation_spacing_by_aircraft.pop(elected.name, None)
        group.follower_command_versions.pop(elected.name, None)
        for platform in group.platforms:
            if platform.alive and platform.name != elected.name:
                group.formation_spacing_by_aircraft.setdefault(platform.name, self.formation_spacing_m)
                group.follower_command_versions[platform.name] = 0
        return elected

    def is_group_leader(self, group: ReconGroup, platform) -> bool:
        leader = self.ensure_group_leader(group)
        return leader is not None and platform.name == leader.name

    def formation_index(self, group: ReconGroup, platform) -> int:
        leader = self.ensure_group_leader(group)
        if leader is None:
            return -1
        if platform.name == leader.name:
            return 0
        followers = [
            member for member in group.platforms
            if member.alive and member.name != leader.name
        ]
        return followers.index(platform) + 1 if platform in followers else -1

    def formation_spacing(self, group: ReconGroup, aircraft_name: str) -> float:
        return min(
            self.max_formation_spacing_m,
            max(
                self.min_formation_spacing_m,
                float(group.formation_spacing_by_aircraft.get(aircraft_name, self.formation_spacing_m)),
            ),
        )

    def adjust_formation_spacing(self, group: ReconGroup, aircraft_name: str, delta_m: float) -> float:
        updated = self.formation_spacing(group, aircraft_name) + float(delta_m)
        updated = min(self.max_formation_spacing_m, max(self.min_formation_spacing_m, updated))
        group.formation_spacing_by_aircraft[aircraft_name] = updated
        return updated


    def _build_recon_message(
        self,
        platform,
        area: dict,
        formation_index: int = 0,
        formation_spacing_m: Optional[float] = None,
        leader=None,
    ) -> dict:
        lat = area["lat"]
        lon = area["lon"]
        if formation_index > 0:
            rank = (formation_index + 1) // 2
            side = -1.0 if formation_index % 2 == 1 else 1.0
            spacing_m = self.formation_spacing_m if formation_spacing_m is None else float(formation_spacing_m)
            forward_north, forward_east = self._relative_north_east(
                float(getattr(leader, "lat", platform.lat)),
                float(getattr(leader, "lon", platform.lon)),
                lat,
                lon,
            )
            magnitude = math.hypot(forward_north, forward_east)
            if magnitude <= 1.0:
                heading_rad = math.radians(float(getattr(leader, "heading", 0.0)))
                forward_north, forward_east = math.cos(heading_rad), math.sin(heading_rad)
                magnitude = 1.0
            right_north = -forward_east / magnitude
            right_east = forward_north / magnitude
            lat, lon = self._offset_lat_lon(
                lat,
                lon,
                side * rank * spacing_m * right_north,
                side * rank * spacing_m * right_east,
            )
        return {
            "MsgType": "AssignTask",
            "PlatformId": platform.platform_id,
            "PlatformName": platform.name,
            "Task": "RECON",
            "SearchCenter": [lat, lon, area.get("alt", 9144.0)],
            "SearchWidthM": area.get("width_m", 0.0),
            "SearchHeightM": area.get("height_m", 0.0),
            "SearchDurationSec": area.get("duration_sec", 60.0),
            "CommandSpeedMps": min(
                self.max_speed_mps,
                max(
                    self.min_speed_mps,
                    float(area.get("command_speed_mps", self.default_speed_mps)),
                ),
            ),
        }

    @staticmethod
    def _build_retreat_message(platform, command_speed_mps=None) -> dict:
        message = {
            "MsgType": "AssignTask",
            "PlatformId": platform.platform_id,
            "PlatformName": platform.name,
            "Task": "RETREAT",
        }
        if command_speed_mps is not None:
            message["CommandSpeedMps"] = float(command_speed_mps)
        return message

    def _clamp_command_speed(self, speed_mps: float) -> float:
        return min(
            self.max_speed_mps,
            max(self.min_speed_mps, float(speed_mps)),
        )

    def _area_for_path_action(self, area: dict, action: dict, platform=None) -> dict:
        base_lat = float(getattr(platform, "lat", area["lat"]))
        base_lon = float(getattr(platform, "lon", area["lon"]))
        if action.get("toward_area"):
            target_lat = float(area["lat"])
            target_lon = float(area["lon"])
            north_m, east_m = self._relative_north_east(base_lat, base_lon, target_lat, target_lon)
            distance_m = math.sqrt(north_m * north_m + east_m * east_m)
            step_m = max(1000.0, float(action.get("move_step_m", 30000.0)))
            if distance_m <= step_m or distance_m <= 1.0:
                lat, lon = target_lat, target_lon
            else:
                scale = step_m / distance_m
                lat, lon = self._offset_lat_lon(base_lat, base_lon, north_m * scale, east_m * scale)
        else:
            north_m = float(action.get("move_north_m", action.get("north_m", 0.0)))
            east_m = float(action.get("move_east_m", action.get("east_m", 0.0)))
            if ("altitude_delta_m" in action or "speed_delta_mps" in action) and abs(north_m) + abs(east_m) <= 1.0:
                heading_rad = math.radians(float(getattr(platform, "heading", 0.0)))
                forward_m = float(action.get("forward_m", 5000.0))
                north_m = math.cos(heading_rad) * forward_m
                east_m = math.sin(heading_rad) * forward_m
            lat, lon = self._offset_lat_lon(base_lat, base_lon, north_m, east_m)
        updated = dict(area)
        updated["lat"] = lat
        updated["lon"] = lon
        current_alt = float(getattr(platform, "alt", 0.0))
        if "altitude_delta_m" in action:
            requested_alt = current_alt + float(action.get("altitude_delta_m", 0.0))
        else:
            requested_alt = current_alt
        updated["alt"] = min(self.max_altitude_m, max(self.min_altitude_m, requested_alt))
        # The caller injects the group's persistent command speed. Path actions
        # must never derive or overwrite it from instantaneous telemetry speed.
        updated["width_m"] = float(action.get("search_width_m", area.get("local_search_width_m", 10000.0)))
        updated["height_m"] = float(action.get("search_height_m", area.get("local_search_height_m", 10000.0)))
        updated["duration_sec"] = float(action.get("duration_sec", min(30.0, float(area.get("duration_sec", 60.0)))))
        return updated

    def _area_for_aircraft_action(self, area: dict, action: dict, platform, threat: Optional[dict]) -> dict:
        return self._area_for_path_action(area, action, platform)
    def _area_for_threat_action(self, area: dict, action: dict, platform, threat: dict) -> dict:
        radius = max(5000.0, float(area.get("radius_m", 15000.0)))
        threat_lat = float(threat["lat"])
        threat_lon = float(threat["lon"])
        north_m, east_m = self._relative_north_east(threat_lat, threat_lon, platform.lat, platform.lon)
        length = math.sqrt(north_m * north_m + east_m * east_m)
        if length < 1.0:
            return self._area_for_path_action(area, action)

        unit_north = north_m / length
        unit_east = east_m / length
        action_name = str(action.get("name", ""))
        if action_name == "EVADE_THREAT":
            move_m = max(12000.0, radius * 0.8)
            lat, lon = self._offset_lat_lon(platform.lat, platform.lon, unit_north * move_m, unit_east * move_m)
        else:
            move_m = max(10000.0, radius * 0.65)
            tangent_a = (-unit_east, unit_north)
            tangent_b = (unit_east, -unit_north)
            cand_a = self._offset_lat_lon(platform.lat, platform.lon, tangent_a[0] * move_m, tangent_a[1] * move_m)
            cand_b = self._offset_lat_lon(platform.lat, platform.lon, tangent_b[0] * move_m, tangent_b[1] * move_m)
            dist_a, _ = self._distance_and_bearing(cand_a[0], cand_a[1], area["lat"], area["lon"])
            dist_b, _ = self._distance_and_bearing(cand_b[0], cand_b[1], area["lat"], area["lon"])
            lat, lon = cand_a if dist_a <= dist_b else cand_b

        updated = dict(area)
        updated["lat"] = lat
        updated["lon"] = lon
        updated["width_m"] = 0.0
        updated["height_m"] = 0.0
        updated["duration_sec"] = min(30.0, float(area.get("duration_sec", 60.0)))
        return updated

    @staticmethod
    def _default_action_specs() -> Dict[int, dict]:
        return {
            0: {"id": 0, "name": "HOLD", "afsim_task": "RECON", "move_north_m": 0.0, "move_east_m": 0.0, "search_width_m": 10000.0, "search_height_m": 10000.0, "duration_sec": 30.0},
            1: {"id": 1, "name": "MOVE_NORTH", "afsim_task": "RECON", "move_north_m": 8000.0, "move_east_m": 0.0, "search_width_m": 10000.0, "search_height_m": 10000.0, "duration_sec": 30.0},
            2: {"id": 2, "name": "MOVE_SOUTH", "afsim_task": "RECON", "move_north_m": -8000.0, "move_east_m": 0.0, "search_width_m": 10000.0, "search_height_m": 10000.0, "duration_sec": 30.0},
            3: {"id": 3, "name": "MOVE_EAST", "afsim_task": "RECON", "move_north_m": 0.0, "move_east_m": 8000.0, "search_width_m": 10000.0, "search_height_m": 10000.0, "duration_sec": 30.0},
            4: {"id": 4, "name": "MOVE_WEST", "afsim_task": "RECON", "move_north_m": 0.0, "move_east_m": -8000.0, "search_width_m": 10000.0, "search_height_m": 10000.0, "duration_sec": 30.0},
            5: {"id": 5, "name": "MOVE_NORTH_EAST", "afsim_task": "RECON", "move_north_m": 5657.0, "move_east_m": 5657.0, "search_width_m": 10000.0, "search_height_m": 10000.0, "duration_sec": 30.0},
            6: {"id": 6, "name": "MOVE_NORTH_WEST", "afsim_task": "RECON", "move_north_m": 5657.0, "move_east_m": -5657.0, "search_width_m": 10000.0, "search_height_m": 10000.0, "duration_sec": 30.0},
            7: {"id": 7, "name": "MOVE_SOUTH_EAST", "afsim_task": "RECON", "move_north_m": -5657.0, "move_east_m": 5657.0, "search_width_m": 10000.0, "search_height_m": 10000.0, "duration_sec": 30.0},
            8: {"id": 8, "name": "MOVE_SOUTH_WEST", "afsim_task": "RECON", "move_north_m": -5657.0, "move_east_m": -5657.0, "search_width_m": 10000.0, "search_height_m": 10000.0, "duration_sec": 30.0},
            9: {"id": 9, "name": "RETURN_HOME", "afsim_task": "RETREAT"},
            10: {"id": 10, "name": "MOVE_TOWARD_AREA", "afsim_task": "RECON", "toward_area": True, "move_step_m": 30000.0, "search_width_m": 16000.0, "search_height_m": 16000.0, "duration_sec": 30.0},
            11: {"id": 11, "name": "MOVE_UP", "afsim_task": "RECON", "altitude_delta_m": 1000.0, "forward_m": 5000.0, "search_width_m": 10000.0, "search_height_m": 10000.0, "duration_sec": 30.0},
            12: {"id": 12, "name": "MOVE_DOWN", "afsim_task": "RECON", "altitude_delta_m": -1000.0, "forward_m": 5000.0, "search_width_m": 10000.0, "search_height_m": 10000.0, "duration_sec": 30.0},
            13: {"id": 13, "name": "SPEED_UP", "afsim_task": "RECON", "speed_delta_mps": 20.0, "forward_m": 5000.0, "search_width_m": 10000.0, "search_height_m": 10000.0, "duration_sec": 30.0},
            14: {"id": 14, "name": "SPEED_DOWN", "afsim_task": "RECON", "speed_delta_mps": -20.0, "forward_m": 5000.0, "search_width_m": 10000.0, "search_height_m": 10000.0, "duration_sec": 30.0},
            15: {"id": 15, "name": "FORMATION_CLOSE", "afsim_task": "FORMATION", "formation_spacing_delta_m": -2500.0},
            16: {"id": 16, "name": "FORMATION_SPREAD", "afsim_task": "FORMATION", "formation_spacing_delta_m": 2500.0},
        }
    @staticmethod
    def _offset_lat_lon(lat: float, lon: float, north_m: float, east_m: float) -> Tuple[float, float]:
        lat_offset = north_m / 111320.0
        cos_lat = max(0.1, math.cos(math.radians(lat)))
        lon_offset = east_m / (111320.0 * cos_lat)
        return lat + lat_offset, lon + lon_offset

    @staticmethod
    def _relative_north_east(origin_lat: float, origin_lon: float, target_lat: float, target_lon: float) -> Tuple[float, float]:
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

    @staticmethod
    def _is_available(platform, is_busy: Callable[[object], bool]) -> bool:
        return (
            platform is not None
            and platform.role == "recon_aircraft"
            and platform.side == "red"
            and platform.alive
            and platform.platform_id is not None
            and not is_busy(platform)
        )
