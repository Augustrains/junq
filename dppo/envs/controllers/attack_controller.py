import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class AttackAssignment:
    platform: object
    message: Optional[dict]
    action_name: str


@dataclass
class AttackGroup:
    group_id: str
    target_name: str
    target: dict
    platforms: List[object]
    fixed_team_id: str = ""
    last_action_id: int = 0
    leader_action_id: int = 0
    start_step: int = 0
    fired_at_primary: Dict[str, bool] = field(default_factory=dict)
    fired_any_weapon: Dict[str, bool] = field(default_factory=dict)
    entered_mission_area: Dict[str, bool] = field(default_factory=dict)
    target_reservations: Dict[str, str] = field(default_factory=dict)
    mission_area: dict = field(default_factory=dict)
    leader_name: str = ""
    leader_target_position: List[float] = field(default_factory=list)
    leader_task: str = "ATTACK_HOLD"
    command_speed_mps: float = 100.0
    command_altitude_m: float = 3000.0
    formation_spacing_by_aircraft: Dict[str, float] = field(default_factory=dict)
    leader_command_version: int = 0
    follower_command_versions: Dict[str, int] = field(default_factory=dict)
    weapons_expended: Dict[str, set] = field(default_factory=dict)
    member_modes: Dict[str, str] = field(default_factory=dict)


class AttackController(object):
    """Bottom controller for always-on attack teams."""

    def __init__(self, red_config, action_config=None):
        action_config = action_config or {}
        self.attack_aircraft = list(red_config.get("attack_aircraft", []))
        self.commandable_attack_aircraft = list(
            red_config.get("commandable_attack_aircraft", self.attack_aircraft)
        )
        configured_teams = red_config.get("fixed_attack_teams", [])
        self.fixed_attack_teams = [
            tuple(str(name) for name in team if str(name))
            for team in configured_teams if team
        ]
        flattened = [name for team in self.fixed_attack_teams for name in team]
        if len(flattened) != len(set(flattened)):
            raise ValueError("fixed_attack_teams contains duplicate aircraft")
        unknown = sorted(
            set(flattened) - set(self.commandable_attack_aircraft)
        )
        if unknown:
            raise ValueError(
                "fixed_attack_teams contains non-commandable aircraft: {0}".format(
                    unknown
                )
            )
        self.active_groups: Dict[str, AttackGroup] = {}
        self.next_group_index = 1
        # Attack altitude and speed are fixed actuators, not policy actions.
        self.command_altitude_m = float(action_config.get("initial_task_altitude_m", 3000.0))
        self.min_altitude_m = self.command_altitude_m
        self.min_weapon_launch_altitude_m = self.command_altitude_m
        self.max_altitude_m = self.command_altitude_m
        self.default_altitude_m = self.command_altitude_m
        self.initial_task_altitude_m = self.command_altitude_m
        self.max_speed_mps = float(action_config.get("max_speed_mps", 500.0 / 3.6))
        self.min_speed_mps = self.max_speed_mps
        self.default_speed_mps = self.max_speed_mps
        self.formation_spacing_m = float(action_config.get("formation_spacing_m", 2500.0))
        self.min_formation_spacing_m = float(action_config.get("min_formation_spacing_m", self.formation_spacing_m))
        self.max_formation_spacing_m = float(action_config.get("max_formation_spacing_m", 15000.0))
        self.formation_spacing_step_m = float(action_config.get("formation_spacing_step_m", 2500.0))
        self._leader_rng = random.Random(int(action_config.get("leader_election_seed", 0)))
        configured_actions = action_config.get("actions", [])
        self.action_specs = {int(action["id"]): dict(action) for action in configured_actions}
        if not self.action_specs:
            self.action_specs = self._default_action_specs()
        self.path_actions = {action_id: spec["name"] for action_id, spec in self.action_specs.items()}

    def initialize_teams(self, platforms: Dict[str, object]) -> List[AttackGroup]:
        """Create stable coordination contexts without a target or area task."""
        groups = []
        for team_index, member_names in enumerate(self.fixed_attack_teams):
            group_id = "attack_team_{0}".format(team_index + 1)
            existing = self.active_groups.get(group_id)
            if existing is not None:
                groups.append(existing)
                continue
            members = [platforms[name] for name in member_names if name in platforms]
            if not members:
                continue
            group = self._new_group(
                group_id, "", {}, members, fixed_team_id=group_id
            )
            group.fired_at_primary = {platform.name: False for platform in members}
            group.fired_any_weapon = {platform.name: False for platform in members}
            self.active_groups[group_id] = group
            groups.append(group)
        return groups

    def select_fixed_team(
        self,
        platforms: Dict[str, object],
        is_busy: Callable[[object], bool],
        fixed_team_index: int,
    ):
        """Return one exact complete attack formation; never mix teams."""
        if not 0 <= int(fixed_team_index) < len(self.fixed_attack_teams):
            return None
        member_names = self.fixed_attack_teams[int(fixed_team_index)]
        members = [platforms.get(name) for name in member_names]
        if len(members) != len(member_names) or not all(
            self._is_available(platform, is_busy) for platform in members
        ):
            return None
        return "attack_fixed_team_{0}".format(int(fixed_team_index) + 1), members

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
        target_name: str,
        target: dict,
        platforms: Dict[str, object],
        is_busy: Callable[[object], bool],
        fixed_team_index: int,
    ) -> Tuple[Optional[AttackGroup], List[AttackAssignment], Optional[str]]:
        if not target_name or not target:
            return None, [], "attack_target_not_known"
        fixed_team = self.select_fixed_team(platforms, is_busy, fixed_team_index)
        if fixed_team is None:
            return None, [], "fixed_attack_team_unavailable"
        fixed_team_id, selected = fixed_team

        group_id = "attack_group_{0}".format(self.next_group_index)
        self.next_group_index += 1
        group = self._new_group(
            group_id, target_name, dict(target), selected,
            fixed_team_id=fixed_team_id,
        )
        group.fired_at_primary = {platform.name: False for platform in selected}
        group.fired_any_weapon = {platform.name: False for platform in selected}
        group.entered_mission_area = {platform.name: False for platform in selected}
        self.active_groups[group_id] = group

        assignments = [
            AttackAssignment(
                platform=platform,
                message=self._build_exact_move_message(
                    platform,
                    [platform.lat, platform.lon, group.command_altitude_m],
                    group.command_speed_mps,
                ),
                action_name="ATTACK:{0}".format(target_name),
            )
            for platform in selected
        ]
        return group, assignments, None

    def start_area_group(
        self,
        mission_area: dict,
        platforms: Dict[str, object],
        is_busy: Callable[[object], bool],
        fixed_team_index: int,
    ) -> Tuple[Optional[AttackGroup], List[AttackAssignment], Optional[str]]:
        if not mission_area:
            return None, [], "attack_area_missing"
        area_name = str(mission_area.get("name", ""))
        if not area_name:
            return None, [], "attack_area_missing"
        fixed_team = self.select_fixed_team(platforms, is_busy, fixed_team_index)
        if fixed_team is None:
            return None, [], "fixed_attack_team_unavailable"
        fixed_team_id, selected = fixed_team

        center_lat = float(mission_area.get("center_lat", mission_area.get("lat", 0.0)))
        center_lon = float(mission_area.get("center_lon", mission_area.get("lon", 0.0)))
        area_target = {
            "name": "",
            "known": True,
            "alive": True,
            "lat": center_lat,
            "lon": center_lon,
            "alt": float(mission_area.get("attack_alt_m", self.default_altitude_m)),
            "type": "ATTACK_AREA",
        }
        group_id = "attack_group_{0}".format(self.next_group_index)
        self.next_group_index += 1
        group = self._new_group(
            group_id, "", area_target, selected,
            mission_area=dict(mission_area), fixed_team_id=fixed_team_id,
        )
        group.fired_at_primary = {platform.name: False for platform in selected}
        group.fired_any_weapon = {platform.name: False for platform in selected}
        group.entered_mission_area = {platform.name: False for platform in selected}
        self.active_groups[group_id] = group
        assignments = [
            AttackAssignment(
                platform=platform,
                message=self._build_exact_move_message(
                    platform,
                    [platform.lat, platform.lon, group.command_altitude_m],
                    group.command_speed_mps,
                ),
                action_name="ATTACK:{0}".format(area_name),
            )
            for platform in selected
        ]
        return group, assignments, None

    def _new_group(
        self, group_id, target_name, target, selected,
        mission_area=None, fixed_team_id="",
    ):
        command_speed = self.max_speed_mps
        return AttackGroup(
            group_id=group_id,
            target_name=target_name,
            target=dict(target),
            platforms=selected,
            fixed_team_id=str(fixed_team_id),
            mission_area=dict(mission_area or {}),
            leader_name=selected[0].name,
            leader_target_position=[
                float(selected[0].lat),
                float(selected[0].lon),
                self.initial_task_altitude_m,
            ],
            leader_task="ATTACK_MOVE_POINT",
            command_speed_mps=command_speed,
            command_altitude_m=self.initial_task_altitude_m,
            formation_spacing_by_aircraft={
                platform.name: self.formation_spacing_m for platform in selected[1:]
            },
            leader_command_version=1,
            follower_command_versions={
                platform.name: 1 for platform in selected[1:]
            },
            member_modes={
                platform.name: ("LEADER" if platform.name == selected[0].name else "FOLLOWING")
                for platform in selected
            },
        )

    def create_aircraft_action_message(
        self,
        group_id: str,
        aircraft_name: str,
        action_id: int,
        target_override_name: Optional[str] = None,
        target_override: Optional[dict] = None,
    ) -> Tuple[Optional[dict], Optional[str]]:
        group = self.active_groups.get(group_id)
        if not group:
            return None, "attack_group_not_found"
        if action_id not in self.action_specs:
            return None, "invalid_attack_action"

        platform = next((p for p in group.platforms if p.name == aircraft_name), None)
        if platform is None:
            return None, "attack_aircraft_not_in_group"
        leader = self.ensure_group_leader(group)
        if leader is None:
            return None, "attack_group_has_no_surviving_aircraft"

        action = self.action_specs[action_id]
        task = str(action.get("afsim_task", ""))
        is_leader = platform.name == leader.name
        is_target_action = task in ("ATTACK_TARGET_SLOT", "FIRE_AAM", "FIRE_AGM")

        if action.get("name") == "HOLD":
            group.last_action_id = action_id
            if is_leader:
                group.leader_action_id = action_id
                group.leader_task = "ATTACK_HOLD"
                group.leader_target_position = []
                group.leader_command_version += 1
                return self._build_hold_message(platform), None
            # Wingman HOLD returns the member to normal formation-follow mode.
            group.member_modes[platform.name] = "FOLLOWING"
            # Fall through to follower section (formation following).

        if is_target_action:
            target_name = target_override_name or group.target_name
            target = target_override or group.target
            if not target_name or not target:
                return None, "attack_target_slot_empty"
            weapon = self._compatible_weapon(target)
            fire_task = "FIRE_AAM" if weapon == "fox3" else "FIRE_AGM"
            if task in ("FIRE_AAM", "FIRE_AGM"):
                fire_task = task
                weapon = str(action.get("weapon", weapon))
            group.last_action_id = action_id
            if self._target_in_weapon_range(platform, target):
                # Prevent double-fire: each weapon type can only be used once
                # per sortie.  The carrier reload resets this set.
                expended = group.weapons_expended.setdefault(platform.name, set())
                if weapon in expended:
                    return None, None
                expended.add(weapon)
                if target_name == group.target_name:
                    group.fired_at_primary[platform.name] = True
                group.fired_any_weapon[platform.name] = True
                return self._build_fire_message(
                    platform, fire_task, target_name, target, weapon
                ), None

            # Not in weapon range. A follower deliberately leaves formation
            # and uses the same two-stage approach/fire behavior as its leader.
            if not is_leader:
                group.member_modes[platform.name] = "INDEPENDENT_ATTACK"
                target_lat = float(target.get("lat", platform.lat))
                target_lon = float(target.get("lon", platform.lon))
                message = self._build_exact_move_message(
                    platform,
                    [target_lat, target_lon, group.command_altitude_m],
                    group.command_speed_mps,
                )
                platform.task = "ATTACK"
                platform.task_status = "ASSIGNED"
                platform.task_assigned = True
                platform.at_home = False
                return message, None
            # Leader: always fly directly toward the target. The environment
            # applies the weapon-specific launch-distance gate.
            target_lat = float(target.get("lat", platform.lat))
            target_lon = float(target.get("lon", platform.lon))
            message = self._build_exact_move_message(
                platform,
                [target_lat, target_lon, group.command_altitude_m],
                group.command_speed_mps,
            )
            platform.task = "ATTACK"
            platform.task_status = "ASSIGNED"
            platform.task_assigned = True
            platform.at_home = False
            group.leader_action_id = action_id
            group.leader_task = "ATTACK_MOVE_POINT"
            group.leader_target_position = list(message["MovePosition"])
            group.leader_command_version += 1
            return message, None

        if task == "ATTACK_REJOIN_FORMATION":
            if is_leader:
                return None, "leader_cannot_return_to_formation"
            group.member_modes[platform.name] = "REJOINING"
            return self._build_rejoin_move_message(group, platform, leader), None

        if task == "RETREAT":
            platform.task = "RETREAT"
            platform.task_status = "ASSIGNED"
            platform.task_assigned = True
            platform.at_home = False
            group.last_action_id = action_id
            return self._build_retreat_message(
                platform, group.command_speed_mps
            ), None

        if not is_leader:
            if action.get("formation_spacing_delta_m"):
                self.adjust_formation_spacing(
                    group, platform.name,
                    float(action.get("formation_spacing_delta_m", 0.0)),
                )
            elif action.get("name") != "HOLD":
                return None, "attack_follower_action_requires_hold_spacing_or_fire"
            group.last_action_id = action_id
            applied = int(group.follower_command_versions.get(platform.name, 0))
            spacing_changed = bool(action.get("formation_spacing_delta_m"))
            if not spacing_changed and applied >= group.leader_command_version:
                return None, None
            group.follower_command_versions[platform.name] = group.leader_command_version
            if group.leader_task == "RETREAT":
                return self._build_retreat_message(
                    platform, group.command_speed_mps
                ), None
            if not group.leader_target_position:
                return None, None
            return self._build_follower_move_message(group, platform, leader), None

        if action.get("formation_spacing_delta_m"):
            return None, "leader_cannot_adjust_formation_spacing"
        group.last_action_id = action_id
        if "speed_delta_mps" in action:
            group.command_speed_mps = self._clamp_command_speed(
                group.command_speed_mps + float(action.get("speed_delta_mps", 0.0))
            )
            group.leader_command_version += 1
            if group.leader_task == "RETREAT":
                return self._build_retreat_message(
                    platform, group.command_speed_mps
                ), None
            if group.leader_target_position:
                return self._build_exact_move_message(
                    platform,
                    group.leader_target_position,
                    group.command_speed_mps,
                ), None
            message = self._build_move_message(
                platform, action, group.target,
                command_speed_mps=group.command_speed_mps,
                command_altitude_m=group.command_altitude_m,
            )
            group.leader_target_position = list(message["MovePosition"])
            group.leader_task = "ATTACK_MOVE_POINT"
            return message, None
        if task == "ATTACK_MOVE_POINT":
            self.mark_group_task(group, "ATTACK")
            group.leader_action_id = action_id
            group.leader_task = "ATTACK_MOVE_POINT"
            group.leader_command_version += 1
            if "altitude_delta_m" in action:
                group.command_altitude_m = min(
                    self.max_altitude_m,
                    max(
                        self.min_altitude_m,
                        group.command_altitude_m + float(action.get("altitude_delta_m", 0.0)),
                    ),
                )
            message = self._build_move_message(
                platform, action, group.target,
                command_speed_mps=group.command_speed_mps,
                command_altitude_m=group.command_altitude_m,
            )
            group.leader_target_position = list(message["MovePosition"])
            return message, None
        return None, "unsupported_attack_action"

    def mark_assigned(self, platform):
        platform.task = "ATTACK"
        platform.task_status = "ASSIGNED"
        platform.task_assigned = True
        platform.at_home = False

    def mark_group_assigned(self, group: AttackGroup):
        for platform in group.platforms:
            self.mark_assigned(platform)

    def has_fired_at_primary(self, group_id: str, aircraft_name: str) -> bool:
        group = self.active_groups.get(group_id)
        if not group:
            return False
        return bool(group.fired_at_primary.get(aircraft_name, False))

    def has_fired_any_weapon(self, group_id: str, aircraft_name: str) -> bool:
        group = self.active_groups.get(group_id)
        if not group:
            return False
        return bool(group.fired_any_weapon.get(aircraft_name, False))


    def _build_move_message(
        self,
        platform,
        action: dict,
        target: Optional[dict] = None,
        formation_index: int = 0,
        formation_size: int = 1,
        formation_reference=None,
        command_speed_mps: Optional[float] = None,
        command_altitude_m: Optional[float] = None,
    ) -> dict:
        if action.get("toward_target") and target:
            target_lat = float(target.get("lat", platform.lat))
            target_lon = float(target.get("lon", platform.lon))
            if formation_size > 1:
                target_lat, target_lon = self._formation_slot_target(
                    formation_reference or platform,
                    target_lat,
                    target_lon,
                    formation_index,
                    float(action.get("formation_spacing_m", 2500.0)),
                    float(action.get("formation_trail_m", 1500.0)),
                )
            north_m, east_m = self._relative_north_east(platform.lat, platform.lon, target_lat, target_lon)
            distance_m = math.sqrt(north_m * north_m + east_m * east_m)
            step_m = max(1000.0, float(action.get("move_step_m", 30000.0)))
            if distance_m <= step_m or distance_m <= 1.0:
                lat, lon = target_lat, target_lon
            else:
                scale = step_m / distance_m
                lat, lon = self._offset_lat_lon(platform.lat, platform.lon, north_m * scale, east_m * scale)
        else:
            north_m = float(action.get("north_m", 0.0))
            east_m = float(action.get("east_m", 0.0))
            if ("altitude_delta_m" in action or "speed_delta_mps" in action) and abs(north_m) + abs(east_m) <= 1.0:
                heading_rad = math.radians(float(getattr(platform, "heading", 0.0)))
                forward_m = float(action.get("forward_m", 6000.0))
                north_m = math.cos(heading_rad) * forward_m
                east_m = math.sin(heading_rad) * forward_m
            lat, lon = self._offset_lat_lon(platform.lat, platform.lon, north_m, east_m)
        current_alt = float(getattr(platform, "alt", 0.0))
        requested_alt = current_alt if command_altitude_m is None else float(command_altitude_m)
        move_alt = min(self.max_altitude_m, max(self.min_altitude_m, requested_alt))
        move_speed = self._clamp_command_speed(
            self.default_speed_mps if command_speed_mps is None else command_speed_mps
        )
        return {
            "MsgType": "AssignTask",
            "PlatformId": platform.platform_id,
            "PlatformName": platform.name,
            "Task": "ATTACK_MOVE_POINT",
            "MovePosition": [lat, lon, move_alt],
            "CommandSpeedMps": move_speed,
        }

    def _clamp_command_speed(self, speed_mps: float) -> float:
        return min(
            self.max_speed_mps,
            max(self.min_speed_mps, float(speed_mps)),
        )

    def _build_exact_move_message(self, platform, position, command_speed_mps):
        return {
            "MsgType": "AssignTask",
            "PlatformId": platform.platform_id,
            "PlatformName": platform.name,
            "Task": "ATTACK_MOVE_POINT",
            "MovePosition": [
                float(position[0]), float(position[1]), float(position[2])
            ],
            "CommandSpeedMps": self._clamp_command_speed(command_speed_mps),
        }

    def _build_follower_move_message(self, group, platform, leader):
        target = list(group.leader_target_position)
        formation_index = group.platforms.index(platform)
        spacing_m = self.formation_spacing(group, platform.name)
        lat, lon = self._formation_slot_target(
            leader,
            float(target[0]),
            float(target[1]),
            formation_index,
            spacing_m,
            1500.0,
        )
        return self._build_exact_move_message(
            platform,
            [lat, lon, float(target[2])],
            group.command_speed_mps,
        )

    def _build_rejoin_move_message(self, group, platform, leader):
        """Route one detached follower to its current formation slot."""
        if group.leader_target_position:
            return self._build_follower_move_message(group, platform, leader)
        formation_index = group.platforms.index(platform)
        spacing_m = self.formation_spacing(group, platform.name)
        lat, lon = self._formation_slot_target(
            leader, float(leader.lat), float(leader.lon), formation_index, spacing_m, 1500.0
        )
        return self._build_exact_move_message(
            platform, [lat, lon, float(leader.alt)], group.command_speed_mps
        )
    def ensure_group_leader(self, group: AttackGroup):
        current = next(
            (
                platform for platform in group.platforms
                if platform.name == group.leader_name and platform.alive
            ),
            None,
        )
        if current is not None:
            return current
        candidates = [platform for platform in group.platforms if platform.alive]
        if not candidates:
            group.leader_name = ""
            group.leader_target_position = []
            return None
        elected = self._leader_rng.choice(candidates)
        group.platforms.remove(elected)
        group.platforms.insert(0, elected)
        group.leader_name = elected.name
        group.member_modes[elected.name] = "LEADER"
        group.leader_action_id = self.action_id("HOLD")
        group.leader_command_version += 1
        group.formation_spacing_by_aircraft.pop(elected.name, None)
        group.follower_command_versions.pop(elected.name, None)
        for platform in group.platforms:
            if platform.alive and platform.name != elected.name:
                group.formation_spacing_by_aircraft.setdefault(
                    platform.name, self.formation_spacing_m
                )
                group.follower_command_versions[platform.name] = 0
        return elected

    def reset_expended_weapons(self, aircraft_name: str):
        """Clear expended weapons after rearming at the carrier."""
        for group in self.active_groups.values():
            group.weapons_expended.pop(aircraft_name, None)

    def action_id(self, name: str) -> int:
        for action_id, spec in self.action_specs.items():
            if spec.get("name") == name:
                return int(action_id)
        return 0

    def mark_group_task(self, group: AttackGroup, task: str):
        for member in group.platforms:
            if not member.alive:
                continue
            member.task = task
            member.task_status = "ASSIGNED"
            member.task_assigned = True
            member.at_home = False

    def formation_spacing(self, group: AttackGroup, aircraft_name: str) -> float:
        return float(
            group.formation_spacing_by_aircraft.get(
                aircraft_name, self.formation_spacing_m
            )
        )

    def adjust_formation_spacing(
        self, group: AttackGroup, aircraft_name: str, delta_m: float
    ) -> float:
        current = self.formation_spacing(group, aircraft_name)
        updated = min(
            self.max_formation_spacing_m,
            max(self.min_formation_spacing_m, current + float(delta_m)),
        )
        group.formation_spacing_by_aircraft[aircraft_name] = updated
        return updated

    @classmethod
    def _formation_slot_target(
        cls,
        reference_platform,
        target_lat: float,
        target_lon: float,
        formation_index: int,
        spacing_m: float,
        trail_m: float = 1500.0,
    ) -> Tuple[float, float]:
        if formation_index <= 0 or spacing_m <= 0.0:
            return target_lat, target_lon

        forward_north, forward_east = cls._relative_north_east(
            reference_platform.lat,
            reference_platform.lon,
            target_lat,
            target_lon,
        )
        magnitude = math.hypot(forward_north, forward_east)
        if magnitude <= 1.0:
            forward_north, forward_east, magnitude = 1.0, 0.0, 1.0

        right_north = -forward_east / magnitude
        right_east = forward_north / magnitude
        rank = (formation_index + 1) // 2
        side = -1.0 if formation_index % 2 == 1 else 1.0
        lateral_m = side * rank * spacing_m
        rear_m = max(0.0, trail_m) * rank
        forward_unit_north = forward_north / magnitude
        forward_unit_east = forward_east / magnitude
        return cls._offset_lat_lon(
            target_lat,
            target_lon,
            right_north * lateral_m - forward_unit_north * rear_m,
            right_east * lateral_m - forward_unit_east * rear_m,
        )

    @staticmethod
    def _build_fire_message(platform, task: str, target_name: str, target: dict, weapon: str) -> dict:
        return {
            "MsgType": "AssignTask",
            "PlatformId": platform.platform_id,
            "PlatformName": platform.name,
            "Task": task,
            "TargetName": target_name,
            "TargetPosition": [target.get("lat", 0.0), target.get("lon", 0.0), target.get("alt", 0.0)],
            "Weapon": weapon,
        }

    @staticmethod
    def _build_hold_message(platform) -> dict:
        return {
            "MsgType": "AssignTask",
            "PlatformId": platform.platform_id,
            "PlatformName": platform.name,
            "Task": "ATTACK_HOLD",
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

    def _target_in_weapon_range(self, platform, target: dict) -> bool:
        if "_in_weapon_range" in target:
            return bool(target.get("_in_weapon_range"))
        target_lat = float(target.get("lat", platform.lat))
        target_lon = float(target.get("lon", platform.lon))
        target_alt = float(target.get("alt", 0.0))
        north_m, east_m = self._relative_north_east(
            platform.lat, platform.lon, target_lat, target_lon
        )
        altitude_m = float(platform.alt) - target_alt
        distance_m = math.sqrt(
            north_m * north_m + east_m * east_m + altitude_m * altitude_m
        )
        return distance_m <= float(target.get("weapon_range_m", 60000.0))

    @staticmethod
    def _compatible_weapon(target: dict) -> str:
        target_type = str(target.get("type", "")).upper()
        if "AIR" in target_type or "FIGHTER" in target_type or "AIRCRAFT" in target_type:
            return "fox3"
        return "agm"

    @staticmethod
    def _default_action_specs() -> Dict[int, dict]:
        return {
            0: {"id": 0, "name": "HOLD", "afsim_task": "ATTACK_HOLD"},
            1: {"id": 1, "name": "RETURN_HOME", "afsim_task": "RETREAT"},
            2: {"id": 2, "name": "ATTACK_TARGET_1", "afsim_task": "ATTACK_TARGET_SLOT", "target_slot": 0},
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
    def _is_available(platform, is_busy: Callable[[object], bool]) -> bool:
        return (
            platform is not None
            and platform.role == "attack_aircraft"
            and platform.side == "red"
            and platform.alive
            and platform.platform_id is not None
            and not is_busy(platform)
        )









