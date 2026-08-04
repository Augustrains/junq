import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple


@dataclass
class GroundAssignment:
    platform: object
    message: Optional[dict]
    action_name: str


@dataclass
class GroundGroup:
    group_id: str
    objective: dict
    platforms: List[object]
    last_action_id: int = 0
    start_step: int = 0


class GroundController(object):
    """SMAC-style bottom-level controller for landed ground forces."""

    def __init__(self, red_config, action_config=None):
        self.ground_forces = list(red_config.get("ground_forces", []))
        self.commandable_ground_forces = list(red_config.get("commandable_ground_forces", self.ground_forces))
        self.active_groups: Dict[str, GroundGroup] = {}
        self.next_group_index = 1
        configured_actions = (action_config or {}).get("actions", [])
        self.action_specs = {int(action["id"]): dict(action) for action in configured_actions}
        if not self.action_specs:
            self.action_specs = self._default_action_specs()
    def initialize_teams(self, platforms: Dict[str, object], objective: dict, team_size: int = 3) -> List[GroundGroup]:
        """Create stable ground teams without an activation decision."""
        names = list(self._candidate_names())
        groups = []
        for offset in range(0, len(names), max(1, int(team_size))):
            members = [platforms[name] for name in names[offset:offset + max(1, int(team_size))] if name in platforms]
            if not members:
                continue
            group_id = "ground_team_{0}".format(len(groups) + 1)
            group = self.active_groups.get(group_id)
            if group is None:
                group = GroundGroup(
                    group_id=group_id,
                    objective=dict(objective or {}),
                    platforms=members,
                )
                self.active_groups[group_id] = group
            groups.append(group)
        return groups

    def can_start(self, platforms: Dict[str, object], is_busy: Callable[[object], bool], is_available_ground: Callable[[object], bool]) -> bool:
        return self.select_platform(platforms, is_busy, is_available_ground) is not None

    def select_platform(self, platforms: Dict[str, object], is_busy: Callable[[object], bool], is_available_ground: Callable[[object], bool]):
        for name in self._candidate_names():
            platform = platforms.get(name)
            if self._is_available(platform, is_busy, is_available_ground):
                return platform
        return None

    def select_platforms(self, platforms: Dict[str, object], is_busy: Callable[[object], bool], is_available_ground: Callable[[object], bool], count: int):
        selected = []
        for name in self._candidate_names():
            platform = platforms.get(name)
            if self._is_available(platform, is_busy, is_available_ground):
                selected.append(platform)
                if len(selected) >= max(1, int(count)):
                    break
        return selected

    def start_group(self, objective: dict, platforms: Dict[str, object], is_busy: Callable[[object], bool], is_available_ground: Callable[[object], bool], group_size: int = 3):
        if not objective:
            return None, [], "ground_objective_required"
        selected = self.select_platforms(platforms, is_busy, is_available_ground, group_size)
        if not selected:
            return None, [], "no_landed_ground_available"
        group_id = "ground_group_{0}".format(self.next_group_index)
        self.next_group_index += 1
        group = GroundGroup(group_id=group_id, objective=dict(objective), platforms=selected)
        self.active_groups[group_id] = group
        assignments = [GroundAssignment(platform=p, message=None, action_name="GROUND:{0}".format(objective.get("name", ""))) for p in selected]
        return group, assignments, None

    def create_ground_action_message(self, group_id: str, unit_name: str, action_id: int, target: Optional[dict] = None):
        group = self.active_groups.get(group_id)
        if not group:
            return None, "ground_group_not_found"
        if action_id not in self.action_specs:
            return None, "invalid_ground_action"
        platform = next((p for p in group.platforms if p.name == unit_name), None)
        if platform is None:
            return None, "ground_unit_not_in_group"
        group.last_action_id = action_id
        action = self.action_specs[action_id]
        task = str(action.get("afsim_task", ""))
        if task == "GROUND_HOLD":
            return None, None
        if task == "GROUND_CAPTURE":
            return self._build_capture_message(platform, group.objective), None
        if task == "GROUND_FIRE":
            if not target:
                return None, "ground_fire_target_required"
            return self._build_fire_message(platform, target), None
        if task == "GROUND_MOVE_POINT":
            return self._build_move_message(platform, action, group.objective), None
        return None, "unsupported_ground_action"

    def mark_group_assigned(self, group: GroundGroup):
        for platform in group.platforms:
            platform.task = "GROUND"
            platform.task_status = "ASSIGNED"
            platform.at_home = False

    def random_action(self) -> int:
        return random.choice(list(self.action_specs.keys()))

    def _candidate_names(self) -> Iterable[str]:
        return self.commandable_ground_forces or self.ground_forces

    def _build_move_message(self, platform, action: dict, objective: dict) -> dict:
        if action.get("to_objective"):
            target_lat = float(objective.get("lat", platform.lat))
            target_lon = float(objective.get("lon", platform.lon))
            north_m, east_m = self._relative_north_east(
                platform.lat, platform.lon, target_lat, target_lon
            )
            distance_m = math.hypot(north_m, east_m)
            step_m = max(100.0, float(action.get("move_step_m", 5000.0)))
            if distance_m <= step_m or distance_m <= 1.0:
                lat, lon = target_lat, target_lon
            else:
                scale = step_m / distance_m
                lat, lon = self._offset_lat_lon(
                    platform.lat, platform.lon, north_m * scale, east_m * scale
                )
        else:
            lat, lon = self._offset_lat_lon(platform.lat, platform.lon, float(action.get("north_m", 0.0)), float(action.get("east_m", 0.0)))
        return {"MsgType": "AssignTask", "PlatformId": platform.platform_id, "PlatformName": platform.name, "Task": "GROUND_MOVE_POINT", "MovePosition": [lat, lon, 0.0], "ObjectivePosition": [float(objective.get("lat", lat)), float(objective.get("lon", lon))], "ObjectiveRadiusM": float(objective.get("radius_m", 5000.0))}

    @staticmethod
    def _build_fire_message(platform, target: dict) -> dict:
        return {"MsgType": "AssignTask", "PlatformId": platform.platform_id, "PlatformName": platform.name, "Task": "GROUND_FIRE", "TargetName": target.get("name", ""), "TargetPosition": [target.get("lat", 0.0), target.get("lon", 0.0), target.get("alt", 0.0)]}

    @staticmethod
    def _build_capture_message(platform, objective: dict) -> dict:
        return {"MsgType": "AssignTask", "PlatformId": platform.platform_id, "PlatformName": platform.name, "Task": "GROUND_CAPTURE", "ObjectivePosition": [float(objective.get("lat", 0.0)), float(objective.get("lon", 0.0))], "ObjectiveRadiusM": float(objective.get("radius_m", 5000.0))}

    @staticmethod
    def _build_simple_message(platform, task: str) -> dict:
        return {"MsgType": "AssignTask", "PlatformId": platform.platform_id, "PlatformName": platform.name, "Task": task}

    @staticmethod
    def _default_action_specs() -> Dict[int, dict]:
        return {
            0: {"id": 0, "name": "HOLD", "afsim_task": "GROUND_HOLD"},
            1: {"id": 1, "name": "MOVE_NORTH", "afsim_task": "GROUND_MOVE_POINT", "north_m": 1000.0, "east_m": 0.0},
            2: {"id": 2, "name": "MOVE_SOUTH", "afsim_task": "GROUND_MOVE_POINT", "north_m": -1000.0, "east_m": 0.0},
            3: {"id": 3, "name": "MOVE_EAST", "afsim_task": "GROUND_MOVE_POINT", "north_m": 0.0, "east_m": 1000.0},
            4: {"id": 4, "name": "MOVE_WEST", "afsim_task": "GROUND_MOVE_POINT", "north_m": 0.0, "east_m": -1000.0},
            5: {"id": 5, "name": "MOVE_NORTH_EAST", "afsim_task": "GROUND_MOVE_POINT", "north_m": 707.0, "east_m": 707.0},
            6: {"id": 6, "name": "MOVE_NORTH_WEST", "afsim_task": "GROUND_MOVE_POINT", "north_m": 707.0, "east_m": -707.0},
            7: {"id": 7, "name": "MOVE_SOUTH_EAST", "afsim_task": "GROUND_MOVE_POINT", "north_m": -707.0, "east_m": 707.0},
            8: {"id": 8, "name": "MOVE_SOUTH_WEST", "afsim_task": "GROUND_MOVE_POINT", "north_m": -707.0, "east_m": -707.0},
            9: {"id": 9, "name": "MOVE_TOWARD_OBJECTIVE", "afsim_task": "GROUND_MOVE_POINT", "to_objective": True, "move_step_m": 5000.0},
            **{
                10 + slot: {
                    "id": 10 + slot,
                    "name": "ATTACK_TARGET_{0}".format(slot + 1),
                    "afsim_task": "GROUND_FIRE",
                    "target_slot": slot,
                }
                for slot in range(8)
            },
            18: {"id": 18, "name": "CAPTURE_OBJECTIVE", "afsim_task": "GROUND_CAPTURE"},
        }

    @staticmethod
    def _offset_lat_lon(lat: float, lon: float, north_m: float, east_m: float) -> Tuple[float, float]:
        lat_offset = north_m / 111320.0
        cos_lat = max(0.1, math.cos(math.radians(lat)))
        lon_offset = east_m / (111320.0 * cos_lat)
        return lat + lat_offset, lon + lon_offset

    @staticmethod
    def _relative_north_east(lat1: float, lon1: float, lat2: float, lon2: float) -> Tuple[float, float]:
        cos_lat = max(0.1, math.cos(math.radians(float(lat1))))
        return (
            (float(lat2) - float(lat1)) * 111320.0,
            (float(lon2) - float(lon1)) * 111320.0 * cos_lat,
        )

    @staticmethod
    def _is_available(platform, is_busy: Callable[[object], bool], is_available_ground: Callable[[object], bool]) -> bool:
        return platform is not None and platform.role == "ground_force" and platform.side == "red" and platform.alive and platform.platform_id is not None and not is_busy(platform) and is_available_ground(platform)

