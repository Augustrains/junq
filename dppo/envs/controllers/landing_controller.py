import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple


@dataclass
class LandingAssignment:
    platform: object
    message: Optional[dict]
    action_name: str


@dataclass
class LandingGroup:
    group_id: str
    landing_zone: dict
    platforms: List[object]
    last_action_id: int = 0
    start_step: int = 0
    command_speed_mps: float = 6.0


class LandingController(object):
    """SMAC-style bottom-level controller for transport landing tasks."""

    def __init__(self, red_config, action_config=None):
        action_config = action_config or {}
        self.transports = list(red_config.get("transports", []))
        self.commandable_transports = list(red_config.get("commandable_transports", self.transports))
        self.active_groups: Dict[str, LandingGroup] = {}
        self.next_group_index = 1
        self.min_speed_mps = float(action_config.get("min_speed_mps", 2.0))
        self.max_speed_mps = float(action_config.get("max_speed_mps", 13.888889))
        self.default_speed_mps = float(action_config.get("default_speed_mps", 6.0))
        configured_actions = action_config.get("actions", [])
        self.action_specs = {int(action["id"]): dict(action) for action in configured_actions}
        if not self.action_specs:
            self.action_specs = self._default_action_specs()
        self.action_names = {action_id: spec["name"] for action_id, spec in self.action_specs.items()}
        self.continuous_action_dim = int(action_config.get("continuous_action_dim", 3))
        self.decision_sim_seconds = float(action_config.get("decision_sim_seconds", 72.0))
        self.horizontal_step_m = float(action_config.get(
            "horizontal_step_m", self.max_speed_mps * self.decision_sim_seconds
        ))
        if self.continuous_action_dim != 3:
            raise ValueError("landing continuous action dimension must be 3")
        self.horizontal_step_m = max(1.0, self.horizontal_step_m)

    def initialize_teams(self, platforms: Dict[str, object], landing_zone: dict, team_size: int = 3) -> List[LandingGroup]:
        """Create stable transport teams without a dispatch decision."""
        names = list(self._candidate_names())
        groups = []
        for offset in range(0, len(names), max(1, int(team_size))):
            members = [platforms[name] for name in names[offset:offset + max(1, int(team_size))] if name in platforms]
            if not members:
                continue
            group_id = "landing_team_{0}".format(len(groups) + 1)
            group = self.active_groups.get(group_id)
            if group is None:
                group = LandingGroup(
                    group_id=group_id,
                    landing_zone=dict(landing_zone or {}),
                    platforms=members,
                    command_speed_mps=self.default_speed_mps,
                )
                self.active_groups[group_id] = group
            groups.append(group)
        return groups

    def can_start(
        self,
        platforms: Dict[str, object],
        is_busy: Callable[[object], bool],
        is_available_transport: Optional[Callable[[object], bool]] = None,
    ) -> bool:
        return self.select_platform(platforms, is_busy, is_available_transport) is not None

    def select_platform(
        self,
        platforms: Dict[str, object],
        is_busy: Callable[[object], bool],
        is_available_transport: Optional[Callable[[object], bool]] = None,
    ):
        for name in self._candidate_names():
            platform = platforms.get(name)
            if self._is_available(platform, is_busy) and (
                is_available_transport is None or is_available_transport(platform)
            ):
                return platform
        return None

    def select_platforms(
        self,
        platforms: Dict[str, object],
        is_busy: Callable[[object], bool],
        count: int,
        is_available_transport: Optional[Callable[[object], bool]] = None,
    ):
        selected = []
        for name in self._candidate_names():
            platform = platforms.get(name)
            if self._is_available(platform, is_busy) and (
                is_available_transport is None or is_available_transport(platform)
            ):
                selected.append(platform)
                if len(selected) >= max(1, int(count)):
                    break
        return selected

    def start_group(
        self,
        landing_zone: dict,
        platforms: Dict[str, object],
        is_busy: Callable[[object], bool],
        group_size: int = 1,
        is_available_transport: Optional[Callable[[object], bool]] = None,
    ) -> Tuple[Optional[LandingGroup], List[LandingAssignment], Optional[str]]:
        landing_zone = landing_zone or {"name": "EXPLORE"}
        if not landing_zone:
            return None, [], "landing_zone_required"
        selected = self.select_platforms(
            platforms,
            is_busy,
            group_size,
            is_available_transport=is_available_transport,
        )
        if not selected:
            return None, [], "no_transport_available"

        group_id = "landing_group_{0}".format(self.next_group_index)
        self.next_group_index += 1
        group = LandingGroup(
            group_id=group_id,
            landing_zone=dict(landing_zone or {}),
            platforms=selected,
            command_speed_mps=self.default_speed_mps,
        )
        self.active_groups[group_id] = group

        assignments = [
            LandingAssignment(platform=platform, message=None, action_name="LANDING:{0}".format((landing_zone or {}).get("name", "EXPLORE")))
            for platform in selected
        ]
        return group, assignments, None

    def create_ship_action_message(
        self,
        group_id: str,
        ship_name: str,
        action_id: int,
    ) -> Tuple[Optional[dict], Optional[str]]:
        group = self.active_groups.get(group_id)
        if not group:
            return None, "landing_group_not_found"
        if action_id not in self.action_specs:
            return None, "invalid_landing_action"

        platform = next((p for p in group.platforms if p.name == ship_name), None)
        if platform is None:
            return None, "transport_not_in_group"

        group.last_action_id = action_id
        action = self.action_specs[action_id]
        task = str(action.get("afsim_task", ""))
        if task == "LANDING_HOLD":
            return self._build_simple_message(platform, "LANDING_HOLD"), None
        if task == "LANDING_UNLOAD":
            return self._build_unload_message(platform, group.landing_zone), None
        if task == "LANDING_RETURN_WAIT":
            return self._build_return_wait_message(
                platform, group.landing_zone, group.command_speed_mps
            ), None
        if task == "LANDING_MOVE_POINT":
            if "speed_delta_mps" in action:
                group.command_speed_mps = self._clamp_speed(
                    group.command_speed_mps + float(action.get("speed_delta_mps", 0.0))
                )
            return self._build_move_message(
                platform, action, group.landing_zone, group.command_speed_mps
            ), None
        return None, "unsupported_landing_action"

    def random_action(self) -> int:
        return random.choice(list(self.action_specs.keys()))
    def create_ship_continuous_action_message(self, group_id, ship_name, action):
        """Build one bounded surface-movement command from a Box(3) action.

        Components are east, north, and a reserved vertical component. The
        last component is retained for a uniform tactical action interface but
        has no physical effect for a surface vessel.
        """
        group = self.active_groups.get(group_id)
        if not group:
            return None, "landing_group_not_found"
        platform = next((p for p in group.platforms if p.name == ship_name), None)
        if platform is None:
            return None, "transport_not_in_group"
        if not platform.alive:
            return None, "transport_destroyed"
        try:
            values = [float(value) for value in action]
        except (TypeError, ValueError):
            return None, "invalid_landing_continuous_action"
        if len(values) != self.continuous_action_dim or not all(math.isfinite(value) for value in values):
            return None, "invalid_landing_continuous_action"
        east, north = [min(1.0, max(-1.0, value)) for value in values[:2]]
        magnitude = math.hypot(east, north)
        if magnitude > 1.0:
            east /= magnitude
            north /= magnitude
        action_spec = {
            "afsim_task": "LANDING_MOVE_POINT",
            "north_m": north * self.horizontal_step_m,
            "east_m": east * self.horizontal_step_m,
        }
        group.command_speed_mps = self.max_speed_mps
        message = self._build_move_message(
            platform, action_spec, group.landing_zone, group.command_speed_mps
        )
        message["ContinuousAction"] = [east, north, values[2]]
        return message, None


    def mark_assigned(self, platform):
        platform.task = "LANDING"
        platform.task_status = "ASSIGNED"
        platform.at_home = False

    def mark_group_assigned(self, group: LandingGroup):
        for platform in group.platforms:
            self.mark_assigned(platform)

    def _candidate_names(self) -> Iterable[str]:
        return self.commandable_transports or self.transports

    def _build_move_message(self, platform, action: dict, landing_zone: dict, command_speed_mps=None) -> dict:
        if action.get("to_landing_zone"):
            target_lat = float(landing_zone.get("navigation_lat", landing_zone.get("berth_lat", landing_zone.get("lat", platform.lat))))
            target_lon = float(landing_zone.get("navigation_lon", landing_zone.get("berth_lon", landing_zone.get("lon", platform.lon))))
            north_m, east_m = self._relative_north_east(platform.lat, platform.lon, target_lat, target_lon)
            distance_m = math.hypot(north_m, east_m)
            step_m = max(1000.0, float(action.get("move_step_m", 30000.0)))
            if distance_m <= step_m or distance_m <= 1.0:
                lat, lon = target_lat, target_lon
            else:
                scale = step_m / distance_m
                lat, lon = self._offset_lat_lon(platform.lat, platform.lon, north_m * scale, east_m * scale)
        else:
            north_m = float(action.get("north_m", 0.0))
            east_m = float(action.get("east_m", 0.0))
            if "speed_delta_mps" in action and abs(north_m) + abs(east_m) <= 1.0:
                heading_rad = math.radians(float(getattr(platform, "heading", 0.0)))
                forward_m = float(action.get("forward_m", 2000.0))
                north_m = math.cos(heading_rad) * forward_m
                east_m = math.sin(heading_rad) * forward_m
            berth_lat = float(landing_zone.get("navigation_lat", landing_zone.get("berth_lat", landing_zone.get("lat", platform.lat))))
            berth_lon = float(landing_zone.get("navigation_lon", landing_zone.get("berth_lon", landing_zone.get("lon", platform.lon))))
            distance_to_berth = self._distance_m(platform.lat, platform.lon, berth_lat, berth_lon)
            if landing_zone and distance_to_berth <= float(landing_zone.get("near_maneuver_distance_m", 2000.0)):
                action_step = math.hypot(north_m, east_m)
                near_step = float(landing_zone.get("near_step_m", 250.0))
                if action_step > near_step > 0.0:
                    scale = near_step / action_step
                    north_m *= scale
                    east_m *= scale
            lat, lon = self._offset_lat_lon(platform.lat, platform.lon, north_m, east_m)
        return {
            "MsgType": "AssignTask",
            "PlatformId": platform.platform_id,
            "PlatformName": platform.name,
            "Task": "LANDING_MOVE_POINT",
            "MovePosition": [lat, lon, 0.0],
            "CommandSpeedMps": self._clamp_speed(
                self.default_speed_mps if command_speed_mps is None else command_speed_mps
            ),
            "LandingZone": [
                float(landing_zone.get("lat", landing_zone.get("center_lat", lat))),
                float(landing_zone.get("lon", landing_zone.get("center_lon", lon))),
            ],
            "BerthPosition": [
                float(landing_zone.get("berth_lat", landing_zone.get("navigation_lat", landing_zone.get("lat", lat)))),
                float(landing_zone.get("berth_lon", landing_zone.get("navigation_lon", landing_zone.get("lon", lon)))),
            ],
            "ArrivalToleranceM": float(landing_zone.get("arrival_tolerance_m", 200.0)),
            "LandingRadiusM": float(landing_zone.get("unload_radius_m", landing_zone.get("radius_m", 350.0))),
        }

    @staticmethod
    def _build_unload_message(platform, landing_zone: dict) -> dict:
        return {
            "MsgType": "AssignTask",
            "PlatformId": platform.platform_id,
            "PlatformName": platform.name,
            "Task": "LANDING_UNLOAD",
            "LandingZone": [
                float(landing_zone.get("lat", landing_zone.get("center_lat", 0.0))),
                float(landing_zone.get("lon", landing_zone.get("center_lon", 0.0))),
            ],
            "BerthPosition": [
                float(landing_zone.get("berth_lat", landing_zone.get("navigation_lat", landing_zone.get("lat", 0.0)))),
                float(landing_zone.get("berth_lon", landing_zone.get("navigation_lon", landing_zone.get("lon", 0.0)))),
            ],
            "ArrivalToleranceM": float(landing_zone.get("arrival_tolerance_m", 200.0)),
            "LandingRadiusM": float(landing_zone.get("unload_radius_m", landing_zone.get("radius_m", 350.0))),
        }

    def _build_return_wait_message(
        self, platform, landing_zone: dict, command_speed_mps=None
    ) -> dict:
        lat = float(landing_zone.get("wait_lat", 25.0))
        lon = float(landing_zone.get("wait_lon", 120.0))
        return {
            "MsgType": "AssignTask",
            "PlatformId": platform.platform_id,
            "PlatformName": platform.name,
            "Task": "LANDING_RETURN_WAIT",
            "MovePosition": [lat, lon, 0.0],
            "CommandSpeedMps": self._clamp_speed(
                self.default_speed_mps if command_speed_mps is None else command_speed_mps
            ),
        }

    @staticmethod
    def _build_simple_message(platform, task: str) -> dict:
        return {
            "MsgType": "AssignTask",
            "PlatformId": platform.platform_id,
            "PlatformName": platform.name,
            "Task": task,
        }

    @staticmethod
    def _default_action_specs() -> Dict[int, dict]:
        return {
            0: {"id": 0, "name": "HOLD", "afsim_task": "LANDING_HOLD"},
            1: {"id": 1, "name": "MOVE_NORTH", "afsim_task": "LANDING_MOVE_POINT", "north_m": 1000.0, "east_m": 0.0},
            2: {"id": 2, "name": "MOVE_SOUTH", "afsim_task": "LANDING_MOVE_POINT", "north_m": -1000.0, "east_m": 0.0},
            3: {"id": 3, "name": "MOVE_EAST", "afsim_task": "LANDING_MOVE_POINT", "north_m": 0.0, "east_m": 1000.0},
            4: {"id": 4, "name": "MOVE_WEST", "afsim_task": "LANDING_MOVE_POINT", "north_m": 0.0, "east_m": -1000.0},
            5: {"id": 5, "name": "MOVE_NORTH_EAST", "afsim_task": "LANDING_MOVE_POINT", "north_m": 707.0, "east_m": 707.0},
            6: {"id": 6, "name": "MOVE_NORTH_WEST", "afsim_task": "LANDING_MOVE_POINT", "north_m": 707.0, "east_m": -707.0},
            7: {"id": 7, "name": "MOVE_SOUTH_EAST", "afsim_task": "LANDING_MOVE_POINT", "north_m": -707.0, "east_m": 707.0},
            8: {"id": 8, "name": "MOVE_SOUTH_WEST", "afsim_task": "LANDING_MOVE_POINT", "north_m": -707.0, "east_m": -707.0},
            9: {"id": 9, "name": "MOVE_TO_LANDING_ZONE", "afsim_task": "LANDING_MOVE_POINT", "to_landing_zone": True},
            10: {"id": 10, "name": "UNLOAD", "afsim_task": "LANDING_UNLOAD"},
            11: {"id": 11, "name": "RETURN_WAIT_AREA", "afsim_task": "LANDING_RETURN_WAIT"},
        }

    @staticmethod
    def _distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        mean_lat = math.radians((lat1 + lat2) / 2.0)
        north_m = (lat2 - lat1) * 111320.0
        east_m = (lon2 - lon1) * 111320.0 * math.cos(mean_lat)
        return math.hypot(north_m, east_m)

    @staticmethod
    def _offset_lat_lon(lat: float, lon: float, north_m: float, east_m: float) -> Tuple[float, float]:
        lat_offset = north_m / 111320.0
        cos_lat = max(0.1, math.cos(math.radians(lat)))
        lon_offset = east_m / (111320.0 * cos_lat)
        return lat + lat_offset, lon + lon_offset

    @staticmethod
    def _relative_north_east(lat1: float, lon1: float, lat2: float, lon2: float) -> Tuple[float, float]:
        mean_lat = math.radians((lat1 + lat2) / 2.0)
        return (lat2 - lat1) * 111320.0, (lon2 - lon1) * 111320.0 * math.cos(mean_lat)

    def _clamp_speed(self, speed_mps: float) -> float:
        return min(self.max_speed_mps, max(self.min_speed_mps, float(speed_mps)))

    @staticmethod
    def _is_available(platform, is_busy: Callable[[object], bool]) -> bool:
        return (
            platform is not None
            and platform.role == "transport"
            and platform.side == "red"
            and platform.alive
            and platform.platform_id is not None
            and not is_busy(platform)
        )
