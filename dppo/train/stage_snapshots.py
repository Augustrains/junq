"""Dynamic stage-boundary snapshots for the AFSIM curriculum.

Snapshots store the observable AFSIM world plus the Python-side state that is
not represented by platform position messages.  They are start-state data,
not replay data: transitions must always be recollected with the current
policy after a snapshot is restored.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
import uuid
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Mapping


SNAPSHOT_SCHEMA_VERSION = 2
SNAPSHOT_STAGES = ("landing", "ground")


def checkpoint_fingerprint(path: str | Path | None) -> str:
    if not path:
        return "unversioned"
    checkpoint = Path(path)
    if not checkpoint.is_file():
        return "missing:{0}".format(checkpoint)
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_boundary_reached(env, stage: str) -> tuple[bool, str]:
    stage = str(stage).lower()
    if stage == "landing":
        if not env._is_landing_window_open():
            return False, "landing_window_closed"
        if not any(values.get("has_army", False) for values in env.landing_cargo.values()):
            return False, "no_transport_cargo"
        if any(values.get("landed", False) for values in env.ground_status.values()):
            return False, "ground_already_landed"
        return True, "landing_window_open"
    if stage == "ground":
        landed = [
            name for name, values in env.ground_status.items()
            if values.get("landed", False) and not values.get("on_ship", False)
            and env.platforms.get(name) is not None and env.platforms[name].alive
        ]
        if not landed:
            return False, "no_live_landed_ground"
        return True, "ground_unloaded"
    raise ValueError("dynamic snapshots are only defined for {0}".format(SNAPSHOT_STAGES))


def snapshot_quiescence_issues(env) -> list[str]:
    issues = []
    pending_names = (
        "pending_attack_returns",
        "pending_attack_fire_commands",
        "pending_landing_unloads",
        "attack_target_reservations",
        "ground_target_reservations",
    )
    for name in pending_names:
        if getattr(env, name, None):
            issues.append("{0}_not_empty".format(name))
    for controller_name in ("recon_controller", "attack_controller", "landing_controller", "ground_controller"):
        controller = getattr(env, controller_name, None)
        if controller is not None and getattr(controller, "active_groups", None):
            issues.append("{0}_has_active_groups".format(controller_name))
    active_red = []
    for platform in env.platforms.values():
        if platform.side != "red" or not platform.alive:
            continue
        task = str(platform.task or "").upper()
        status = str(platform.task_status or "").upper()
        if task not in ("", "PARKED", "HOLD", "GROUND_HOLD", "ATTACK_HOLD"):
            active_red.append(platform.name)
        elif status not in ("", "IDLE", "ACK"):
            active_red.append(platform.name)
    if active_red:
        issues.append("active_red_platforms:{0}".format(",".join(sorted(active_red))))
    return issues


def capture_stage_snapshot(
    env,
    stage: str,
    policy_checkpoint: str | Path | None = None,
    source_scenario: str = "scenarios/island_assault_min.txt",
    require_quiescent: bool = True,
    tags: Iterable[str] = (),
) -> dict:
    reached, reason = stage_boundary_reached(env, stage)
    if not reached:
        raise RuntimeError("stage boundary not reached: {0}".format(reason))
    issues = snapshot_quiescence_issues(env)
    if require_quiescent and issues:
        raise RuntimeError("stage boundary is not quiescent: {0}".format("; ".join(issues)))

    snapshot_id = "{0}-{1}-{2}".format(
        stage, time.strftime("%Y%m%dT%H%M%S", time.gmtime()), uuid.uuid4().hex[:8]
    )
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "stage": str(stage).lower(),
        "created_unix": time.time(),
        "source_scenario": str(source_scenario),
        "source_episode": int(getattr(env, "episode_id", 0)),
        "source_step": int(getattr(env, "step_count", 0)),
        "source_sim_time": float(env._current_sim_time()),
        "policy_checkpoint": str(policy_checkpoint or ""),
        "policy_fingerprint": checkpoint_fingerprint(policy_checkpoint),
        "boundary_reason": reason,
        "quiescence_issues": issues,
        "tags": list(tags),
        "platforms": {name: asdict(platform) for name, platform in sorted(env.platforms.items())},
        "python_state": {
            "detected_targets": dict(env.detected_targets),
            "enemy_track_memory": deepcopy(env.enemy_track_memory),
            "ground_detected_targets": dict(env.ground_detected_targets),
            "attack_ammo": dict(env.attack_ammo),
            "carrier_ammo_stock": dict(env.carrier_ammo_stock),
            "attack_target_reservations": dict(env.attack_target_reservations),
            "landing_cargo": dict(env.landing_cargo),
            "ground_status": dict(env.ground_status),
            "ground_ammo": dict(env.ground_ammo),
            "task_flags": dict(env.task_flags),
        },
    }


def validate_snapshot(snapshot: Mapping[str, object], expected_stage: str | None = None) -> list[str]:
    errors = []
    if int(snapshot.get("schema_version", -1)) not in (1, SNAPSHOT_SCHEMA_VERSION):
        errors.append("unsupported_schema_version")
    stage = str(snapshot.get("stage", ""))
    if stage not in SNAPSHOT_STAGES:
        errors.append("invalid_stage")
    if expected_stage and stage != str(expected_stage):
        errors.append("stage_mismatch")
    platforms = snapshot.get("platforms")
    if not isinstance(platforms, Mapping) or not platforms:
        errors.append("platforms_missing")
    else:
        for name, platform in platforms.items():
            if not isinstance(platform, Mapping):
                errors.append("invalid_platform:{0}".format(name))
                continue
            if bool(platform.get("alive", True)):
                lat = float(platform.get("lat", 0.0))
                lon = float(platform.get("lon", 0.0))
                if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                    errors.append("invalid_position:{0}".format(name))
    state = snapshot.get("python_state")
    if not isinstance(state, Mapping):
        errors.append("python_state_missing")
    if snapshot.get("quiescence_issues"):
        errors.append("snapshot_not_quiescent")
    return errors


class StageSnapshotPool:
    def __init__(self, root: str | Path, stage: str, max_size: int = 50, seed: int = 0):
        if stage not in SNAPSHOT_STAGES:
            raise ValueError("snapshot pool stage must be one of {0}".format(SNAPSHOT_STAGES))
        self.root = Path(root)
        self.stage = stage
        self.max_size = max(1, int(max_size))
        self.rng = random.Random(seed)
        self.stage_dir = self.root / stage

    def paths(self) -> list[Path]:
        if not self.stage_dir.exists():
            return []
        return sorted(self.stage_dir.glob("*.json"), key=lambda path: path.stat().st_mtime)

    def add(self, snapshot: Mapping[str, object]) -> Path:
        errors = validate_snapshot(snapshot, self.stage)
        if errors:
            raise ValueError("invalid stage snapshot: {0}".format(", ".join(errors)))
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        path = self.stage_dir / "{0}.json".format(snapshot["snapshot_id"])
        path.write_text(json.dumps(snapshot, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
        paths = self.paths()
        for stale in paths[:-self.max_size]:
            stale.unlink()
        return path

    def load(self, path: str | Path) -> dict:
        snapshot = json.loads(Path(path).read_text(encoding="utf-8"))
        errors = validate_snapshot(snapshot, self.stage)
        if errors:
            raise ValueError("invalid stage snapshot {0}: {1}".format(path, ", ".join(errors)))
        return snapshot

    def sample(self, current_policy_fingerprint: str = "", max_policy_age: int = 0) -> tuple[Path, dict]:
        del max_policy_age  # Reserved for a future ordered policy-version registry.
        candidates = self.paths()
        if current_policy_fingerprint:
            exact = []
            for path in candidates:
                snapshot = self.load(path)
                if snapshot.get("policy_fingerprint") == current_policy_fingerprint:
                    exact.append(path)
            if exact:
                candidates = exact
        if not candidates:
            raise RuntimeError("snapshot pool is empty for stage {0}".format(self.stage))
        path = self.rng.choice(candidates)
        return path, self.load(path)
