#!/usr/bin/env python3
"""Move one or more AFSIM platforms without editing scenario text by hand."""

import argparse
import datetime as dt
import fnmatch
import json
import math
import re
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENARIO = (
    PROJECT_ROOT
    / "afsim_work"
    / "afsim-2.9.0-win64_bin"
    / "demos"
    / "air_to_air"
    / "scenarios"
    / "island_assault_min.txt"
)
DEFAULT_CONFIG = PROJECT_ROOT / "dppo" / "envs" / "afsim_units.json"

PLATFORM_START_RE = re.compile(r"^[ \t]*platform[ \t]+([^\s#]+)\b", re.MULTILINE)
POSITION_RE = re.compile(
    r"(\bposition\s+)(\S+)(\s+)(\S+)(?=\s+altitude\b)",
    re.IGNORECASE,
)
DMS_RE = re.compile(r"^(\d{1,3}):(\d{1,2}):(\d+(?:\.\d+)?)([NSEWnsew])$")
NUMBER_RE = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Move AFSIM platforms while preserving altitude, heading, and block layout."
    )
    parser.add_argument("platform", nargs="?", help="Platform name, for example blue_sam_1.")
    parser.add_argument("latitude", nargs="?", help="DMS (24:31:01.54n) or decimal latitude.")
    parser.add_argument("longitude", nargs="?", help="DMS (121:05:06.23e) or decimal longitude.")
    parser.add_argument("--scenario", default=str(DEFAULT_SCENARIO))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--moves-file", help="JSON list containing platform, lat/latitude, and lon/longitude.")
    parser.add_argument("--list", metavar="PATTERN", help="List matching platforms and positions without editing.")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing files.")
    parser.add_argument("--no-backup", action="store_true", help="Do not create timestamped backups.")
    parser.add_argument(
        "--no-sync-config",
        action="store_true",
        help="Do not synchronize matching ground_objectives in afsim_units.json.",
    )
    return parser.parse_args()


def platform_blocks(text):
    starts = list(PLATFORM_START_RE.finditer(text))
    blocks = []
    for index, match in enumerate(starts):
        start = match.start()
        next_start = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        candidate = text[start:next_start]
        end_match = re.search(r"\bend_platform\b", candidate)
        if end_match is None:
            raise ValueError("platform {0} has no end_platform".format(match.group(1)))
        end = start + end_match.end()
        blocks.append((match.group(1), start, end, text[start:end]))
    return blocks


def coordinate_to_dms(value, latitude):
    raw = str(value).strip()
    match = DMS_RE.fullmatch(raw)
    if match:
        degrees = int(match.group(1))
        minutes = int(match.group(2))
        seconds = float(match.group(3))
        hemisphere = match.group(4).lower()
        valid_hemispheres = ("n", "s") if latitude else ("e", "w")
        limit = 90 if latitude else 180
        if hemisphere not in valid_hemispheres:
            raise ValueError("invalid hemisphere for {0}: {1}".format("latitude" if latitude else "longitude", raw))
        if degrees > limit or minutes >= 60 or seconds >= 60.0:
            raise ValueError("invalid DMS coordinate: {0}".format(raw))
        return "{0:02d}:{1:02d}:{2:06.3f}{3}".format(degrees, minutes, seconds, hemisphere)

    decimal = float(raw)
    limit = 90.0 if latitude else 180.0
    if not math.isfinite(decimal) or not -limit <= decimal <= limit:
        raise ValueError("coordinate outside valid range: {0}".format(raw))
    hemisphere = ("n" if decimal >= 0.0 else "s") if latitude else ("e" if decimal >= 0.0 else "w")
    absolute = abs(decimal)
    degrees = int(absolute)
    minutes_full = (absolute - degrees) * 60.0
    minutes = int(minutes_full)
    seconds = (minutes_full - minutes) * 60.0
    if seconds >= 59.9995:
        seconds = 0.0
        minutes += 1
    if minutes >= 60:
        minutes = 0
        degrees += 1
    return "{0:02d}:{1:02d}:{2:06.3f}{3}".format(degrees, minutes, seconds, hemisphere)


def dms_to_decimal(value):
    match = DMS_RE.fullmatch(str(value).strip())
    if match is None:
        raise ValueError("expected normalized DMS coordinate: {0}".format(value))
    result = int(match.group(1)) + int(match.group(2)) / 60.0 + float(match.group(3)) / 3600.0
    return -result if match.group(4).lower() in ("s", "w") else result


def normalize_move(item):
    name = str(item.get("platform", item.get("name", ""))).strip()
    latitude = item.get("latitude", item.get("lat"))
    longitude = item.get("longitude", item.get("lon"))
    if not name or latitude is None or longitude is None:
        raise ValueError("each move requires platform, latitude/lat, and longitude/lon")
    return {
        "platform": name,
        "latitude": coordinate_to_dms(latitude, latitude=True),
        "longitude": coordinate_to_dms(longitude, latitude=False),
    }


def load_moves(args):
    if args.moves_file:
        data = json.loads(Path(args.moves_file).read_text(encoding="utf-8-sig"))
        if not isinstance(data, list):
            raise ValueError("moves file must contain a JSON list")
        if any(value is not None for value in (args.platform, args.latitude, args.longitude)):
            raise ValueError("do not combine positional move arguments with --moves-file")
        return [normalize_move(item) for item in data]
    if not all((args.platform, args.latitude, args.longitude)):
        raise ValueError("provide PLATFORM LATITUDE LONGITUDE or use --moves-file")
    return [normalize_move({"platform": args.platform, "lat": args.latitude, "lon": args.longitude})]


def list_platforms(text, pattern):
    matches = 0
    for name, _, _, block in platform_blocks(text):
        if not fnmatch.fnmatchcase(name, pattern):
            continue
        position = POSITION_RE.search(block)
        coordinates = "<no position>" if position is None else "{0} {1}".format(position.group(2), position.group(4))
        print("{0:24s} {1}".format(name, coordinates))
        matches += 1
    return matches


def apply_moves(text, moves):
    indexed = {}
    for name, start, end, block in platform_blocks(text):
        if name in indexed:
            raise ValueError("duplicate platform definition: {0}".format(name))
        indexed[name] = (start, end, block)

    replacements = []
    changes = []
    for move in moves:
        name = move["platform"]
        if name not in indexed:
            raise ValueError("platform not found: {0}".format(name))
        start, end, block = indexed[name]
        position = POSITION_RE.search(block)
        if position is None:
            raise ValueError("platform has no position before altitude: {0}".format(name))
        old_latitude, old_longitude = position.group(2), position.group(4)
        replacement = (
            position.group(1)
            + move["latitude"]
            + position.group(3)
            + move["longitude"]
        )
        updated_block = block[: position.start()] + replacement + block[position.end() :]
        replacements.append((start, end, updated_block))
        changes.append((name, old_latitude, old_longitude, move["latitude"], move["longitude"]))

    for start, end, replacement in sorted(replacements, reverse=True):
        text = text[:start] + replacement + text[end:]
    return text, changes


def sync_ground_objectives(config_text, moves):
    parsed = json.loads(config_text)
    objective_names = {
        str(item.get("name", "")) for item in parsed.get("ground_objectives", [])
    }
    synced = []
    for move in moves:
        name = move["platform"]
        if name not in objective_names:
            continue
        pattern = re.compile(
            r'("ground_objectives"\s*:\s*\[.*?"name"\s*:\s*"'
            + re.escape(name)
            + r'".*?"lat"\s*:\s*)'
            + NUMBER_RE
            + r'(\s*,\s*"lon"\s*:\s*)'
            + NUMBER_RE,
            re.DOTALL,
        )
        latitude = dms_to_decimal(move["latitude"])
        longitude = dms_to_decimal(move["longitude"])
        replacement = r"\g<1>{0:.10f}\g<2>{1:.10f}".format(latitude, longitude)
        config_text, count = pattern.subn(replacement, config_text, count=1)
        if count != 1:
            raise ValueError("could not synchronize ground objective: {0}".format(name))
        synced.append(name)
    json.loads(config_text)
    return config_text, synced


def backup(path, timestamp):
    target = path.with_name(path.name + ".bak_" + timestamp)
    shutil.copy2(path, target)
    return target


def main():
    args = parse_args()
    scenario_path = Path(args.scenario).resolve()
    text = scenario_path.read_text(encoding="utf-8")
    if args.list:
        count = list_platforms(text, args.list)
        print("matched", count)
        return 0 if count else 1

    moves = load_moves(args)
    updated, changes = apply_moves(text, moves)
    config_path = Path(args.config).resolve()
    config_updated = None
    synced = []
    if not args.no_sync_config and config_path.exists():
        config_text = config_path.read_text(encoding="utf-8")
        config_updated, synced = sync_ground_objectives(config_text, moves)

    for name, old_lat, old_lon, new_lat, new_lon in changes:
        print("{0}: {1} {2} -> {3} {4}".format(name, old_lat, old_lon, new_lat, new_lon))
    if synced:
        print("synced_ground_objectives", ",".join(synced))
    if args.dry_run:
        print("dry_run no files changed")
        return 0

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    if not args.no_backup:
        print("backup", backup(scenario_path, timestamp))
        if config_updated is not None and config_updated != config_path.read_text(encoding="utf-8"):
            print("backup", backup(config_path, timestamp))
    scenario_path.write_text(updated, encoding="utf-8", newline="")
    if config_updated is not None:
        config_path.write_text(config_updated, encoding="utf-8", newline="")
    print("updated", scenario_path)
    print("restart Warlock/Wizard to load the new initial positions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
