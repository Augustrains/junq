"""Verify blue units patrol and observe without firing in the live scenario."""

import argparse
import json
import sys

from envs.afsim_env import AFSIMIslandEnv
from test.diagnostics.task_pipeline_diagnostics import parse_address


def snapshot(env, names):
    result = {}
    for name in names:
        platform = env.platforms[name]
        result[name] = {
            "alive": bool(platform.alive),
            "hp": float(platform.current_hp),
            "lat": float(platform.lat),
            "lon": float(platform.lon),
            "alt": float(platform.alt),
            "aam": int(env.attack_ammo.get(name, {}).get("fox3", 0)),
            "agm": int(env.attack_ammo.get(name, {}).get("agm", 0)),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="envs/afsim_units.json")
    parser.add_argument("--local-address", default="0.0.0.0:50050")
    parser.add_argument("--observe-seconds", type=float, default=12.0)
    args = parser.parse_args()

    names = ["blue_attack_1", "blue_attack_2", "red_recon_1", "red_attack_1"]
    env = AFSIMIslandEnv(
        config_path=args.config,
        bind=True,
        auto_start_warlock=False,
        local_address=parse_address(args.local_address),
    )
    try:
        ready = env.wait_for_platforms(names, timeout=60)
        print("live_ready", ready)
        if not ready:
            return 2
        before = snapshot(env, names)
        env._drain_messages(timeout=args.observe_seconds)
        after = snapshot(env, names)
        moved = {}
        for name in ("blue_attack_1", "blue_attack_2"):
            distance, _ = env._slant_distance_and_bearing(
                before[name]["lat"], before[name]["lon"], before[name]["alt"],
                after[name]["lat"], after[name]["lon"], after[name]["alt"],
            )
            moved[name] = round(float(distance), 1)
        print("before", json.dumps(before, sort_keys=True))
        print("after", json.dumps(after, sort_keys=True))
        print("blue_moved_m", json.dumps(moved, sort_keys=True))
        no_ammo_use = all(
            before[name][weapon] == after[name][weapon]
            for name in ("blue_attack_1", "blue_attack_2")
            for weapon in ("aam", "agm")
        )
        no_red_damage = all(
            before[name]["hp"] == after[name]["hp"]
            for name in ("red_recon_1", "red_attack_1")
        )
        blue_moved = any(value > 100.0 for value in moved.values())
        print("result", json.dumps({
            "blue_moved": blue_moved,
            "no_blue_ammo_use": no_ammo_use,
            "no_red_damage": no_red_damage,
        }, sort_keys=True))
        return 0 if blue_moved and no_ammo_use and no_red_damage else 3
    finally:
        env.close()


if __name__ == "__main__":
    sys.exit(main())
