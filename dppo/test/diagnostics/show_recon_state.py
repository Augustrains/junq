"""Interactive recon state viewer.

Usage:
  python show_recon_state.py              # synthetic data
  python show_recon_state.py --live       # connect to AFSIM + warlock

Commands:
  s           - show full state for all members
  u <name>    - show available actions for aircraft
  u <name> <east> <north> <alt> - apply continuous action to aircraft
  u <name1> <e1> <n1> <a1> <name2> <e2> <n2> <a2> ... - apply to multiple
  w           - advance one decision cycle (drain + show)
  q           - quit
"""

import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from envs.afsim_env import AFSIMIslandEnv
from envs.rl_interface import AFSIMRLInterface
from train.decision_timing import apply_bottom_decision_timing, resolve_bottom_decision_timing
from test.tools.state_field_diagnostic import create_synthetic_active_groups, set_synthetic_platforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--simulation-clock-rate", type=float, default=40.0)
    parser.add_argument("--bottom-decisions-per-hour", type=float, default=50.0)
    args = parser.parse_args()
    live = args.live

    env = AFSIMIslandEnv(bind=live, auto_start_warlock=live)
    apply_bottom_decision_timing(env, resolve_bottom_decision_timing(
        args.bottom_decisions_per_hour, args.simulation_clock_rate))
    try:
        env.reset()
        if live:
            _wait_for_platforms(env)
            group, members = _create_live_recon_group(env)
        else:
            set_synthetic_platforms(env)
            create_synthetic_active_groups(env)
            group = next(iter(env.recon_controller.active_groups.values()))
            members = [p.name for p in group.platforms]

        api = AFSIMRLInterface(env)
        spec = api.get_agent_specs()["recon"]
        print(f"Recon obs_dim={spec['obs_dim']}  action_dim={spec['action_dim']} (continuous: east, north, altitude)")
        print(f"Group: {group.group_id}  Members: {members}  Leader: {group.leader_name}")

        _show_help()
        while True:
            try:
                cmd = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not cmd:
                continue
            parts = cmd.split()
            op = parts[0].lower()

            if op == "q":
                break
            elif op == "s":
                state = api.get_agent_state("recon", group.group_id)
                _print_state(state, members)
            elif op == "w":
                env._drain_messages(timeout=env.decision_seconds)
                state = api.get_agent_state("recon", group.group_id)
                _print_key_fields(state, members)
            elif op == "u" and len(parts) >= 2:
                _user_action(env, api, group, parts, members, live=live)
            else:
                _show_help()
    finally:
        env.close()


def _user_action(env, api, group, parts, members, live=False):
    """Handle u commands: show actions or apply continuous actions.
    u <name>                    -> show available actions
    u <name> <e> <n> <a>        -> apply [east, north, altitude] to one aircraft
    u <n1> <e1> <n1> <a1> <n2> <e2> <n2> <a2> ... -> apply to multiple
    """
    if live:
        env._drain_messages(timeout=0.5)

    if len(parts) == 2:
        # Show available actions
        name = parts[1]
        state = api.get_agent_state("recon", group.group_id)
        if name not in state["aircraft"]:
            print(f"  {name}: not in group")
            return
        ac = state["aircraft"][name]
        mask = ac.get("action_mask", [])
        role = "LEADER" if ac["obs_by_name"].get("is_leader") == 1.0 else "WINGMAN"
        if not env.platforms.get(name) or not env.platforms[name].alive:
            print(f"  {name}: DEAD")
            return
        print(f"\n  {name} [{role}] continuous action: [east_norm, north_norm, altitude_norm]")
        print(f"  action_dim=3  range=[-1, 1]  horizontal_step=8000m  altitude_step=1000m")
        if all(v > 0.0 for v in mask):
            print(f"  all actions available (no mask)")
        else:
            print(f"  mask: {mask}")
        return

    # Apply continuous actions: u <name1> <e1> <n1> <a1> [<name2> <e2> <n2> <a2> ...]
    if (len(parts) - 1) % 4 != 0:
        print("  Usage: u <name> <east> <north> <alt> [<name2> <east2> <north2> <alt2> ...]")
        return

    num_aircraft = (len(parts) - 1) // 4
    for i in range(num_aircraft):
        idx = 1 + i * 4
        name = parts[idx]
        east = float(parts[idx + 1])
        north = float(parts[idx + 2])
        alt = float(parts[idx + 3])
        action = [east, north, alt]

        ok = env.apply_recon_aircraft_continuous_action(group.group_id, name, action)
        ac = api.get_agent_state("recon", group.group_id)["aircraft"].get(name, {})
        role = "LEADER" if ac.get("obs_by_name", {}).get("is_leader") == 1.0 else "WINGMAN"
        print(f"  [{name}] {role} -> [{east:+.2f}, {north:+.2f}, {alt:+.2f}] ok={ok}")

    if live:
        env._drain_messages(timeout=0.5)
    state = api.get_agent_state("recon", group.group_id)
    # Update members (leader might have changed)
    current = [p.name for p in group.platforms]
    _print_key_fields(state, current)


def _print_key_fields(state, members):
    print(f"\n{'Name':>18s} {'is_leader':>10s} {'hp':>6s} {'speed':>7s} {'alt':>7s} {'dist_ldr':>10s} {'return':>7s} {'staty':>6s}")
    print("-" * 80)
    for name in members:
        ac = state["aircraft"].get(name)
        if not ac:
            continue
        ob = ac["obs_by_name"]
        leader = "YES" if ob.get("is_leader") == 1.0 else "no"
        print(f"{name:>18s} {leader:>10s} {ob.get('hp_norm',0):>6.2f} {ob.get('speed_norm',0):>7.3f} {ob.get('alt_norm',0):>7.3f} {ob.get('distance_to_leader_norm',0):>10.3f} {ob.get('returning_to_carrier',0):>7.1f} {ob.get('is_stationary',0):>6.1f}")


def _print_state(state, members):
    for name in members:
        ac = state["aircraft"].get(name)
        if not ac:
            continue
        ob = ac["obs_by_name"]
        role = "LEADER" if ob.get("is_leader") == 1.0 else "WINGMAN"
        print(f"\n{'='*55}")
        print(f"  {name} [{role}]  hp={ob.get('hp_norm',0):.2f}  speed={ob.get('speed_norm',0):.3f}  alt={ob.get('alt_norm',0):.3f}")
        print(f"{'='*55}")

        fields = ac["fields"]
        for f in fields:
            v = ob.get(f, "??")
            if f.startswith("target_slot_") or f.startswith("friendly_"):
                if isinstance(v, float) and v == 0.0:
                    continue
                print(f"  {f:45s} = {v}")
            else:
                print(f"  {f:35s} = {v}")

    n_target = sum(1 for f in fields if f.startswith("target_slot_"))
    n_friendly = sum(1 for f in fields if f.startswith("friendly_"))
    n_own = len(fields) - n_target - n_friendly
    print(f"\n  Total: {len(fields)} fields  (own+team: {n_own}, friendly: {n_friendly}, target slots: {n_target})")


def _wait_for_platforms(env, timeout=60.0):
    start = time.time()
    while time.time() - start < timeout:
        env._drain_messages(timeout=1.0)
        recon = [p for p in env.platforms.values() if getattr(p, "role", "") == "recon_aircraft" and p.alive]
        valid = [p for p in recon if abs(p.lat) > 1.0 and abs(p.lon) > 1.0]
        elapsed = time.time() - start
        if len(valid) >= 6:
            print(f"Platforms ready ({elapsed:.1f}s): {len(valid)} recon with valid positions")
            return
        print(f"[{elapsed:.0f}s] Waiting... ({len(recon)} recon, {len(valid)} with valid position)")
    print(f"Timeout ({timeout}s)")


def _create_live_recon_group(env):
    teams = env.recon_controller.fixed_recon_teams
    if not teams:
        raise RuntimeError("No fixed recon teams configured")
    team_names = list(teams[0])
    print(f"Creating recon group from team: {team_names}")
    group = env.recon_controller.initialize_teams(env.platforms)
    if not group:
        raise RuntimeError("Failed to create recon group")
    g = group[0]
    return g, [p.name for p in g.platforms]


def _show_help():
    print("=" * 60)
    print("  s=show  u name=actions  u name e n a=apply")
    print("  u n1 e n a n2 e n a ... =apply multiple")
    print("  w=drain+show  q=quit")
    print("=" * 60)


if __name__ == "__main__":
    main()
