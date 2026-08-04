"""Interactive attack state viewer.

Usage:
  python show_attack_state.py              # synthetic data (fast, 371d structure check)
  python show_attack_state.py --live       # connect to AFSIM + warlock (real scenario)

Commands:
  s           - show full state for all members
  x           - network selects & applies actions (one step)
  u <name> <id> - manually apply action_id to aircraft
  u <name>    - show available actions for aircraft
  p           - print action masks for all members
  m <N>       - leader flies toward target slot N (move step)
  f <N>       - fire at target slot N (leader only)
  o <N>       - fire at target slot N (any member with ammo)
  r           - leader retreats (all members RTB)
  h           - leader holds
  d <name>    - damage aircraft (reduce HP)
  k <name>    - kill aircraft (set HP=0)
  a <fox3> <agm> - set ammo for ALL members (synthetic only)
  t <N> <lat> <lon> <alt> - inject known target at slot N
  c <path>    - load checkpoint
  q           - quit
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from envs.afsim_env import AFSIMIslandEnv
from envs.rl_interface import AFSIMRLInterface
from train.happo_trainer import HAPPOConfig, HAPPOTrainer
from train.decision_timing import apply_bottom_decision_timing, resolve_bottom_decision_timing
from test.tools.state_field_diagnostic import create_synthetic_active_groups, set_synthetic_platforms


def _latlon(env, lat_norm, lon_norm):
    cfg = env.config.get("scenario", {}).get("bounds", {})
    lat_min = float(cfg.get("lat_min", 23.5))
    lat_max = float(cfg.get("lat_max", 25.8))
    lon_min = float(cfg.get("lon_min", 118.8))
    lon_max = float(cfg.get("lon_max", 122.2))
    lat = lat_min + lat_norm * (lat_max - lat_min)
    lon = lon_min + lon_norm * (lon_max - lon_min)
    return lat, lon


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Connect to AFSIM warlock")
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
            group, members = _create_live_attack_group(env)
        else:
            set_synthetic_platforms(env)
            create_synthetic_active_groups(env)
            group = next(iter(env.attack_controller.active_groups.values()))
            members = [p.name for p in group.platforms]
            for i, name in enumerate(members):
                env.attack_ammo[name] = {"fox3": 1 if i == 0 else 0, "agm": 1 if i < 2 else 0}

        api = AFSIMRLInterface(env)
        # Reset ammo for all attack aircraft (fresh loadout every script start)
        for name in env.config.get("red", {}).get("attack_aircraft", []):
            env.attack_ammo[name] = {"fox3": 1, "agm": 1}
        # Build a fresh (untrained) HAPPO trainer for action selection.
        specs = api.get_agent_specs()
        attack_spec = specs["attack"]
        global_state_dim = specs.get("global", {}).get("obs_dim", 2048)
        trainer = HAPPOTrainer(
            {"attack": attack_spec},
            global_state_dim=global_state_dim,
            agent_types=("attack",),
            hidden_sizes=(64, 64),
            config=HAPPOConfig(update_epochs=1),
        )
        policy = trainer.entity_policies[members[0]]
        print(f"Network: obs_dim={attack_spec['obs_dim']}  action_dim={attack_spec['action_dim']}  hidden=(64,64)")
        print(f"Checkpoint: none loaded (untrained, random actions)")
        print(f"Group: {group.group_id}  Members: {members}  Leader: {group.leader_name}")
        if not live:
            ammo = {n: env.attack_ammo.get(n) for n in members}
            print(f"Ammo: {ammo}")

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
                state = api.get_agent_state("attack", group.group_id)
                _print_state(state, members)
            elif op == "x":
                _step_network(env, api, trainer, group, members, live=live)
            elif op == "u" and len(parts) >= 3 and len(parts) % 2 == 1:
                # u name1 id1 [name2 id2 ...] — apply multiple actions at once
                pairs = [(parts[i], int(parts[i + 1])) for i in range(1, len(parts), 2)]
                for name, action_id in pairs:
                    _user_action(env, api, group, name, action_id, live=live)
                env._drain_messages(timeout=2.0)
                state = api.get_agent_state("attack", group.group_id)
                _print_key_fields(state, [p.name for p in group.platforms])
            elif op == "u" and len(parts) >= 2:
                _user_action(env, api, group, parts[1], None, live=live)
            elif op == "w" or op == "":
                # Advance one decision cycle: drain + show state
                env._drain_messages(timeout=env.decision_seconds)
                state = api.get_agent_state("attack", group.group_id)
                group = env.attack_controller.active_groups.get(group.group_id)
                if group:
                    members = [p.name for p in group.platforms]
                _print_key_fields(state, members)
            elif op == "ww":
                # Long drain for interactive operations such as rearming.
                env._drain_messages(timeout=5.0)
                state = api.get_agent_state("attack", group.group_id)
                group = env.attack_controller.active_groups.get(group.group_id)
                if group:
                    members = [p.name for p in group.platforms]
                _print_key_fields(state, members)
            elif op == "d" and len(parts) >= 2:
                _show_distance(env, api, group, members, int(parts[1]))
            elif op == "c" and len(parts) >= 2:
                _load_checkpoint(trainer, policy, parts[1])
            elif op == "p":
                state = api.get_agent_state("attack", group.group_id)
                _print_masks(state)
            elif op == "m" and len(parts) >= 2:
                _move_leader(env, api, group, int(parts[1]))
            elif op == "f" and len(parts) >= 2:
                _fire(env, api, group, members[0], int(parts[1]))
            elif op == "o" and len(parts) >= 2:
                _opportunity_fire(env, api, group, members, int(parts[1]))
            elif op == "r":
                _retreat(env, api, group, members[0])
            elif op == "h":
                _hold(env, api, group, members[0])
            elif op == "v":
                _verify_mask(env, group)
            elif op == "d" and len(parts) >= 2:
                _damage(env, parts[1])
            elif op == "k" and len(parts) >= 2:
                _kill(env, api, group, parts[1])
            elif op == "a" and len(parts) >= 4:
                _set_ammo_all(env, members, int(parts[1]), int(parts[2]))
            elif op == "t" and len(parts) >= 5:
                _inject_target(env, int(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
            else:
                _show_help()
    finally:
        env.close()


def _user_action(env, api, group, name, action_id, live=False):
    """Manually apply an action to an aircraft (simulates network output)."""
    state = api.get_agent_state("attack", group.group_id)
    action_names = [a.get("name", "?") for a in state["action_table"]]

    if name not in state["aircraft"]:
        print(f"Unknown aircraft: {name}")
        return

    ac = state["aircraft"][name]
    mask = ac["action_mask"]

    if action_id is None:
        # Show available actions — drain first to get fresh state in live mode
        if live:
            env._drain_messages(timeout=0.5)
            state = api.get_agent_state("attack", group.group_id)
            ac = state["aircraft"].get(name)
            if not ac:
                print(f"  {name}: not in group")
                return
            mask = ac["action_mask"]
        if not env.platforms.get(name) or not env.platforms[name].alive:
            print(f"  {name}: DEAD (no actions)")
            return
        role = "LEADER" if ac["obs_by_name"].get("is_leader") == 1.0 else "WINGMAN"
        n_avail = sum(1 for v in mask if v > 0)
        print(f"\n  {name} [{role}]  {n_avail}/{len(mask)} actions available:")
        for i, (v, aname) in enumerate(zip(mask, action_names)):
            if v > 0:
                print(f"    id={i:2d}  {aname}")
        return

    if action_id < 0 or action_id >= len(mask):
        print(f"Invalid action_id {action_id} (0-{len(mask)-1})")
        return
    if mask[action_id] <= 0:
        action_name = action_names[action_id] if action_id < len(action_names) else "??"
        print(f"Action {action_id} ({action_name}) is MASKED (not available)")
        return

    # Apply
    if not env.platforms.get(name) or not env.platforms[name].alive:
        print(f"{name} is DEAD")
        return

    # Check what the controller will actually do before applying
    leader = env.attack_controller.ensure_group_leader(group)
    is_leader = leader is not None and name == leader.name
    ac = state["aircraft"][name]
    mask = ac["action_mask"]
    if action_id < 0 or action_id >= len(mask):
        print(f"  Invalid action_id {action_id}")
        return False
    if mask[action_id] <= 0:
        action_name = action_names[action_id] if action_id < len(action_names) else "??"
        print(f"  {name}: {action_name} (id={action_id}) is MASKED")
        return False

    # Show range info for target actions
    if action_names[action_id].startswith("ATTACK_TARGET_"):
        slot = int(action_names[action_id].split("_")[-1])
        tgt_slots = state["team"].get("target_slots", [])
        if slot <= len(tgt_slots):
            tgt = tgt_slots[slot - 1]
            p = env.platforms.get(name)
            if p:
                t_lat = float(tgt.get("lat", 0))
                t_lon = float(tgt.get("lon", 0))
                t_alt = float(tgt.get("alt", 0))
                # Use actual platform position if available
                actual = env.platforms.get(tgt.get("name", ""))
                if actual:
                    t_lat = actual.lat
                    t_lon = actual.lon
                    t_alt = actual.alt
                slant, _ = env._slant_distance_and_bearing(p.lat, p.lon, p.alt, t_lat, t_lon, t_alt)
                horiz, _ = env._distance_and_bearing(p.lat, p.lon, t_lat, t_lon)
                w_range = env._attack_weapon_range(tgt)
                weapon = env.attack_controller._compatible_weapon(tgt)
                launch_ok = env._target_in_attack_launch_range(p, tgt, slant)
                ammo = env.attack_ammo.get(name, {"fox3": 0, "agm": 0})
                has_ammo = int(ammo.get(weapon, 0)) > 0
                flag = "FIRE" if (launch_ok and has_ammo) else ("MOVE" if has_ammo else "NO AMMO")
                extra = ""
                if weapon == "agm":
                    extra = f" horiz={horiz/1000:.1f}km limit=1.0km"
                print(f"  {name}: {action_names[action_id]}  slant={slant/1000:.1f}km{extra}  weapon={weapon}  ammo={'Y' if has_ammo else 'N'}  launch_ok={launch_ok}  [{flag}]")
            else:
                print(f"  {name}: platform not found")
        else:
            print(f"  {name}: slot {slot} out of range")
    else:
        action_name = action_names[action_id] if action_id < len(action_names) else "??"
        print(f"  {name}: {action_name} [{('LEADER' if is_leader else 'WINGMAN')}]")

    ok = env.apply_attack_aircraft_action(group.group_id, name, action_id)
    action_name = action_names[action_id] if action_id < len(action_names) else "??"
    print(f"  -> applied ok={ok}")
    return ok


def _step_network(env, api, trainer, group, members, live=False):
    """One decision step: network selects actions, applies them.
    In --live mode, real UDP commands go to AFSIM and we drain for updates.
    In synthetic mode, positions are simulated directly.
    """
    state = api.get_agent_state("attack", group.group_id)
    global_obs = np.zeros(2048, dtype=np.float32)
    action_names = [a.get("name", "?") for a in state["action_table"]]

    print(f"\n{'='*70}")
    print(f"  NETWORK STEP  (leader first, wingmen follow)")
    print(f"{'='*70}")

    decisions = {}
    for name in members:
        ac = state["aircraft"].get(name)
        if not ac:
            continue
        if not env.platforms.get(name) or not env.platforms[name].alive:
            print(f"  [{name}] DEAD, skipping")
            continue
        decision = trainer.act_bottom("attack", name, ac, global_state=global_obs)
        decisions[name] = int(decision["action"])

    # Apply all decisions
    for name in members:
        action_id = decisions[name]
        action_name = action_names[action_id] if action_id < len(action_names) else "??"
        mask = state["aircraft"][name]["action_mask"]
        in_mask = "Y" if mask[action_id] > 0 else "N"
        ok = env.apply_attack_aircraft_action(group.group_id, name, action_id)
        tag = "LEADER" if name == group.leader_name else "WINGMAN"
        print(f"  [{name}] {tag:>7s} -> {action_name:25s} mask={in_mask} ok={ok}")

    if live:
        env._drain_messages(timeout=1.0)

    state = api.get_agent_state("attack", group.group_id)
    _print_key_fields(state, members)


def _load_checkpoint(trainer, policy, path):
    """Load a saved checkpoint."""
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "attack_policy" in ckpt:
        policy.load_state_dict(ckpt["attack_policy"])
        print(f"Loaded attack policy from {path}")
    else:
        print(f"Checkpoint keys: {list(ckpt.keys())}")
        print("No 'attack_policy' key found")


def _wait_for_platforms(env, timeout=60.0):
    """Drain UDP messages until attack platforms appear with valid positions."""
    import time
    start = time.time()
    while time.time() - start < timeout:
        env._drain_messages(timeout=1.0)
        attack_platforms = [p for p in env.platforms.values()
                            if getattr(p, "role", "") == "attack_aircraft" and p.alive]
        valid = [p for p in attack_platforms if abs(p.lat) > 1.0 and abs(p.lon) > 1.0]
        elapsed = time.time() - start
        if len(valid) >= 6:
            names = sorted([p.name for p in valid])
            print(f"Platforms ready ({elapsed:.1f}s): {len(valid)} attack with valid positions, e.g. {names[:3]}...")
            return
        print(f"[{elapsed:.0f}s] Waiting... ({len(attack_platforms)} attack, {len(valid)} with valid position)")
    print(f"Timeout ({timeout}s). Check that warlock is running and udpnet is configured.")


def _show_distance(env, api, group, members, slot):
    """Show slant range from each member to target slot, and weapon range."""
    state = api.get_agent_state("attack", group.group_id)
    target_slots = state["team"].get("target_slots", [])
    if slot < 1 or slot > len(target_slots):
        print(f"Invalid slot {slot}")
        return
    tgt = target_slots[slot - 1]
    t_name = tgt.get("name", "?")
    t_type = tgt.get("type", "?")
    t_known = tgt.get("known", False)
    t_alive = tgt.get("alive", True)
    t_lat, t_lon, t_alt = float(tgt.get("lat", 0)), float(tgt.get("lon", 0)), float(tgt.get("alt", 0))
    w_range = env._attack_weapon_range(tgt)
    weapon = env.attack_controller._compatible_weapon(tgt)
    print(f"\n  Target slot {slot}: {t_name}  type={t_type}  known={t_known}  alive={t_alive}")
    print(f"  Position: ({t_lat:.4f}, {t_lon:.4f}, {t_alt:.0f}m)  weapon={weapon}  range={w_range/1000:.0f}km")
    for name in members:
        p = env.platforms.get(name)
        if not p or not p.alive:
            continue
        dist, _ = env._slant_distance_and_bearing(p.lat, p.lon, p.alt, t_lat, t_lon, t_alt)
        in_range = "IN RANGE -> FIRE" if dist <= w_range else f"{(dist - w_range)/1000:.0f}km short"
        ammo = env.attack_ammo.get(name, {"fox3": 0, "agm": 0})
        has_ammo = int(ammo.get(weapon, 0)) > 0
        ammo_flag = "" if has_ammo else " NO AMMO"
        print(f"  {name:>18s}: dist={dist/1000:.1f}km  [{in_range}]{ammo_flag}")


def _verify_mask(env, group):
    """Show why each target slot is available or masked."""
    leader = env.attack_controller.ensure_group_leader(group)
    if not leader:
        print("No leader")
        return
    slots = env._attack_action_target_slots()
    ammo = env.attack_ammo.get(leader.name, {"fox3": 1, "agm": 1})
    print(f"\n  Leader: {leader.name}  ammo: {ammo}")
    print(f"  {'Slot':>5s} {'Target':20s} {'Type':10s} {'Alive':>6s} {'Weapon':>7s} {'HasAmmo':>8s} {'Avail':>6s}")
    print(f"  {'-'*5} {'-'*20} {'-'*10} {'-'*6} {'-'*7} {'-'*8} {'-'*6}")
    for i, tgt in enumerate(slots):
        slot = i
        name = tgt.get("name", "?")
        ttype = tgt.get("type", "?")
        alive = "YES" if tgt.get("alive") else "no"
        weapon = env.attack_controller._compatible_weapon(tgt)
        has_ammo = "YES" if int(ammo.get(weapon, 0)) > 0 else "no"
        avail = "YES" if tgt.get("alive") and int(ammo.get(weapon, 0)) > 0 else "no"
        print(f"  {slot:>5d} {name:20s} {ttype:10s} {alive:>6s} {weapon:>7s} {has_ammo:>8s} {avail:>6s}")

    # Directly test _is_attack_slot_available for slot 0 and 10
    print(f"\n  Direct _is_attack_slot_available test:")
    for test_slot in [0, 10, 30, 33]:
        ok = env._is_attack_slot_available(leader, group, slots, test_slot, ammo, 0.0)
        print(f"    slot={test_slot} ({slots[test_slot]['name']}): {ok}")


def _create_live_attack_group(env):
    """Create the first attack team using real platforms."""
    # Use the first fixed attack team
    teams = env.attack_controller.fixed_attack_teams
    if not teams:
        raise RuntimeError("No fixed attack teams configured")
    team_names = list(teams[0])
    print(f"Creating attack group from team: {team_names}")
    group = env.attack_controller.initialize_teams(env.platforms)
    if not group:
        raise RuntimeError("Failed to create attack group")
    g = group[0]
    members = [p.name for p in g.platforms]
    return g, members


def _show_help():
    print("=" * 60)
    print("  s=show  u A=acts  u A N ... =apply  d N=distance")
    print("  x=net-step  p=masks")
    print("  h=hold  r=retreat  tN=inject  c=load-ckpt  q=quit")
    print("=" * 60)


def _move_leader(env, api, group, slot):
    leader = group.platforms[0]
    state = api.get_agent_state("attack", group.group_id)
    targets = state["team"].get("target_slots", [])
    if slot < 1 or slot > len(targets):
        print(f"Invalid slot {slot} (1-{len(targets)})")
        return
    target = targets[slot - 1]
    if not target.get("known") or not target.get("alive"):
        print(f"Target slot {slot} not known/alive, injecting...")
        lat, lon = _latlon(env, 0.6, 0.38)
        _inject_target(env, slot, lat, lon, 3000.0)
        state = api.get_agent_state("attack", group.group_id)
        targets = state["team"].get("target_slots", [])

    tgt = targets[slot - 1]
    t_lat = float(tgt.get("lat", 0))
    t_lon = float(tgt.get("lon", 0))

    dist_m, bearing = env._distance_and_bearing(leader.lat, leader.lon, t_lat, t_lon)
    print(f"Target slot {slot}: ({t_lat:.4f}, {t_lon:.4f}) dist={dist_m:.0f}m bearing={bearing:.0f}deg")

    if dist_m > 1000:
        step_m = 30000.0
        scale = min(1.0, step_m / dist_m)
        north_m = (t_lat - leader.lat) * 111320.0
        east_m = (t_lon - leader.lon) * 111320.0 * math.cos(math.radians(leader.lat))
        new_lat = leader.lat + (north_m * scale) / 111320.0
        new_lon = leader.lon + (east_m * scale) / (111320.0 * math.cos(math.radians(leader.lat)))
        leader.speed = 138.89
        leader.heading = bearing
    else:
        new_lat = t_lat
        new_lon = t_lon
        leader.speed = 0.0

    leader.lat = new_lat
    leader.lon = new_lon
    leader.task = "ATTACK"
    leader.task_status = "ASSIGNED"
    leader.task_assigned = True
    leader.at_home = False

    print(f"Leader moved to ({new_lat:.4f}, {new_lon:.4f}) dist={dist_m:.0f}m")
    state = api.get_agent_state("attack", group.group_id)
    _print_key_fields(state, members)


def _fire(env, api, group, actor, slot):
    state = api.get_agent_state("attack", group.group_id)
    action_id = _slot_action_id(state, slot)
    if action_id is None:
        print(f"No action for slot {slot}")
        return

    ok = env.apply_attack_aircraft_action(group.group_id, actor, action_id)
    print(f"Fire result: actor={actor} slot={slot} ok={ok}")
    state = api.get_agent_state("attack", group.group_id)
    _print_key_fields(state, members=[actor])


def _opportunity_fire(env, api, group, members, slot):
    """Try to fire using any member that has ammo in range."""
    state = api.get_agent_state("attack", group.group_id)
    action_id = _slot_action_id(state, slot)
    if action_id is None:
        print(f"No action for slot {slot}")
        _print_masks(state)
        return

    for name in members:
        ac = state["aircraft"].get(name, {})
        mask = ac.get("action_mask", [])
        if action_id < len(mask) and mask[action_id] > 0:
            ok = env.apply_attack_aircraft_action(group.group_id, name, action_id)
            print(f"Fire: {name} slot={slot} ok={ok}")
            state = api.get_agent_state("attack", group.group_id)
            _print_key_fields(state, members=[name])
            return
    print(f"No member has slot {slot} available in mask")
    _print_masks(state)


def _retreat(env, api, group, actor):
    state = api.get_agent_state("attack", group.group_id)
    action_names = {a.get("name", ""): i for i, a in enumerate(state["action_table"])}
    action_id = action_names.get("RETURN_HOME")
    if action_id is not None:
        ok = env.apply_attack_aircraft_action(group.group_id, actor, action_id)
        print(f"Retreat: actor={actor} ok={ok}")
        for p in group.platforms:
            p.task = "RETREAT"
            p.task_status = "ASSIGNED"
            p.at_home = False
    state = api.get_agent_state("attack", group.group_id)
    _print_key_fields(state, members=[actor])


def _hold(env, api, group, actor):
    ok = env.apply_attack_aircraft_action(group.group_id, actor, 0)
    print(f"Hold: actor={actor} ok={ok}")
    state = api.get_agent_state("attack", group.group_id)
    _print_key_fields(state, members=[actor])


def _damage(env, name):
    p = env.platforms.get(name)
    if p:
        p.current_hp = 0.5
        print(f"Damaged {name}: HP={p.current_hp}")
    else:
        print(f"Platform {name} not found")


def _kill(env, api, group, name):
    p = env.platforms.get(name)
    if p:
        p.current_hp = 0.0
        p.alive = False
        print(f"Killed {name}")
        leader = env.attack_controller.ensure_group_leader(group)
        print(f"New leader: {leader.name if leader else 'NONE'}")
    else:
        print(f"Platform {name} not found")


def _set_ammo_all(env, members, fox3, agm):
    for name in members:
        env.attack_ammo[name] = {"fox3": fox3, "agm": agm}
    print(f"Ammo set: fox3={fox3} agm={agm} for {members}")


def _inject_target(env, slot, lat, lon, alt):
    target_names = list(env.attack_fixed_target_names)
    if slot < 1 or slot > len(target_names):
        print(f"Slot {slot} out of range (1-{len(target_names)})")
        return
    target_name = target_names[slot - 1]
    role = env._configured_blue_target_role(target_name)
    env.detected_targets[target_name] = {
        "Name": target_name, "Lat": lat, "Lon": lon, "Alt": alt,
        "Type": role, "alive": True, "CurrentHP": 1.0, "MaxHP": 1.0,
    }
    env.attack_target_snapshots[target_name] = {
        "name": target_name, "known": True, "alive": True,
        "lat": lat, "lon": lon, "alt": alt,
        "type": role, "hp_norm": 1.0,
        "aam_ammo_norm": 0.0, "agm_ammo_norm": 0.0,
        "ground_ammo_norm": 0.0, "sam_ammo_norm": 0.0,
    }
    print(f"Injected target slot {slot}: {target_name} ({lat:.4f}, {lon:.4f}, {alt:.0f}m) role={role}")


def _slot_action_id(state, slot):
    for i, action in enumerate(state["action_table"]):
        name = action.get("name", "")
        if name == f"ATTACK_TARGET_{slot}":
            return i
    return None


def _print_key_fields(state, members):
    print(f"\n{'Name':>18s} {'is_leader':>10s} {'aam':>5s} {'agm':>5s} {'hp':>6s} {'dist_ldr':>10s} {'return':>7s} {'rearm':>7s} {'g_aam':>7s} {'g_agm':>7s}")
    print("-" * 90)
    for name in members:
        ac = state["aircraft"].get(name)
        if not ac:
            continue
        ob = ac["obs_by_name"]
        leader = "YES" if ob.get("is_leader") == 1.0 else "no"
        print(f"{name:>18s} {leader:>10s} {ob.get('aam_count_norm',0):>5.1f} {ob.get('agm_count_norm',0):>5.1f} {ob.get('hp_norm',0):>6.2f} {ob.get('distance_to_leader_norm',0):>10.3f} {ob.get('returning_to_carrier',0):>7.1f} {ob.get('rearming',0):>7.1f} {ob.get('formation_aam_total_norm',0):>7.2f} {ob.get('formation_agm_total_norm',0):>7.2f}")


def _print_masks(state):
    for name, ac in state["aircraft"].items():
        mask = ac.get("action_mask", [])
        action_names = [a.get("name", "?") for a in state["action_table"]]
        valid = [action_names[i] for i, v in enumerate(mask) if v > 0.0]
        print(f"\n  {name}: {len(valid)} actions")
        print(f"  {valid[:20]}{'...' if len(valid) > 20 else ''}")


def _print_state(state, members):
    for name in members:
        ac = state["aircraft"][name]
        ob = ac["obs_by_name"]
        role = "LEADER" if ob.get("is_leader") == 1.0 else "WINGMAN"
        print(f"\n{'='*55}")
        print(f"  {name} [{role}]  aam={ob.get('aam_count_norm',0):.1f} agm={ob.get('agm_count_norm',0):.1f} hp={ob.get('hp_norm',0):.1f}")
        print(f"{'='*55}")

        for f in ac["fields"]:
            v = ob.get(f, "??")
            if f.startswith("target_slot_"):
                slot = f.split("_", 3)[2]
                suffix = f.split("_", 3)[3]
                print(f"  slot_{slot:>3s} {suffix:15s} = {v}")
            else:
                print(f"  {f:35s} = {v}")

    n_target = sum(1 for f in state["aircraft"][members[0]]["fields"] if f.startswith("target_slot_"))
    print(f"\n  Total: {len(state['aircraft'][members[0]]['fields'])} fields  (non-target: {len(state['aircraft'][members[0]]['fields']) - n_target}, target slots: {n_target})")


if __name__ == "__main__":
    main()
