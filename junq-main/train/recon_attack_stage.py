"""Reconnaissance/attack curriculum boundary and terminal quality scoring."""

from __future__ import annotations


def high_quality_landing_status(env, min_recon_alive=3, min_attack_alive=3, min_loaded_transports=1):
    window = env.get_landing_window_status()
    recon_alive = sum(
        1 for platform in env.platforms.values()
        if platform.side == "red" and platform.role == "recon_aircraft" and platform.alive
    )
    attack_alive = sum(
        1 for platform in env.platforms.values()
        if platform.side == "red" and platform.role == "attack_aircraft" and platform.alive
    )
    loaded_transports = sum(
        1 for name, cargo in env.landing_cargo.items()
        if cargo.get("has_army", False) and not cargo.get("army_landed", False)
        and env.platforms.get(name) is not None and env.platforms[name].alive
    )
    checks = {
        "landing_window_open": bool(window["open"]),
        "recon_viable": recon_alive >= int(min_recon_alive),
        "attack_viable": attack_alive >= int(min_attack_alive),
        "transport_viable": loaded_transports >= int(min_loaded_transports),
    }
    ready = all(checks.values())
    return {
        "ready": ready,
        "checks": checks,
        "destroyed_blue_air": int(window["destroyed_blue_air"]),
        "required_blue_air_kills": int(window["required_blue_air_kills"]),
        "alive_blue_sams": int(window["alive_blue_sams"]),
        "max_alive_blue_sams": int(window["max_alive_blue_sams"]),
        "landing_trigger_reason": str(window["trigger_reason"]),
        "landing_combat_conditions_met": bool(window["combat_conditions_met"]),
        "landing_time_override_met": bool(window["time_override_met"]),
        "landing_time_seconds": float(window["time_seconds"]),
        "landing_force_open_time_seconds": float(window["force_open_time_seconds"]),
        "recon_alive": recon_alive,
        "attack_alive": attack_alive,
        "loaded_transports": loaded_transports,
    }


def landing_boundary_bonus(status, sim_time, deadline_seconds=10800.0,
                           deadline_miss_penalty=-50.0, negative_rewards_enabled=False):
    """Return only stage success (or the opt-in deadline penalty)."""
    combat_qualified = bool(status.get("landing_combat_conditions_met", False))
    deadline_missed = not combat_qualified and float(sim_time) >= float(deadline_seconds)
    if deadline_missed:
        penalty = float(deadline_miss_penalty)
        return penalty if negative_rewards_enabled or penalty >= 0.0 else 0.0
    if status.get("ready", False) and combat_qualified:
        return 1.0
    return 0.0


def recon_attack_terminal_bonus(
    status,
    terminal_reason,
    sim_time,
    deadline_seconds=10800.0,
    deadline_miss_penalty=-50.0,
    negative_rewards_enabled=False,
):
    """Return the stage bonus for both success and deadline termination."""
    if terminal_reason not in ("combat_landing_ready", "landing_deadline_missed"):
        return 0.0
    return landing_boundary_bonus(
        status,
        sim_time,
        deadline_seconds=deadline_seconds,
        deadline_miss_penalty=deadline_miss_penalty,
        negative_rewards_enabled=negative_rewards_enabled,
    )


def canonicalize_landing_snapshot(env):
    for controller in (env.recon_controller, env.attack_controller, env.landing_controller, env.ground_controller):
        controller.active_groups.clear()
        controller.next_group_index = 1
    env.pending_attack_returns = {}
    env.pending_attack_fire_commands = {}
    env.pending_landing_unloads = {}
    env.attack_target_reservations = {}
    env.ground_target_reservations = {}
    for platform in env.platforms.values():
        if platform.side == "red" and platform.alive:
            platform.task = "PARKED"
            platform.task_status = "IDLE"
            platform.speed = 0.0
