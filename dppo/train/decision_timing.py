"""Convert a bottom-policy frequency in simulation time to wall-clock pacing."""

SIMULATION_SECONDS_PER_HOUR = 3600.0
DEFAULT_BOTTOM_DECISIONS_PER_HOUR = 50.0
DEFAULT_SIMULATION_CLOCK_RATE = 40.0


def resolve_bottom_decision_timing(
    decisions_per_hour=DEFAULT_BOTTOM_DECISIONS_PER_HOUR,
    simulation_clock_rate=DEFAULT_SIMULATION_CLOCK_RATE,
    explicit_wall_seconds=0.0,
):
    """Return timing metadata for one bottom-network decision.

    ``decision_seconds`` in AFSIMIslandEnv is a wall-clock UDP drain duration.
    The requested policy frequency is expressed in simulation time, so it must
    be divided by AFSIM's clock rate before assigning it to the environment.
    """
    decisions = float(decisions_per_hour)
    clock_rate = float(simulation_clock_rate)
    explicit = float(explicit_wall_seconds)
    if decisions <= 0.0:
        raise ValueError("bottom decisions per hour must be positive")
    if clock_rate <= 0.0:
        raise ValueError("simulation clock rate must be positive")
    simulation_interval = SIMULATION_SECONDS_PER_HOUR / decisions
    wall_interval = explicit if explicit > 0.0 else simulation_interval / clock_rate
    return {
        "decisions_per_sim_hour": decisions,
        "simulation_interval_seconds": simulation_interval,
        "simulation_clock_rate": clock_rate,
        "wall_interval_seconds": wall_interval,
        "explicit_wall_override": explicit > 0.0,
    }



def apply_bottom_decision_timing(environment, timing):
    """Apply one resolved timing policy to an AFSIMIslandEnv instance.

    Keep the runtime environment and its public config in sync. Callers only
    supply the AFSIM clock rate (and, optionally, the policy frequency); every
    wall-clock timeout is then derived from this one result.
    """
    environment.decision_seconds = float(timing["wall_interval_seconds"])
    environment.decision_sim_seconds = float(timing["simulation_interval_seconds"])
    environment.simulation_clock_rate = float(timing["simulation_clock_rate"])
    scenario = environment.config.setdefault("scenario", {})
    scenario["decision_seconds"] = environment.decision_seconds
    scenario["decision_sim_seconds"] = environment.decision_sim_seconds
    scenario["simulation_clock_rate"] = environment.simulation_clock_rate
    return timing