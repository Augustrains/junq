"""Print every attack state field with its current value for verification."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from envs.afsim_env import AFSIMIslandEnv
from envs.rl_interface import AFSIMRLInterface
from test.tools.state_field_diagnostic import create_synthetic_active_groups, set_synthetic_platforms


def main():
    env = AFSIMIslandEnv(bind=False)
    try:
        env.reset()
        set_synthetic_platforms(env)
        create_synthetic_active_groups(env)
        api = AFSIMRLInterface(env)

        group = next(iter(env.attack_controller.active_groups.values()))
        members = [p.name for p in group.platforms]
        print(f"Attack group: {group.group_id}")
        print(f"Members ({len(members)}): {members}")
        print(f"Leader: {group.leader_name}")

        leader_ammo = {"fox3": 1, "agm": 1}
        for i, name in enumerate(members):
            env.attack_ammo[name] = {"fox3": 1 if i == 0 else 0, "agm": 1 if i < 2 else 0}

        print(f"\nAmmo setup: { {name: env.attack_ammo.get(name) for name in members} }")

        for i, name in enumerate(members):
            state = api.get_agent_state("attack", group.group_id)
            ob = state["aircraft"][name]["obs_by_name"]
            mask = state["aircraft"][name]["action_mask"]
            action_names = [a.get("name", "") for a in state["action_table"]]
            valid_actions = [action_names[j] for j, v in enumerate(mask) if v > 0.0]

            role = "LEADER" if i == 0 else f"WINGMAN_{i}"
            print(f"\n{'='*60}")
            print(f"  {name} [{role}]")
            print(f"{'='*60}")
            print(f"  Valid actions: {valid_actions}")

            for field_name in state["aircraft"][name]["fields"]:
                value = ob.get(field_name, "MISSING")
                if isinstance(value, float):
                    mark = "MISSING" if value == 0.0 and field_name not in _expected_zero_fields(env, name, i, members) else ""
                else:
                    mark = "MISSING"
                if mark:
                    print(f"  {field_name:40s} = {value:<10} *** {mark}")
                else:
                    print(f"  {field_name:40s} = {value:<10}")

        spec = api.get_agent_specs()["attack"]
        print(f"\n{'='*60}")
        print(f"  SUMMARY")
        print(f"{'='*60}")
        print(f"  obs_dim:          {spec['obs_dim']}")
        print(f"  non-target fields: {sum(1 for f in spec['fields'] if not f.startswith('target_slot_'))}")
        print(f"  target slot fields:{sum(1 for f in spec['fields'] if f.startswith('target_slot_'))}")
        print(f"  total fields:     {len(spec['fields'])}")

        _non_target = [f for f in spec["fields"] if not f.startswith("target_slot_")]
        _target = [f for f in spec["fields"] if f.startswith("target_slot_")]
        print(f"\n  Non-target fields ({len(_non_target)}):")
        for f in _non_target:
            print(f"    {f}")
        print(f"\n  Target slot unique suffixes: {sorted(set('_'.join(f.split('_')[3:]) for f in _target))}")
        print(f"  Target slot count: {max(int(f.split('_')[2]) for f in _target) if _target else 0}")

    finally:
        env.close()


def _expected_zero_fields(env, name, idx, members):
    """Fields that are legitimately zero for certain roles."""
    is_leader = idx == 0
    zero_fields = set()
    if is_leader:
        zero_fields.update({"distance_to_leader_norm", "is_stationary"})
    return zero_fields


if __name__ == "__main__":
    main()
