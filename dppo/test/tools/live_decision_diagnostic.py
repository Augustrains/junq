import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from envs.afsim_env import AFSIMIslandEnv
from envs.rl_interface import AFSIMRLInterface


def first_valid_action_by_prefix(env, prefix):
    mask = env.get_action_mask()
    for idx, name in enumerate(env.action_names):
        if name.startswith(prefix) and idx < len(mask) and mask[idx] > 0:
            return idx, name
    return 0, "WAIT"


def main():
    env = AFSIMIslandEnv(bind=True, auto_start_warlock=True)
    try:
        ready = env.wait_for_platforms(["red_recon_1", "red_recon_2", "red_recon_3", "red_attack_1", "red_transport_1"], timeout=75)
        known = {name: p.platform_id for name, p in env.platforms.items() if p.platform_id is not None}
        print("live_platforms_ready", ready, "known_count", len(known))
        if not ready:
            print("known_platforms", known)
            raise SystemExit(1)

        api = AFSIMRLInterface(env)
        api.reset()

        action_id, action_name = first_valid_action_by_prefix(env, "RECON:")
        commander_state, team_reward, done, info = api.step_commander(action_id)
        print("commander_decision", action_id, action_name, "reward", round(team_reward, 4), "done", done)
        print("commander_last_command", info.get("last_command"))
        print("commander_events", info.get("events"))
        for name in ["red_recon_1", "red_recon_2", "red_recon_3"]:
            p = env.platforms[name]
            print("after_commander", name, "task", p.task, "status", p.task_status, "lat", round(p.lat, 6), "lon", round(p.lon, 6), "heading", round(p.heading, 2), "speed", round(p.speed, 2))

        recon_state = api.get_persistent_agent_state("recon")
        actions = {}
        chosen = {}
        for name, agent_state in recon_state["agents"].items():
            mask = agent_state["action_mask"]
            action = 0
            for idx, value in enumerate(mask):
                if idx != 0 and value > 0:
                    action = idx
                    break
            actions[name] = action
            if action != 0:
                chosen[name] = action
        next_state, bottom_reward, bottom_done, bottom_info = api.step_persistent_agents("recon", actions, advance_sim=True)
        for name in sorted(chosen.keys()):
            obs_by_name = next_state["agents"][name].get("obs_by_name", {})
            print(
                "state_check",
                name,
                "speed_norm",
                round(float(obs_by_name.get("speed_norm", 0.0)), 4),
                "heading_sin",
                round(float(obs_by_name.get("heading_sin", 0.0)), 4),
                "heading_cos",
                round(float(obs_by_name.get("heading_cos", 0.0)), 4),
            )
        print("bottom_recon_actions", chosen)
        print("bottom_reward", round(bottom_reward, 4), "done", bottom_done)
        print("bottom_events", bottom_info.get("events"))
        print("bottom_reward_details", bottom_info.get("reward_details"))
        for name in sorted(chosen.keys()):
            p = env.platforms[name]
            print("after_bottom", name, "task", p.task, "status", p.task_status, "lat", round(p.lat, 6), "lon", round(p.lon, 6), "heading", round(p.heading, 2), "speed", round(p.speed, 2))
        for drain_id in range(3):
            env._drain_messages(timeout=5.0)
            print("extra_drain", drain_id + 1)
            for name in sorted(chosen.keys()):
                p = env.platforms[name]
                print("after_drain", drain_id + 1, name, "task", p.task, "status", p.task_status, "lat", round(p.lat, 6), "lon", round(p.lon, 6), "heading", round(p.heading, 2), "speed", round(p.speed, 2))
        print("known_targets", len(env.detected_targets), sorted(env.detected_targets.keys())[:10])
    finally:
        env.close()


if __name__ == "__main__":
    main()
