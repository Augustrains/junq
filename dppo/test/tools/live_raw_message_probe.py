import json
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
    original_handle = env._handle_message
    printed = {"count": 0}

    def traced_handle(msg):
        msg_type = msg.get("MsgType", "")
        name = msg.get("PlatformName", "")
        if msg_type in ("PlatFormAdd", "MoveUpdate", "TaskAck") and (
            name.startswith("red_recon_") or msg.get("PlatformId") in {env.platforms.get("red_recon_1").platform_id if "red_recon_1" in env.platforms else None}
        ):
            if printed["count"] < 40:
                print("RAW", json.dumps(msg, ensure_ascii=False, sort_keys=True))
                printed["count"] += 1
        original_handle(msg)

    env._handle_message = traced_handle
    try:
        ready = env.wait_for_platforms(["red_recon_1", "red_recon_2", "red_recon_3"], timeout=75)
        print("ready", ready)
        api = AFSIMRLInterface(env)
        api.reset()
        action_id, action_name = first_valid_action_by_prefix(env, "RECON:")
        api.step_commander(action_id)
        print("sent_commander", action_id, action_name)
        recon_state = api.get_persistent_agent_state("recon")
        actions = {name: 1 for name in recon_state["agents"].keys()}
        api.step_persistent_agents("recon", actions, advance_sim=True)
        print("sent_bottom", sorted(actions.keys()))
        for _ in range(5):
            env._drain_messages(timeout=3.0)
        for name in ["red_recon_1", "red_recon_2", "red_recon_3"]:
            p = env.platforms[name]
            print("FINAL", name, p.lat, p.lon, p.alt, p.heading, p.speed, p.last_update)
    finally:
        env.close()


if __name__ == "__main__":
    main()
