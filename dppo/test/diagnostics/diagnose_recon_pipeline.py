"""Diagnose high-level RECON -> bottom recon movement -> UDP pipeline."""

import argparse
import sys

from envs.rl_interface import AFSIMRLInterface
from test.diagnostics.task_pipeline_diagnostics import add_common_args, choose_actor_and_action, find_commander_action, make_env, persistent_summary, print_json, ready_or_offline, step_bottom


def main(argv=None):
    parser = argparse.ArgumentParser(description="Diagnose RECON high-level to bottom-agent execution.")
    add_common_args(parser)
    parser.add_argument("--area", default="western_screen")
    parser.add_argument("--actor", default="red_recon_1")
    parser.add_argument("--bottom-action", default="MOVE_TOWARD_AREA")
    args = parser.parse_args(argv)

    env, sent = make_env(args)
    try:
        recon_names = env.config.get("red", {}).get("commandable_recon_aircraft", [])[: env.config.get("scenario", {}).get("recon_group_size", 3)]
        if not ready_or_offline(env, args, recon_names or [args.actor]):
            return 2
        api = AFSIMRLInterface(env)
        action_id, action_name, valid = find_commander_action(api, "RECON:", args.area)
        print("commander_recon_action", action_id, action_name, "mask", valid)
        print_json("before_recon_state", persistent_summary(api, "recon"))
        if action_id is None or valid <= 0.0:
            print("DIAGNOSIS recon_commander_action_not_available")
            return 3
        _state, reward, done, info = api.step_commander(action_id)
        print("commander_step", "reward", round(float(reward), 4), "done", bool(done))
        print_json("commander_info", info)
        after = persistent_summary(api, "recon")
        print_json("after_recon_state", after)
        group_ids = after["active_group_ids"]
        if not group_ids:
            print("DIAGNOSIS recon_group_not_created")
            return 4
        state, actor, bottom_id, bottom_name, valid_actions = choose_actor_and_action(
            api, "recon", group_ids[0], args.actor, [args.bottom_action, "MOVE_EAST", "HOLD"]
        )
        print("bottom_actor", actor)
        print_json("bottom_valid_actions", valid_actions)
        _next, bottom_reward, bottom_done, bottom_info = step_bottom(api, "recon", group_ids[0], actor, bottom_id, not args.no_advance)
        print("forced_bottom_action", actor, bottom_id, bottom_name, "reward", round(float(bottom_reward), 4), "done", bool(bottom_done))
        print_json("bottom_info_actions", bottom_info.get("actions", {}))
        print_json("bottom_info_events", bottom_info.get("events", []))
        print_json("last_udp_send", sent[-1] if sent else {})
        if sent and sent[-1].get("Task") == "RECON":
            print("DIAGNOSIS recon_bottom_message_sent")
            return 0
        print("DIAGNOSIS recon_bottom_message_not_sent")
        return 5
    finally:
        env.close()


if __name__ == "__main__":
    sys.exit(main())
