"""Diagnose high-level LANDING -> bottom transport movement/unload -> UDP pipeline."""

import argparse
import sys

from envs.rl_interface import AFSIMRLInterface
from test.diagnostics.task_pipeline_diagnostics import add_common_args, choose_actor_and_action, find_commander_action, make_env, persistent_summary, print_json, ready_or_offline, step_bottom


def open_landing_window(env):
    for platform in env.platforms.values():
        if platform.side == "blue" and platform.role == "attack_aircraft":
            platform.alive = False
        if platform.side == "blue" and platform.role == "sam":
            platform.alive = False


def main(argv=None):
    parser = argparse.ArgumentParser(description="Diagnose LANDING high-level to bottom-agent execution.")
    add_common_args(parser)
    parser.add_argument("--zone", default="north_landing")
    parser.add_argument("--ship", default="red_transport_1")
    parser.add_argument("--bottom-action", default="MOVE_EAST")
    args = parser.parse_args(argv)

    env, sent = make_env(args)
    try:
        if not ready_or_offline(env, args, [args.ship]):
            return 2
        if args.offline_ready:
            open_landing_window(env)
        api = AFSIMRLInterface(env)
        action_id, action_name, valid = find_commander_action(api, "LANDING:", args.zone)
        print("commander_landing_action", action_id, action_name, "mask", valid)
        print_json("landing_window", env.get_landing_window_status())
        print_json("before_landing_state", persistent_summary(api, "landing"))
        if action_id is None or valid <= 0.0:
            print("DIAGNOSIS landing_commander_action_not_available")
            return 3
        _state, reward, done, info = api.step_commander(action_id)
        print("commander_step", "reward", round(float(reward), 4), "done", bool(done))
        print_json("commander_info", info)
        after = persistent_summary(api, "landing")
        print_json("after_landing_state", after)
        group_ids = after["active_group_ids"]
        if not group_ids:
            print("DIAGNOSIS landing_group_not_created")
            return 4
        state, actor, bottom_id, bottom_name, valid_actions = choose_actor_and_action(
            api, "landing", group_ids[0], args.ship, [args.bottom_action, "MOVE_EAST", "HOLD"]
        )
        print("bottom_actor", actor)
        print_json("bottom_valid_actions", valid_actions)
        _next, bottom_reward, bottom_done, bottom_info = step_bottom(api, "landing", group_ids[0], actor, bottom_id, not args.no_advance)
        print("forced_bottom_action", actor, bottom_id, bottom_name, "reward", round(float(bottom_reward), 4), "done", bool(bottom_done))
        print_json("bottom_info_actions", bottom_info.get("actions", {}))
        print_json("bottom_info_events", bottom_info.get("events", []))
        print_json("last_udp_send", sent[-1] if sent else {})
        if sent and sent[-1].get("Task") in ("LANDING_MOVE_POINT", "LANDING_UNLOAD", "LANDING_HOLD"):
            print("DIAGNOSIS landing_bottom_message_sent")
            return 0
        print("DIAGNOSIS landing_bottom_message_not_sent")
        return 5
    finally:
        env.close()


if __name__ == "__main__":
    sys.exit(main())
