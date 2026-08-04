"""Deterministic live audit of reconnaissance and attack command closure.

This script does not load or update a policy.  It uses the same AFSIMIslandEnv
command methods as training and writes one JSONL audit record per phase.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.afsim_env import AFSIMIslandEnv


def write_record(stream, phase, **values):
    row = {"wall_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "phase": phase}
    row.update(values)
    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    stream.flush()
    print("ACTION_AUDIT", json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)


def platform_snapshot(env, names):
    result = {}
    for name in names:
        p = env.platforms.get(name)
        if p is None:
            continue
        result[name] = {
            "id": p.platform_id, "alive": bool(p.alive), "task": p.task,
            "task_status": p.task_status, "location": [p.lat, p.lon, p.alt],
            "hp": p.current_hp,
        }
        if p.role == "attack_aircraft":
            result[name]["ammo"] = dict(env.attack_ammo.get(name, {}))
    return result


def wait_until(env, timeout, predicate):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        env._drain_messages(timeout=min(0.5, max(0.05, deadline - time.monotonic())))
        env.step_count += 1
        if predicate():
            return True
    return False


def start_task(task_name):
    command = (
        "$ErrorActionPreference='Stop'; "
        "Stop-ScheduledTask -TaskName '{0}' -ErrorAction SilentlyContinue; "
        "Start-Sleep -Seconds 2; Start-ScheduledTask -TaskName '{0}'"
    ).format(task_name.replace("'", "''"))
    subprocess.run(["powershell", "-NoProfile", "-Command", command], check=True)


def first_detected_target(env):
    candidates = []
    for name, target in env.detected_targets.items():
        if bool(target.get("alive", True)) and name in env.platforms and env.platforms[name].side == "blue":
            candidates.append(name)
    for name, target in env.attack_target_snapshots.items():
        if bool(target.get("alive", True)) and name in env.platforms and env.platforms[name].side == "blue":
            candidates.append(name)
    return next(iter(dict.fromkeys(candidates)), "")


def target_action_id(env, target_name):
    slots = env._attack_action_target_slots()
    target_slot = next((index for index, target in enumerate(slots) if target.get("name") == target_name), None)
    if target_slot is None:
        return None
    for action_id, spec in sorted(env.attack_controller.action_specs.items()):
        if spec.get("afsim_task") == "ATTACK_TARGET_SLOT" and int(spec.get("target_slot", -1)) == target_slot:
            return int(action_id)
    return None


def wait_for_attack_fire(env, group_id, attacker, action_id, ammo_before, timeout):
    """Follow a moving target until AFSIM reports arrival at launch range, then fire."""
    deadline = time.monotonic() + timeout
    reissued = 0
    approach_completed = False
    while time.monotonic() < deadline:
        env._drain_messages(timeout=min(0.5, max(0.05, deadline - time.monotonic())))
        env.step_count += 1
        if env.attack_ammo.get(attacker, {}) != ammo_before:
            return True, reissued, approach_completed
        if attacker in env.pending_attack_approaches:
            continue
        # The first target action is ATTACK_MOVE_POINT. Once live geometry
        # reaches weapon range, a second explicit decision produces FIRE.
        approach_completed = True
        if env.apply_attack_aircraft_action(group_id, attacker, action_id):
            reissued += 1
    return False, reissued, approach_completed

def successful_ack(task_acks, task_name):
    return any(
        bool(ack.get("Success")) and str(ack.get("Task", "")) == task_name
        for ack in task_acks
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-address", default="0.0.0.0:50160")
    parser.add_argument("--config", default="test/configs/afsim_units_action3_aam_test.json")
    parser.add_argument("--task-name", default="AFSIM-ActionAudit-0")
    parser.add_argument("--start-task", action="store_true")
    parser.add_argument("--recon-timeout", type=float, default=90.0)
    parser.add_argument("--attack-timeout", type=float, default=120.0)
    parser.add_argument("--report-dir", default="test/artifacts/action_chain")
    args = parser.parse_args()
    host, port_text = args.local_address.rsplit(":", 1)
    port = int(port_text)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    audit_path = report_dir / "action_audit_worker_0.jsonl"
    summary_path = report_dir / "action_audit_summary.json"
    if args.start_task:
        start_task(args.task_name)
        time.sleep(3.0)

    env = AFSIMIslandEnv(config_path=str(Path(args.config).resolve()), bind=True,
                         auto_start_warlock=False, local_address=(host, port))
    task_acks = []
    original_handle = env._handle_message
    def audited_handle(message):
        if str(message.get("MsgType", "")) == "TaskAck":
            task_acks.append(dict(message))
        return original_handle(message)
    env._handle_message = audited_handle
    success = False
    failure = ""
    try:
        required = ["red_recon_1", "red_attack_1", "blue_attack_1"]
        with audit_path.open("w", encoding="utf-8") as audit:
            ready = env.wait_for_platforms(required, timeout=45.0)
            write_record(audit, "startup", ready=ready, port=port,
                         registered=sum(p.platform_id is not None for p in env.platforms.values()),
                         platforms=platform_snapshot(env, required))
            if not ready:
                raise RuntimeError("required platforms did not register")
            env.initialize_bottom_teams()
            areas = list(env.recon_areas)
            # The production flat-action configuration deliberately has no
            # static recon-area file. The audit supplies the standard
            # western screen used by the task definitions.
            if not areas:
                areas = [{
                    "name": "action_audit_western_screen", "center_lat": 24.98,
                    "center_lon": 121.055, "radius_m": 14000.0,
                    "altitude_min_m": 3000.0, "altitude_max_m": 10000.0,
                    "default_alt_m": 9144.0, "duration_sec": 300.0, "priority": 1,
                }]
            recon_group = env.start_recon_group(areas[0], fixed_team_index=0)
            if recon_group is None:
                raise RuntimeError("recon group dispatch rejected: {0}".format(env.last_reward_events[-8:]))
            recon_members = [p.name for p in recon_group.platforms]
            write_record(audit, "recon_command_sent", group=recon_group.group_id,
                         members=recon_members, events_tail=list(env.last_reward_events[-20:]),
                         before=platform_snapshot(env, recon_members))
            detected = wait_until(env, args.recon_timeout, lambda: bool(first_detected_target(env)))
            target_name = first_detected_target(env)
            recon_ack = successful_ack(task_acks, "RECON")
            write_record(audit, "recon_result", contacts_detected=detected, target=target_name,
                         recon_task_ack=recon_ack, task_acks=list(task_acks), detected_targets=env.detected_targets,
                         attack_snapshots=env.attack_target_snapshots,
                         after=platform_snapshot(env, recon_members))
            if not detected:
                raise RuntimeError("no blue target reached the attack observation after recon timeout")

            attack_group = env.start_attack_group(target_name, fixed_team_index=0)
            if attack_group is None:
                raise RuntimeError("attack group dispatch rejected: {0}".format(env.last_reward_events[-8:]))
            attacker = attack_group.leader_name
            action_id = target_action_id(env, target_name)
            if action_id is None:
                raise RuntimeError("no attack action maps to detected target {0}".format(target_name))
            target_before = platform_snapshot(env, [target_name]).get(target_name, {})
            ammo_before = dict(env.attack_ammo.get(attacker, {}))
            sent = env.apply_attack_aircraft_action(attack_group.group_id, attacker, action_id)
            write_record(audit, "attack_command_sent", group=attack_group.group_id, attacker=attacker,
                         target=target_name, action_id=action_id, sent=bool(sent),
                         ammo_before=ammo_before, target_before=target_before,
                         pending_approach=dict(env.pending_attack_approaches), task_acks=list(task_acks))
            if not sent:
                raise RuntimeError("attack command was rejected by the live action mask")

            fired, reissued, approach_completed = wait_for_attack_fire(
                env, attack_group.group_id, attacker, action_id, ammo_before, args.attack_timeout
            )
            target_after = platform_snapshot(env, [target_name]).get(target_name, {})
            ammo_after = dict(env.attack_ammo.get(attacker, {}))
            damage = float(target_after.get("hp", 0.0)) < float(target_before.get("hp", 0.0))
            fire_ack = successful_ack(task_acks, "FIRE_AAM") or successful_ack(task_acks, "FIRE_AGM")
            write_record(audit, "attack_result", weapon_fired=bool(fired), attacker=attacker,
                         target=target_name, ammo_before=ammo_before, ammo_after=ammo_after,
                         target_before=target_before, target_after=target_after, target_damaged=damage,
                         approach_completed=approach_completed, fire_reissues=reissued,
                         fire_task_ack=fire_ack, task_acks=list(task_acks), reward_events_tail=list(env.last_reward_events[-20:]))
            success = bool(detected and sent and fired)
            if not success:
                failure = "closure gate failed: detected={0} command_sent={1} approach_completed={2} fire_command_sent={3}".format(
                    detected, sent, approach_completed, fired
                )
    except Exception as error:
        failure = str(error)
    finally:
        summary = {"passed": success, "failure": failure, "audit_file": str(audit_path), "port": port}
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print("ACTION_CHAIN_SUMMARY", json.dumps(summary, ensure_ascii=False), flush=True)
        env.close()
    return 0 if success else 1

if __name__ == "__main__":
    raise SystemExit(main())