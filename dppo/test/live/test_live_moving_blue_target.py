import json
import time
from pathlib import Path
from envs.afsim_env import AFSIMIslandEnv

CFG = str(Path("test/configs/afsim_units_action3_aam_test.json").resolve())

def show(label, **data): print(label, json.dumps(data, ensure_ascii=False), flush=True)

def main():
    env=AFSIMIslandEnv(config_path=CFG, bind=True, auto_start_warlock=True)
    try:
        assert env.wait_for_platforms(["red_attack_1", "blue_attack_2"], timeout=45)
        env.reset()
        red=env.platforms["red_attack_1"]; blue=env.platforms["blue_attack_2"]
        old=[blue.lat, blue.lon, blue.alt]
        env._capture_attack_target_snapshot(blue.name, {"Lat":old[0],"Lon":old[1],"Alt":old[2],"Type":blue.platform_type or blue.role,"CurrentHP":blue.current_hp,"MaxHP":blue.max_hp}, env._current_sim_time())
        # Let the actual CAP controller move blue_attack_2 away from this track.
        time.sleep(8.0)
        env._drain_messages(timeout=1.0)
        blue=env.platforms["blue_attack_2"]
        moved=env._slant_distance_and_bearing(old[0],old[1],old[2],blue.lat,blue.lon,blue.alt)[0]
        show("BLUE_MOVED", old_track=old, actual=[blue.lat,blue.lon,blue.alt], displacement_m=moved)
        assert moved > 1000.0, "blue CAP did not move"
        assert env.apply_attack_aircraft_action("attack_team_1", "red_attack_1", 3)
        deadline=time.time()+65; stopped=False
        while time.time()<deadline:
            env._drain_messages(timeout=.5)
            red=env.platforms["red_attack_1"]; blue=env.platforms["blue_attack_2"]
            if "red_attack_1" not in env.pending_attack_approaches and red.task=="ATTACK_HOLD":
                stopped=True; break
        dist=env._slant_distance_and_bearing(red.lat,red.lon,red.alt,blue.lat,blue.lon,blue.alt)[0]
        show("REAL_RANGE_STOP", stopped=stopped, red=[red.lat,red.lon,red.alt], blue_actual=[blue.lat,blue.lon,blue.alt], distance_m=dist, red_task=red.task)
        assert stopped
        assert env.apply_attack_aircraft_action("attack_team_1", "red_attack_1", 3)
        for _ in range(8): env._drain_messages(timeout=.4)
        fire=env.pending_attack_fire_commands.get("red_attack_1",{})
        show("NEXT_DECISION_FIRE", pending_fire=fire, events=env.last_reward_events[-8:])
        assert fire.get("target_name")=="blue_attack_2"
        print("MOVING_BLUE_LIVE_TEST_PASS", flush=True)
    finally:
        env.close()

if __name__=="__main__": main()