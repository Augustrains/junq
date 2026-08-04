import os
import sys

current_path = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(current_path, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from envs.afsim_env import AFSIMIslandEnv


def assign_fake_platform_ids(env):
    for idx, platform in enumerate(env.platforms.values(), start=1):
        platform.platform_id = idx


def choose_greedy(env, group_id, ship_name):
    state = env.get_landing_task_state(group_id)
    ship_state = state['ships'][ship_name]
    actions = state['action_table']
    mask = ship_state['action_mask']
    platform = env.platforms[ship_name]
    group = env.landing_controller.active_groups[group_id]
    zone = group.landing_zone
    current_dist, _ = env._distance_and_bearing(platform.lat, platform.lon, zone['lat'], zone['lon'])
    best = None
    for i, action in enumerate(actions):
        if i >= len(mask) or mask[i] <= 0.0:
            continue
        name = action['name']
        if name in ('HOLD', 'UNLOAD'):
            continue
        spec = env.landing_controller.action_specs[int(action['id'])]
        target = env._landing_move_target(platform, group, spec)
        if not target:
            continue
        dist, _ = env._distance_and_bearing(target[0], target[1], zone['lat'], zone['lon'])
        improvement = current_dist - dist
        if best is None or improvement > best[0]:
            best = (improvement, int(action['id']), name, target, dist)
    return best, current_dist, mask, actions


def run_zone(zone_name, ship_name, max_steps=400):
    env = AFSIMIslandEnv(bind=False)
    try:
        assign_fake_platform_ids(env)
        starts = {
            'red_transport_1': (24 + 49.0 / 60.0 + 23.65 / 3600.0, 118 + 58.0 / 60.0 + 31.64 / 3600.0),
            'red_transport_2': (25 + 5.0 / 60.0 + 1.77 / 3600.0, 119 + 15.0 / 60.0 + 24.70 / 3600.0),
            'red_transport_3': (25 + 8.0 / 60.0 + 21.75 / 3600.0, 119 + 27.0 / 60.0 + 9.33 / 3600.0),
        }
        for name, (lat, lon) in starts.items():
            env.platforms[name].lat = lat
            env.platforms[name].lon = lon
            env.platforms[name].alive = name == ship_name
            env.platforms[name].task = 'PARKED'
            env.platforms[name].task_status = 'IDLE'
        zone = next(z for z in env.config['landing_zones'] if z['name'] == zone_name)
        group = env.start_landing_group(zone, group_size=1)
        assert group is not None
        ship = group.platforms[0]
        assert ship.name == ship_name, (ship.name, ship_name)
        path = [(ship.lat, ship.lon)]
        for step in range(max_steps):
            state = env.get_landing_task_state(group.group_id)
            ship_state = state['ships'][ship.name]
            actions = {a['name']: a['id'] for a in state['action_table']}
            mask = ship_state['action_mask']
            dist, _ = env._distance_and_bearing(ship.lat, ship.lon, zone['lat'], zone['lon'])
            if mask[actions['UNLOAD']] > 0.0:
                return True, step, dist, path, 'UNLOAD_AVAILABLE'
            best, current_dist, _, _ = choose_greedy(env, group.group_id, ship.name)
            if best is None or best[0] <= -1.0:
                return False, step, dist, path, 'NO_PROGRESS'
            _, action_id, name, target, next_dist = best
            ship.lat, ship.lon = target
            path.append(target)
        dist, _ = env._distance_and_bearing(ship.lat, ship.lon, zone['lat'], zone['lon'])
        return False, max_steps, dist, path, 'MAX_STEPS'
    finally:
        env.close()


def main():
    failures = []
    ship_names = ['red_transport_1', 'red_transport_2', 'red_transport_3']
    zone_names = ['north_landing', 'central_landing', 'south_landing']
    for ship_name in ship_names:
        for zone_name in zone_names:
            ok, steps, dist, path, reason = run_zone(zone_name, ship_name)
            route_name = ship_name + '->' + zone_name
            print(route_name, 'ok=', ok, 'steps=', steps, 'dist_m=', round(dist, 1), 'reason=', reason, 'last=', path[-1], 'path_len=', len(path))
            if not ok:
                failures.append(route_name)
    if failures:
        raise SystemExit('unreachable landing routes: ' + ', '.join(failures))


if __name__ == '__main__':
    main()
