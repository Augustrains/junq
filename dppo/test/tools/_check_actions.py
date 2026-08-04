from envs.afsim_env import AFSIMIslandEnv
env = AFSIMIslandEnv(bind=False)
env.reset()
specs = env.attack_controller.action_specs
target_actions = [s for s in specs.values() if s.get("afsim_task") == "ATTACK_TARGET_SLOT"]
print(f"Target actions: {len(target_actions)}")
print(f"Total actions: {len(specs)}")
for s in sorted(specs.values(), key=lambda x: x["id"]):
    print(f"  id={s['id']:2d}  slot={s.get('target_slot', '-'):3s}  {s['name']}")
env.close()
