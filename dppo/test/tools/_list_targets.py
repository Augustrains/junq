import json
cfg = json.load(open("envs/afsim_units.json", "r", encoding="utf-8-sig"))
blue = cfg["blue"]
names = []
for key in ("attack_aircraft", "ground_forces", "radars", "sams"):
    for n in blue.get(key, []):
        if n not in names:
            names.append(n)
for i, n in enumerate(names):
    slot = i + 1
    action_id = i + 2  # id=0 HOLD, id=1 RETURN_HOME
    if n in blue.get("attack_aircraft", []): role = "aircraft"
    elif n in blue.get("ground_forces", []): role = "ground"
    elif n in blue.get("radars", []): role = "radar"
    elif n in blue.get("sams", []): role = "SAM"
    else: role = "?"
    print(f"  id={action_id:2d}  ATTACK_TARGET_{slot:<2d}  {n:20s}  {role}")
