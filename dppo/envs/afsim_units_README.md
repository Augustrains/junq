# AFSIM unit registry

`afsim_units.json` is the unit registry used by `AFSIMIslandEnv`.

When new platforms are added in `island_assault_min.txt`, keep this file in sync.
The PPO environment reads this file at startup and generates:

- red controllable units
- blue possible targets
- recon areas
- landing zones
- ground objectives
- fixed discrete action list

Current generated action types:

- `WAIT`
- `RECON:<red_recon>:<recon_area>`
- `ATTACK:<red_attack>:<blue_target>`
- `TRANSPORT:<red_transport>:<landing_zone>`
- `MOVE:<red_ground>:<ground_objective>`
- `RETREAT:<red_air_unit>` only when `scenario.expose_retreat_to_commander`
  is set to `true`

Important notes:

- Attack actions are masked until a target has been detected by `ReconReport`.
- Platform actions are masked until AFSIM sends `PlatFormAdd` and the platform id is known.
- Busy units cannot receive a new task except `RETREAT`.
- Tactical retreat is normally handled by sub-controllers through
  `AFSIMIslandEnv.request_retreat(platform_name)`.

For a quick offline shape check:

```powershell
python D:\junq\dppo\afsim_env_demo.py
```

For live use, start Warlock with `island_assault_min.txt` first, then create
`AFSIMIslandEnv()` from Python.
