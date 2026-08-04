# Mission Termination And Scoring

The live mission uses AFSIM simulation time rather than Python decision-step
count. Its fixed horizon is six hours (`21600` simulation seconds).

## Operational timing

- `0-2 h`: reconnaissance and strike preparation emphasis.
- `2-5 h`: landing and ground combat emphasis; air-ground coordination remains available.
- `5-6 h`: completion buffer.
- `6 h`: mandatory score settlement and episode termination.

These windows do not hard-disable agent types. Commander `progress` is
`TIME_NOW / 21600`, allowing the policy to learn the timing without removing
late reconnaissance or strike actions.

## Final score

Only red ground-force units satisfying every condition are counted:

- alive with positive `CurrentHP`;
- landed and no longer on a transport;
- no farther than `1000 m` from `blue_base` center.

`final_score_raw` is the sum of their remaining HP. Ten red ground units with
five HP each produce a range of `0-50`. `final_score_norm` is
`final_score_raw / 50`.

The terminal transition receives one reward point per scored HP. Process
rewards for reconnaissance, damage, landing, and capture remain active.
Capture is no longer an early terminal condition. Complete red ground-force
destruction may terminate early with score zero.

Terminal output fields are:

- `episode_result=FIXED_HORIZON_COMPLETE`;
- `done_reason=fixed_horizon`;
- `final_score_raw`;
- `final_score_norm`;
- `final_score_unit_count`;
- `final_score_units`;
- `final_score_sim_time`.

AFSIM logs the corresponding `[FINAL_SCORE]` and `[SCENARIO_END]` lines before
calling `WsfSimulation.Terminate()`.
