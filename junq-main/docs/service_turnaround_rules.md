# Turnaround and unload service rules

## Red attack aircraft

- Firing AAM or AGM consumes only that weapon and does not force return.
- The bottom attack policy may continue moving or firing with remaining ammunition.
- Only `RETURN_HOME` starts return-to-carrier service.
- On carrier arrival the aircraft enters `rearming` for 600 simulation seconds.
- During rearming only `HOLD` is valid and no replacement UDP task is sent.
- At completion the missing one-AAM/one-AGM load is restored subject to carrier reserve stock.
- Carrier reserve stock is decremented only for missiles actually loaded.

Attack observation fields added:

- `returning_to_carrier`
- `rearming`
- `rearm_remaining_norm`
- `carrier_aam_stock_norm`
- `carrier_agm_stock_norm`

## Red transport unloading

- One `UNLOAD` action starts a whole-manifest unload operation.
- The operation lasts 900 simulation seconds per transport.
- Different transports may unload concurrently.
- During unloading the transport remains stationary and only `HOLD` is valid.
- Ground forces remain `ON_SHIP`, protected, and uncontrollable until completion.
- At completion all ground units assigned to that transport become landed together.
- The manifest is 3/4/3 ground units across `red_transport_1/2/3`.

Landing observation fields added:

- `unloading`
- `unload_remaining_norm`
- `cargo_unit_count_norm`

AFSIM logs use `REARM_START`, `REARM_COMPLETE`, `UNLOAD_START`, and `UNLOAD_COMPLETE`.