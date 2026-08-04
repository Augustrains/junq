# Combat Model

The authoritative Python-side parameters are in `envs/combat_model.json`.
AFSIM scenario scripts mirror the same rules for physical execution.

## Durability

| Unit | Max HP | Notes |
| --- | ---: | --- |
| Red/blue reconnaissance and attack aircraft | 1 | Destroyed after one successful damage event |
| Red/blue ground force | 5 | Each successful hit removes 1 HP |
| Blue SAM and radar | 3 | Each successful hit removes 1 HP |
| Red carrier, red transports, blue base | 0 | Explicitly indestructible |

The same role always receives the same combat fields regardless of side:
`MaxHP`, `CurrentHP`, `LastAttackTime`, `LastDamageTime`,
`FireCooldownUntil`, and `CombatLockUntil`.

## Hit Rules

| Attack | Probability after physical contact | Damage |
| --- | ---: | ---: |
| AAM against aircraft | 0.50 | 1 |
| SAM against aircraft | 0.50 | 1 |
| AGM against radar/SAM/ground force | 0.80 | 1 |
| Ground fire against radar/SAM/ground force | 1.00 | 1 |

AAM and SAM use native target-track guidance and native weapon contact first.
AGM uses an AFSIM guided ground weapon platform; its actual position is sampled
every 0.1 simulation seconds. A compatible target inside 3 km horizontally and
500 m vertically causes a contact-resolution probability draw. The assigned
target is not granted an automatic hit. Aircraft climb to at least 3000 m
before releasing AAM/AGM so a parked aircraft cannot fire at zero altitude.

## Ground Combat

Ground detection and ground-fire authorization are local to the firing ground
unit. A reconnaissance-aircraft track alone cannot authorize ground fire.
Maximum range is 5 km. A successful attack removes 1 HP.

After firing or receiving ground fire, a unit receives a 300-second combat
movement lock. A shooter also receives a 300-second fire cooldown. During the
lock, movement and capture actions are masked; during cooldown, fire is masked.

## Training State

Commander state includes aggregate red and known-blue HP ratios. Recon and
attack observations include own HP; attack target slots include target HP.
Ground observations include own HP, damage/cooldown/lock status, and nearest
locally detected target HP. Stage snapshots preserve all combat and timing
fields so resumed curricula do not reset durability or cooldowns.

Relevant logs are `[COMBAT_DAMAGE] MISS|HIT|DESTROYED`,
`[AIR_MISSILE] SCRIPT_CONTACT|TERMINAL_CONTACT`, and
`[GROUND_COMBAT] LOCKED`.
