# Island Assault Unit Parameter Baseline

This document records the first public-source parameter baseline for `island_assault_min.txt`.
The values are not classified or exact tactical data. They are simulation parameters derived from public fact sheets and open sources, then adjusted to keep the scenario trainable.

## Current AFSIM Values Found

File: `D:/junq/afsim_work/afsim-2.9.0-win64_bin/demos/air_to_air/scenarios/island_assault_min.txt`

| Unit / system | Current parameter | Current value |
| --- | --- | --- |
| RED_RECON_AIRCRAFT | air sensor range | 10 nm / 18.5 km |
| RED_RECON_AIRCRAFT | ground sensor range | 5 nm / 9.3 km |
| RED_ATTACK_AIRCRAFT | onboard sensor range | 8 nm / 14.8 km |
| RED/BLUE_ATTACK_AIRCRAFT | AAM load | 1 fox3 |
| RED/BLUE_ATTACK_AIRCRAFT | AGM load | 1 agm |
| BLUE_ATTACK_AIRCRAFT | fox3 firing range in rule script | 30 km |
| BLUE_ATTACK_AIRCRAFT | agm firing range in rule script | 20 km |
| RED_TRANSPORT_SHIP | mover | WSF_AIR_MOVER placeholder |
| GROUND_FORCE | direct fire round speed | 500 m/s |
| GROUND_FORCE | detect range | 10 km |
| GROUND_FORCE | attack range | 5 km |
| BLUE_RADAR_SITE | radar range | 27.78 km / 15 nm |
| BLUE_RADAR_SITE | beam width | 60 deg |
| BLUE_RADAR_SITE | scan rate | 30 deg/s |
| BLUE_SAM_LAUNCHER | fire range | 20 km |
| BLUE_SAM_LAUNCHER | ammo | 10 missiles |
| RED_CARRIER | air radar range | 150 nm / 277.8 km |

## Public-Source Baseline

| Scenario unit | Representative real-world class | Public-source facts | Simulation recommendation |
| --- | --- | --- | --- |
| Red recon aircraft | MQ-9 Reaper-like MALE ISR UAV | USAF states MQ-9 is primarily ISR/strike, has MTS-B visual/IR sensors, SAR, 1,150 mi / 1,000 nmi range, 3,750 lb payload. Open sources list about 260 kt max speed and 150-170 kt cruise. | Cruise speed 80-100 m/s; operating altitude 20,000-30,000 ft. Keep ground/air sensor range around 10-25 km unless using a dedicated SAR wide-area mode. |
| Red/Blue attack aircraft | F-16-like multirole fighter | USAF F-16 fact sheet: Mach 2 / 1,500 mph at altitude, >500 mi air-to-surface mission radius, >50,000 ft ceiling, AAM and A/G munitions. | Mission speed 250-300 m/s, altitude 25,000-35,000 ft. AAM range 30-50 km for game balance; AGM 10-20 km if modeled as Hellfire/Maverick-like short tactical strike. |
| Medium-range AAM | AIM-120 AMRAAM-like | Open source commonly lists Mach 4 and tens to >100 km depending variant. | Keep actual firing range 30-50 km in this small Taiwan-north scenario; avoid 100+ km early because it collapses maneuver/recon learning. |
| Air-to-ground missile | AGM-114 Hellfire-like for short tactical strike | CSIS lists Hellfire range 7-11 km, subsonic to Mach 1.3 / about 450 m/s. | If using Hellfire-like missile: 8-12 km. If using fighter-launched Maverick/stand-off generic missile: 15-25 km. Current 20 km is acceptable as generic AGM. |
| Blue radar site | Ground air-search radar, simplified sector scan | Exact radar ranges are often sensitive; Patriot/THAAD-class radars can be far longer than this scenario needs. | Use 30-50 km sector scan for training. Current 27.78 km is conservative; increase only if blue defense is too weak. |
| Blue SAM launcher | Medium/point air defense, not full Patriot envelope | Patriot-class public max missile ranges can be ~160 km; RAM point-defense is ~9 km. | Use 20-30 km to create avoidable threat bubbles. Current 20 km is reasonable for learning path planning. |
| Red carrier | Nimitz-class carrier-like sea base | Public sources: Nimitz class over 30 kt, large air wing; U.S. Navy fact sheet describes carriers as survivable mobile airfields. | In this scenario keep carrier fixed as required. If moving later, use 12-15 m/s. Carrier radar can remain long, but if it gives red omniscience, reduce to 50-100 km. |
| Red transport ship | LPD/landing craft-like amphibious transport | San Antonio-class LPD public speed >22 kt and troop capacity hundreds; landing craft can be slower. | Use 8-12 m/s for transport ship movement. If modeling small landing craft, use 5-7 m/s. |
| Ground force | Battalion-sized mechanized/infantry group abstraction | Infantry foot movement is much slower than vehicles; exact combat movement varies. | If infantry-only: 1-2 m/s. If mechanized battalion abstraction: 5-10 m/s. Keep ground fire 3-5 km only if representing mortars/ATGM/vehicle weapons; otherwise rifle fire should be far shorter. |

## Recommended First Parameter Pass

These are deliberately moderate values for a learnable first scenario.

| Parameter | Recommended value | Why |
| --- | --- | --- |
| red recon cruise speed | 90 m/s | MQ-9-like cruise; slow enough for SAM avoidance to matter. |
| red recon altitude | 25,000 ft | Public MQ-9 operating altitude is around this order. |
| red recon air detect range | 20 km | Enough to find aircraft without becoming omniscient. |
| red recon ground detect range | 15 km | Slightly longer than current 5 nm but still local. |
| attack aircraft mission speed | 275 m/s | Subsonic tactical cruise for fighter-like aircraft. |
| attack aircraft altitude | 30,000 ft | Matches current fighter altitude style. |
| attack aircraft sensor range | 25 km | Current 8 nm is too short for a fighter abstraction; 25 km is still not overpowered. |
| AAM firing range | 35-40 km | Based on AMRAAM-like class but compressed for scenario scale. |
| AGM firing range | 15-20 km | Current 20 km is acceptable for generic short tactical A/G missile. |
| blue radar range | 35-45 km | Gives blue early warning while preserving red path planning. |
| blue radar beam width | 60 deg | Current value is acceptable. |
| blue radar scan rate | 30 deg/s | Current value is acceptable. |
| blue SAM firing range | 20-25 km | Current 20 km is acceptable and creates avoidable SAM bubbles. |
| transport ship speed | 10 m/s | Approx. 19.4 kt, close to LPD/landing craft order of magnitude. |
| ground force speed | 5 m/s | Mechanized/abstract battalion movement, not pure foot march. |
| ground detect range | 8-10 km | Current 10 km is acceptable for unit-level ground observation abstraction. |
| ground fire range | 3-5 km | Current 5 km is acceptable if representing heavy weapons / ATGM / mortar-like fire. |

## Source Links

- U.S. Air Force F-16 Fighting Falcon fact sheet: https://www.af.mil/About-Us/Fact-Sheets/Display/Article/104505/f-16-fighting-falcon/
- U.S. Air Force MQ-9 Reaper fact sheet: https://www.af.mil/About-Us/Fact-Sheets/Display/Article/104470/mq-9-reaper/
- U.S. Navy aircraft carrier fact file: https://www.navy.mil/Resources/Fact-Files/Display-FactFiles/Article/2169795/aircraft-carriers-cvn/
- CSIS Missile Threat Hellfire page: https://missilethreat.csis.org/missile/agm-114-hellfire/
- Open source AMRAAM reference: https://en.wikipedia.org/wiki/AIM-120_AMRAAM
- Open source Nimitz-class reference: https://en.wikipedia.org/wiki/Nimitz-class_aircraft_carrier
- Open source San Antonio-class LPD reference: https://en.wikipedia.org/wiki/San_Antonio-class_amphibious_transport_dock
- Open source Patriot reference: https://en.wikipedia.org/wiki/MIM-104_Patriot
- Open source RAM reference: https://en.wikipedia.org/wiki/RIM-116_Rolling_Airframe_Missile
## Applied Parameter Pass 2026-07-04

Applied to `island_assault_min.txt` and verified through the live interface plus `afsim_ppo_bridge.py`.

| Parameter | Applied value |
| --- | --- |
| RED_RECON_AIRCRAFT air sensor | 20 km |
| RED_RECON_AIRCRAFT ground sensor | 15 km |
| RED_ATTACK_AIRCRAFT sensor | 25 km |
| RED_ATTACK_AIRCRAFT fox3 fire range | 40 km |
| RED_ATTACK_AIRCRAFT agm fire range | 20 km |
| BLUE_ATTACK_AIRCRAFT fox3 fire range | 40 km |
| BLUE_ATTACK_AIRCRAFT agm fire range | 20 km |
| BLUE_RADAR_SITE radar range | 40 km |
| BLUE_RADAR_SITE geometric sensor range | 40 km |
| BLUE_SAM_LAUNCHER fire range | 25 km |
| RED_CARRIER air radar range | 60 nm |
| RED_TRANSPORT_SHIP movement speed | 10 m/s |
| GROUND_FORCE movement speed | 5 m/s |
| GROUND_FORCE detect range | 10 km |
| GROUND_FORCE attack range | 5 km |

Verification:

- Live interface verification: passed.
- `python D:\junq\dppo\afsim_ppo_bridge.py --agent random --episodes 1 --steps 5 --platform-timeout 60`: passed.

## Applied Contract 2026-07-16

These values supersede the earlier balancing recommendations above.

| Parameter | Applied value |
| --- | --- |
| Recon and attack aircraft maximum speed | 500 km/h / 138.888889 m/s |
| Recon aircraft altitude ceiling | 10000 m |
| Attack aircraft altitude ceiling | 6000 m |
| Red transport maximum speed | 50 km/h / 13.888889 m/s |
| Recon aircraft air/ground detection range | 100 km |
| Red carrier air radar range | 300 km |
| Blue ground radar range | 300 km |
| Aircraft commanded-speed range | 40 to 138.888889 m/s, selected by SPEED_UP/SPEED_DOWN |
| Aircraft speed action increment | 20 m/s |

The AFSIM mover, scripted GoToSpeed command, udpnet forwarding, Python observation normalization, and action masks use the same limits.