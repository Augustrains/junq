# Linux Remote Training With Windows AFSIM

This note records the verified setup where Windows runs AFSIM/Warlock and
Linux runs the Python training loop.

## Topology

```text
Windows
  AFSIM / Warlock / island_assault_min.txt
  Sends PlatformState and ReconReport over UDP.
  Receives AssignTask over UDP.

Linux
  dppo Python project
  Listens on UDP 50050.
  Runs the high-level PPO and bottom HAPPO/MAPPO training code and sends actions.
```

The Linux host used in the first successful test was:

```text
HostName 10.184.17.177
User yangguobin
Port 22
```

## Linux Files Required

Linux does not need the Windows AFSIM install. Copy the `dppo` project:

```text
dppo/
  afsim_ppo_bridge.py
  agents/
  envs/
    afsim_env.py
    afsim_units.json
    *_actions.json
    *_state_fields.json
    reward_rules.json
    done_rules.json
    controllers/
    agents/
```

These Windows-side directories are not needed on Linux:

```text
AFSim-2.9.0-win64/
AFSimSdk/
afsim_work/afsim-2.9.0-win64_bin/
warlock.exe
mission.exe
```

## Linux Configuration

In the Linux copy of `dppo/envs/afsim_units.json`, listen on all interfaces:

```json
"local_address": [
  "0.0.0.0",
  50050
]
```

If the firewall is enabled, allow UDP 50050:

```bash
sudo ufw allow 50050/udp
```

For a smoke test, install at least:

```bash
pip install numpy gym pyzmq joblib
```

The current PPO code uses TensorFlow 1.x style APIs such as `tf.Session` and
`tf.placeholder`, so full PPO training should use a compatible Python and
TensorFlow environment.

## Windows Configuration

Edit the AFSIM scenario file:

```text
D:\junq\afsim_work\afsim-2.9.0-win64_bin\demos\air_to_air\scenarios\island_assault_min.txt
```

Set `udpnet address` to the Linux IP:

```text
udpnet
   port 50050
   address 10.184.17.177
   recon_range 100000
end_udpnet
```

The tested Warlock executable is:

```text
D:\junq\afsim_work\afsim-2.9.0-win64_bin\bin_release\warlock.exe
```

The Wizard executable is:

```text
D:\junq\afsim_work\afsim-2.9.0-win64_bin\bin_release\wizard.exe
```

## Start Order

Start Linux first so it is already listening for AFSIM UDP packets:

```bash
cd ~/LLM/junq/dppo
ALGORITHM=happo UPDATES=200 ROLLOUT_STEPS=128 CUDA_VISIBLE_DEVICES=0 bash scripts/linux_train_bottom_mappo.sh
```

Then start Warlock on Windows:

```powershell
cd D:\junq\afsim_work\afsim-2.9.0-win64_bin\demos\air_to_air
D:\junq\afsim_work\afsim-2.9.0-win64_bin\bin_release\warlock.exe -log-server-host localhost -log-server-port 18888 scenarios\island_assault_min.txt
```

If Warlock was already running before the scenario file was edited, close all
old `warlock.exe` and `wizard.exe` processes and reopen the scenario. Old
processes will not automatically reload the changed UDP address.
## Automatic Multi-Episode Training

Automatic reset uses SSH from Linux to Windows. Run the following once in an
Administrator PowerShell window on Windows, supplying the public key from
`~/.ssh/id_ed25519.pub` on Linux:

```powershell
powershell -ExecutionPolicy Bypass -File D:\junq\dppo\scripts\windows_enable_openssh_admin.ps1 -LinuxPublicKey "ssh-ed25519 YOUR_LINUX_PUBLIC_KEY"
```

Then run on Linux:

```bash
conda activate rl
cd ~/LLM/junq/dppo

AUTO_EPISODES=1 \
WINDOWS_AUTO_WARLOCK=1 \
WINDOWS_SSH_TARGET=yang@10.67.93.225 \
WINDOWS_SSH_KEY="$HOME/.ssh/id_ed25519" \
ALGORITHM=happo \
DECISION_SECONDS=0.1 \
UPDATES=200 \
ROLLOUT_STEPS=128 \
CUDA_VISIBLE_DEVICES=0 \
CHECKPOINT_DIR=checkpoints/bottom_happo_live \
bash scripts/linux_train_bottom_mappo.sh
```

`UPDATES` is the absolute final update id in automatic mode. On episode
termination the trainer always saves `latest.pt`, exits with the internal
terminal code, and the launcher restarts Warlock and resumes from that
checkpoint. Curriculum counters and RNG state are restored with the network and
optimizer state. To continue an interrupted run, add:

```bash
RESUME=checkpoints/bottom_happo_live/latest.pt
```

If `WINDOWS_SSH_TARGET` is omitted, the launcher attempts to infer the Windows
client IP from `SSH_CLIENT` and uses `WINDOWS_SSH_USER` (default `yang`). An
explicit target is safer when the Windows address is stable.

## Success Signals

The first failed run had only `WAIT` available:

```text
valid_actions=1
action_name=WAIT
known_targets=0
```

After UDP was connected, the Linux output showed multiple valid actions and
RECON decisions:

```text
valid_actions=6
action_name=RECON:western_defense
action_name=RECON:landing_coast
```

In Warlock/Wizard, select `red_recon_1` or another recon platform and expand
`Aux Data`. A received RECON task should show fields such as:

```text
EXTERNAL_TASK = RECON
RECON... = AREA
SEARCH... = 9144.000000
SEARCH... = 60000.000000
SEARCH... = 24.550000
SEARCH... = 120.920000
```

This confirms:

```text
Linux decision succeeded.
Python sent UDP AssignTask.
Windows AFSIM received the task.
AFSIM wrote the task parameters into platform Aux Data.
```

Task execution is a separate check. Watch for:

```text
TASK_STATUS changes from IDLE
AT_HOME_BASE changes from true
Altitude > 0
Speed > 0
Latitude/Longitude changes
known_targets increases on Linux
```

## Timing And Acceleration

The scenario currently includes:

```text
clock_rate 20.0
```

This accelerates AFSIM simulation time. Once AFSIM receives a task, aircraft
movement, detection, and task execution run under that accelerated simulation
clock.

Linux decision frequency is controlled separately by `decision_seconds` in
`dppo/envs/afsim_units.json`:

```json
"decision_seconds": 10.0
```

Wizard/Warlock acceleration does not automatically increase Linux action
frequency. To train faster, reduce `decision_seconds`, for example:

```json
"decision_seconds": 2.0
```

Reduce this gradually and watch for unstable feedback, stale state, or UDP
message loss.

## Current Code Status

The remote training loop currently exercises the commander-level action path:

```text
commander action -> AFSIMIslandEnv.step(action) -> UDP AssignTask
```

Sub-agent controllers exist for recon, attack, landing, and ground tasks, and
they can convert discrete action IDs into AFSIM task messages. The sub-agent
classes under `envs/agents/` are currently random or first-available wrappers,
not learned policies.

For CTDE work, the next implementation target is:

```text
local_obs per sub-agent
action_mask per sub-agent
global_state for centralized critic
rollout storage containing local_obs + global_state + action + reward + done
decentralized actor pi(action | local_obs, mask)
centralized critic V(global_state)
```
