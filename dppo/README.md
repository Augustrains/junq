# AFSIM PPO Project

This project connects a PPO-style discrete-action agent to the AFSIM island
assault scenario.

## Main entry

```powershell
python D:\junq\dppo\afsim_ppo_bridge.py --agent random --episodes 1 --steps 5
```

Use `--agent random` for live smoke tests. Use `--agent ppo` after activating a
Python environment with the original PPO dependencies such as TensorFlow and Gym.

## Key files

- `afsim_ppo_bridge.py`: PPO-compatible run loop.
- `envs/afsim_env.py`: AFSIM environment wrapper. It starts Warlock, listens to
  udpnet messages, builds state, masks invalid actions, and sends tasks.
- `envs/afsim_units.json`: Unit registry and task-area configuration.
- `envs/afsim_units_README.md`: How to maintain the unit registry.
- `envs/recon_areas.json`: Cylinder-style recon area definitions used by
  commander-level recon actions.
- `envs/agent_observation_fields.json`: Observation field visibility for each
  agent type.
- `envs/spaces/mask_discrete.py`: Masked discrete action space.
- `agents/ppo_agent.py`: PPO actor/learner implementation.
- `agents/ppo_policies.py`: MLP/LSTM policies.
- `agents/utils_tf.py`: TensorFlow utilities for PPO.

## Current action flow

The agent outputs an integer action. `AFSIMIslandEnv` maps that integer to one
of the configured tasks:

- `WAIT`
- `RECON`
- `ATTACK`
- `TRANSPORT`
- `MOVE`

Invalid actions are masked before action selection.

Commander-level `RECON` actions select an area, not a specific aircraft. The
recon controller currently picks the first available recon aircraft, and later
can be replaced by a dedicated ReconAgent.

`RETREAT` is still available as an internal controller hook through
`AFSIMIslandEnv.request_retreat(platform_name)`, but it is hidden from the
commander action space by default. Tactical withdrawal should be handled inside
task/sub-agent controllers rather than selected directly by the commander.

## Agent observation views

`AFSIMIslandEnv` keeps a shared blackboard and exposes per-agent views:

```python
env.get_agent_observation("commander")
env.get_agent_observation("recon")
env.get_agent_observation("attack")
env.get_agent_observation("landing")
env.get_agent_observation("ground")
```

The fields returned by each view are configured in
`envs/agent_observation_fields.json`.
## Training interface

Use `envs.rl_interface.AFSIMRLInterface` as the stable API for RL code:

```python
from envs.afsim_env import AFSIMIslandEnv
from envs.rl_interface import AFSIMRLInterface

env = AFSIMIslandEnv(auto_start_warlock=True)
api = AFSIMRLInterface(env)

commander_state = api.reset()
specs = api.get_agent_specs()
commander_state, reward, done, info = api.step_commander(action_id)

# Preferred bottom-level training view: every configured entity is present every step.
# Unassigned entities receive an idle observation and an action mask that only allows HOLD.
recon_agents = api.get_persistent_agent_state("recon")
recon_agents, reward, done, info = api.step_persistent_agents(
    "recon",
    {"red_recon_1": 1, "red_recon_2": 6, "red_recon_4": 0},
)

# Compatibility view for an already running sub-task group:
recon_state = api.get_agent_state("recon", group_id)
recon_state, reward, done, info = api.step_task_agent(
    "recon",
    group_id,
    {"red_recon_1": 1, "red_recon_2": 6},
)
```

Reward is shared by default. The red side is one cooperative team, so commander,
recon, attack, landing, and ground agents should first optimize the same team
reward returned by `step_commander`, `step_persistent_agents`, or `step_task_agent`.

Commander observations include command-feedback fields for the latest high-level scheduling decision:

- `last_command_type_*`: the previous commander action type.
- `last_command_accepted` / `last_command_rejected`: whether scheduling was accepted.
- `last_command_activated_count_norm`: how many bottom-level entities were activated, normalized by the expected group size.
- `last_command_reject_reason_*`: coarse rejection reason flags.
- `last_command_reward_norm` and `last_command_progress_delta_norm`: latest command outcome diagnostics.
- `current_recon_complete_ratio`, `current_attack_complete_ratio`, `current_landing_complete_ratio`, `current_ground_complete_ratio`: current task-context progress.

Bottom-level observations include unit identity fields:

- `agent_id_norm`: fixed identity within the same controllable unit type.
- `group_slot_norm`: slot inside the active task group; 0 when unassigned.

These fields let shared-parameter bottom policies distinguish concrete units while
keeping parameter sharing within each unit type.

Bottom-level calls also return optional local shaping rewards in `info`:

- `team_reward`: the unchanged shared team reward.
- `local_agent_rewards`: per-entity shaping terms for progress and local credit assignment.
- `agent_rewards`: `team_reward + 0.2 * local_agent_rewards[entity]`.
- `local_reward_details`: explainable per-agent reward terms.

These local terms should help bottom-level credit assignment while preserving the
team objective as the main optimization target.

The online on-policy rollout collector lives in `train.online_rollout_collector`.
It gathers current-policy PPO/MAPPO samples for commander and bottom-level
policy streams, including obs, masks, actions, logprobs, values, rewards, dones,
and global state.

## PPO/MAPPO training scripts

The first training implementation is intentionally thin and project-native:

- `train/ppo_utils.py`: masked categorical actions, GAE, tensor conversion.
- `train/networks.py`: MLP actor/critic and collector-compatible policy wrapper.
- `train/mappo_trainer.py`: bottom-level MAPPO for recon, attack, landing, and ground policies.
- `train/scripted_commander.py`: rule-based high-level schedulers used while pretraining bottom policies, including the original priority scheduler and a condition-aware curriculum scheduler.
- `train/commander_ppo_trainer.py`: high-level commander PPO.

Bottom MAPPO pretraining with a scripted commander:

```powershell
python D:\junq\dppo\train\train_bottom_mappo.py --updates 100 --rollout-steps 128 --checkpoint-dir D:\junq\dppo\checkpoints\bottom_mappo
```

Bottom MAPPO pretraining with the condition-aware curriculum commander:

```powershell
python D:\junq\dppo\train\train_bottom_mappo.py --commander curriculum --updates 100 --rollout-steps 128 --checkpoint-dir D:\junq\dppo\checkpoints\bottom_mappo_curriculum
```

The curriculum commander first respects the high-level `action_mask`, then uses
commander-state gates such as `known_target_ratio`, `landing_window_open`, and
`ground_available` before sampling task types. Training metrics include
`commander_kind_counts` and `commander_accept_counts` so task distribution and
rejections can be audited.

Resume bottom training from the latest checkpoint:

```powershell
python D:\junq\dppo\train\train_bottom_mappo.py --updates 100 --rollout-steps 128 --checkpoint-dir D:\junq\dppo\checkpoints\bottom_mappo --resume D:\junq\dppo\checkpoints\bottom_mappo\latest.pt
```

Commander PPO training with frozen bottom policies:

```powershell
python D:\junq\dppo\train\train_commander_ppo.py --updates 100 --rollout-steps 128 --bottom-checkpoint D:\junq\dppo\checkpoints\bottom_mappo\latest.pt --checkpoint-dir D:\junq\dppo\checkpoints\commander_ppo
```

Resume commander training:

```powershell
python D:\junq\dppo\train\train_commander_ppo.py --updates 100 --rollout-steps 128 --bottom-checkpoint D:\junq\dppo\checkpoints\bottom_mappo\latest.pt --checkpoint-dir D:\junq\dppo\checkpoints\commander_ppo --resume D:\junq\dppo\checkpoints\commander_ppo\latest.pt
```

Both scripts save numbered checkpoints plus `latest.pt`, and append JSONL metrics
to `metrics.jsonl` under the checkpoint directory by default.

For live Windows AFSIM/Warlock interaction, add `--bind`. Use `--auto-start-warlock`
only when Python should start Warlock on the same Windows machine. On Linux-side
remote training with Windows running AFSIM/Warlock, keep Warlock external and use
the UDP configuration documented in `docs/linux_remote_training.md`. For the verified reverse-SSH and Scheduled Task workflow that lets Linux start, restart, and stop Warlock automatically, see `docs/linux_auto_start_warlock.md`.

