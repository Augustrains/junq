"""Flat always-on multi-agent rollout collection."""

import time

import numpy as np

BOTTOM_AGENT_TYPES = ("recon", "attack", "landing", "ground")


class RolloutStream(object):
    def __init__(self, name):
        self.name = name
        self.rows = []

    def append(self, **kwargs):
        self.rows.append(dict(kwargs))

    def __len__(self):
        return len(self.rows)

    def to_numpy(self):
        if not self.rows:
            return {}
        batch = {}
        for key in sorted(self.rows[0]):
            values = [row.get(key) for row in self.rows]
            if key in ("obs", "next_obs", "action_mask", "global_state"):
                batch[key] = np.asarray(values, dtype=np.float32)
            elif key == "action":
                vector = any(np.asarray(value).ndim > 0 for value in values)
                batch[key] = np.asarray(
                    values, dtype=np.float32 if vector else np.int64
                )
            elif key in (
                "reward", "value", "next_value", "logprob", "done",
                "task_done", "team_reward",
                "shared_reward", "local_reward", "weighted_local_reward",
                "target_contribution_reward",
                "weighted_target_contribution_reward",
            ):
                batch[key] = np.asarray(values, dtype=np.float32)
            else:
                batch[key] = np.asarray(values, dtype=object)
        return batch

    def summary(self):
        batch = self.to_numpy()
        if not batch:
            return {"count": 0}
        reward = batch.get("reward", np.asarray([], dtype=np.float32))
        summary = {
            "count": len(self.rows),
            "reward_mean": float(np.mean(reward)) if reward.size else 0.0,
            "reward_sum": float(np.sum(reward)) if reward.size else 0.0,
            "reward_min": float(np.min(reward)) if reward.size else 0.0,
            "reward_max": float(np.max(reward)) if reward.size else 0.0,
        }
        if "obs" in batch:
            summary["obs_shape"] = tuple(batch["obs"].shape)
        if "action_mask" in batch:
            summary["mask_shape"] = tuple(batch["action_mask"].shape)
            summary["valid_action_ratio"] = float(
                np.mean(batch["action_mask"] > 0)
            )
        return summary


class BottomOnlyRollout(object):
    """Transitions produced only by persistent bottom-level agents."""

    def __init__(self, agent_types=BOTTOM_AGENT_TYPES):
        self.bottom = {
            agent_type: RolloutStream(agent_type) for agent_type in agent_types
        }
        self.step_infos = []

    def to_numpy(self):
        return {
            "bottom": {
                agent_type: stream.to_numpy()
                for agent_type, stream in self.bottom.items()
            }
        }

    def summary(self):
        final_info = dict(
            self.step_infos[-1].get("environment", {})
        ) if self.step_infos else {}
        return {
            "control_mode": "flat_always_on",
            "bottom": {
                agent_type: stream.summary()
                for agent_type, stream in self.bottom.items()
            },
            "team_reward_sum": float(sum(float(item.get("team_reward", 0.0)) for item in self.step_infos)),
            "steps": len(self.step_infos),
            "terminal": bool(
                self.step_infos and self.step_infos[-1].get("done", False)
            ),
            "done_reason": str(final_info.get("done_reason", "none")),
            "episode_result": str(final_info.get("episode_result", "")),
            "final_score_raw": float(final_info.get("final_score_raw", 0.0)),
            "final_score_norm": float(final_info.get("final_score_norm", 0.0)),
            "final_score_unit_count": int(
                final_info.get("final_score_unit_count", 0)
            ),
            "final_score_sim_time": float(
                final_info.get("final_score_sim_time", 0.0)
            ),
            "attack_target_selections": [
                dict(event)
                for step_info in self.step_infos
                for event in step_info.get("bottom_action_events", {}).get("attack", [])
                if str(event.get("type", "")) == "attack_target_selected"
            ],
            "combat_events": [
                dict(event)
                for step_info in self.step_infos
                for event in step_info.get("events", [])
                if str(event.get("type", "")) == "target_destroyed"
            ],
        }


class RuleDrivenRolloutCollector(object):
    """Collect HAPPO samples from permanent team contexts.

    The policy set is the bottom HAPPO trainer itself. No high-level state,
    action, policy, value, transition, or checkpoint field is produced.
    """

    def __init__(
        self,
        api,
        bottom_agent_types=BOTTOM_AGENT_TYPES,
        bottom_global_reward_weight=0.1,
        bottom_global_reward_clip=10.0,
        adaptive_decision_timing=False,
        adaptive_timing_gain=0.7,
        adaptive_min_decision_seconds=0.2,
        adaptive_max_decision_seconds=2.0,
    ):
        self.api = api
        self.bottom_agent_types = tuple(bottom_agent_types)
        self.bottom_global_reward_weight = float(bottom_global_reward_weight)
        self.bottom_global_reward_clip = max(
            0.0, float(bottom_global_reward_clip)
        )
        self.adaptive_decision_timing = bool(adaptive_decision_timing)
        self.adaptive_timing_gain = float(adaptive_timing_gain)
        self.adaptive_min_decision_seconds = float(adaptive_min_decision_seconds)
        self.adaptive_max_decision_seconds = float(adaptive_max_decision_seconds)
        specs = api.get_bottom_agent_specs()
        self.action_names = {
            agent_type: self._build_action_name_map(
                specs.get(agent_type, {}).get("action_table", [])
            )
            for agent_type in self.bottom_agent_types
        }

    @staticmethod
    def _build_action_name_map(action_table):
        names = {}
        for index, action in enumerate(action_table or []):
            action_id = int(action.get("id", index))
            names[action_id] = str(action.get("name", action_id))
        return names

    def _action_name(self, agent_type, action_id):
        return self.action_names.get(agent_type, {}).get(
            int(action_id), str(action_id)
        )

    def _negative_rewards_enabled(self):
        return bool(getattr(self.api.env, "negative_rewards_enabled", False))

    def _apply_negative_reward_switch(self, value):
        value = float(value)
        if value < 0.0 and not self._negative_rewards_enabled():
            return 0.0
        return value

    @staticmethod
    def _environment_action(agent_type, action):
        """Preserve continuous vectors and unwrap scalar discrete actions."""
        if agent_type in ("recon", "landing"):
            return np.asarray(action, dtype=np.float32).reshape(-1).copy()
        values = np.asarray(action).reshape(-1)
        if values.size != 1:
            raise ValueError(
                "{0} requires one discrete action id, got shape {1}".format(
                    agent_type, np.asarray(action).shape
                )
            )
        return int(values[0])
    @staticmethod
    def _bottom_value(policy_set, agent_type, entity_name, state, global_state):
        value_fn = getattr(policy_set, "value_bottom", None)
        if not callable(value_fn):
            return 0.0
        return float(value_fn(
            agent_type, entity_name, state, global_state=global_state
        ))

    def collect(self, policy_set, n_steps, reset=True, decision_wall_budget=None):
        if reset:
            self.api.reset_flat()
        else:
            self.api.initialize_bottom_teams()
        rollout = BottomOnlyRollout(self.bottom_agent_types)
        done = False

        for step_id in range(int(n_steps)):
            step_wall_started = time.monotonic()
            sim_time_before = self.api.env._current_sim_time()
            # AFSIM stops emitting DecisionReady after its fixed mission
            # horizon. Detect that terminal boundary before selecting actions
            # or resuming a scenario that has already ended.
            if done or self.api.env.is_done():
                done = True
                break
            self.api.initialize_bottom_teams()
            critic_state = self.api.get_critic_global_state()
            global_state = np.asarray(
                critic_state["obs"], dtype=np.float32
            ).copy()

            action_phase_started = time.monotonic()
            bottom_snapshots = {}
            bottom_decisions = {}
            bottom_action_results = {}
            bottom_action_events = {}
            for agent_type in self.bottom_agent_types:
                state = self.api.get_persistent_agent_state(agent_type)
                bottom_snapshots[agent_type] = state
                decisions = {}
                actions = {}
                for entity_name, agent_state in state.get("agents", {}).items():
                    decision = policy_set.act_bottom(
                        agent_type,
                        entity_name,
                        agent_state,
                        global_state=global_state,
                    )
                    decisions[entity_name] = decision
                    actions[entity_name] = self._environment_action(
                        agent_type, decision["action"]
                    )
                bottom_decisions[agent_type] = decisions
                _state, _reward, _done, info = self.api.step_persistent_agents(
                    agent_type, actions, advance_sim=False
                )
                bottom_action_results[agent_type] = dict(
                    info.get("actions", {})
                )
                bottom_action_events[agent_type] = list(info.get("events", []))

            action_phase_finished = time.monotonic()

            # Keep the entire action-to-observation cycle within one wall-clock
            # decision budget. This compensates for variable policy/action
            # dispatch cost instead of always adding it to the receive wait.
            previous_decision_seconds = None
            if decision_wall_budget is not None:
                previous_decision_seconds = float(self.api.env.decision_seconds)
                elapsed_action_wall = time.monotonic() - step_wall_started
                self.api.env.decision_seconds = max(
                    0.0, float(decision_wall_budget) - elapsed_action_wall
                )
            try:
                team_reward, done, environment_info = self.api.step_flat()
            finally:
                if previous_decision_seconds is not None:
                    self.api.env.decision_seconds = previous_decision_seconds
            simulation_phase_finished = time.monotonic()
            sim_time_after = self.api.env._current_sim_time()
            sim_progress_seconds = max(0.0, sim_time_after - sim_time_before)
            next_decision_seconds = float(self.api.env.decision_seconds)
            if self.adaptive_decision_timing and sim_progress_seconds > 0.0:
                target = float(getattr(self.api.env, "decision_sim_seconds", 72.0))
                rate = max(1.0, float(getattr(self.api.env, "simulation_clock_rate", 1.0)))
                correction = self.adaptive_timing_gain * (sim_progress_seconds - target) / rate
                next_decision_seconds = float(np.clip(next_decision_seconds - correction, self.adaptive_min_decision_seconds, self.adaptive_max_decision_seconds))
                self.api.env.decision_seconds = next_decision_seconds
            team_reward = self._apply_negative_reward_switch(team_reward)
            simulation_events = list(environment_info.get("events", []))
            simulation_reward_details = list(
                environment_info.get("reward_details", [])
            )
            next_global_state = np.asarray(
                self.api.get_critic_global_state()["obs"], dtype=np.float32
            ).copy()
            next_bottom_states = {
                agent_type: self.api.get_persistent_agent_state(agent_type)
                for agent_type in self.bottom_agent_types
            }

            bottom_reward_infos = {}
            all_local_sums = {}
            all_weights = {}
            for agent_type in self.bottom_agent_types:
                state = bottom_snapshots[agent_type]
                reward_events = (
                    bottom_action_events[agent_type] + simulation_events
                )
                info = self.api.build_post_step_bottom_reward_info(
                    agent_type,
                    state,
                    next_bottom_states[agent_type],
                    bottom_action_results[agent_type],
                    float(team_reward),
                    reward_events=reward_events,
                    reward_details=simulation_reward_details,
                )
                bottom_reward_infos[agent_type] = info
                all_local_sums[agent_type] = sum(
                    float(v) for v in info.get("local_agent_rewards", {}).values()
                )
                all_weights[agent_type] = float(info.get("local_reward_weight", 0.0))

            global_total_local = sum(
                all_weights[t] * all_local_sums[t] for t in self.bottom_agent_types
            )
            global_shared_reward = self._apply_negative_reward_switch(
                float(team_reward) + global_total_local
            )

            for agent_type in self.bottom_agent_types:
                state = bottom_snapshots[agent_type]
                info = bottom_reward_infos[agent_type]
                contribution_rewards = info.get(
                    "target_contribution_rewards", {}
                )
                local_rewards = info.get("local_agent_rewards", {})
                weight = all_weights[agent_type]
                for entity_name, agent_state in state.get("agents", {}).items():
                    decision = bottom_decisions[agent_type][entity_name]
                    action_result = dict(
                        bottom_action_results[agent_type].get(entity_name, {})
                    )
                    contribution_reward = float(
                        contribution_rewards.get(entity_name, 0.0)
                    )
                    local_reward = float(local_rewards.get(entity_name, 0.0))
                    shaped_reward = global_shared_reward
                    next_agent_state = next_bottom_states[agent_type]["agents"][entity_name]
                    group_id = str(
                        agent_state.get("group_id")
                        or action_result.get("group_id") or ""
                    )
                    next_group_id = str(next_agent_state.get("group_id") or "")
                    task_done = bool(done)
                    next_value = 0.0 if task_done else self._bottom_value(
                        policy_set,
                        agent_type,
                        entity_name,
                        next_agent_state,
                        next_global_state,
                    )
                    environment_action = self._environment_action(
                        agent_type, decision["action"]
                    )
                    rollout.bottom[agent_type].append(
                        obs=np.asarray(agent_state["obs"], dtype=np.float32).copy(),
                        next_obs=np.asarray(next_agent_state["obs"], dtype=np.float32).copy(),
                        action_mask=np.asarray(agent_state["action_mask"], dtype=np.float32).copy(),
                        action=environment_action,
                        logprob=float(decision.get("logprob", 0.0)),
                        value=float(decision.get("value", 0.0)),
                        next_value=next_value,
                        reward=shaped_reward,
                        team_reward=float(team_reward),
                        shared_reward=float(global_shared_reward),
                        local_reward=float(local_reward),
                        weighted_local_reward=float(weight * local_reward),
                        target_contribution_reward=contribution_reward,
                        weighted_target_contribution_reward=float(weight * contribution_reward),
                        done=float(task_done), task_done=float(task_done),
                        global_state=global_state,

                        entity_name=entity_name, agent_type=agent_type,
                        action_name=("CONTINUOUS_MOVE" if agent_type in ("recon", "landing") else self._action_name(agent_type, environment_action)),
                        requested_action_id=(-1 if agent_type in ("recon", "landing") else int(action_result.get("requested_action_id", environment_action))),
                        executed_action_id=(-1 if agent_type in ("recon", "landing") else int(action_result.get("executed_action_id", environment_action))),
                        requested_sent=float(bool(action_result.get("requested_sent", False))),
                        sent=float(bool(action_result.get("sent", False))),
                        fallback_to_hold=float(bool(action_result.get("fallback_to_hold", False))),
                        group_id=group_id or None,
                        next_group_id=next_group_id or None,
                        task_type=agent_type.upper(),
                        step_id=step_id,
                        assigned=True,
                    )

            rollout.step_infos.append({
                "step_id": step_id,
                "team_reward": float(team_reward),
                "done": bool(done),
                "control_mode": "flat_always_on",
                "environment": dict(environment_info),
                "bottom_actions": bottom_action_results,
                "bottom_action_events": bottom_action_events,
                "timing": {
                    "action_inference_and_dispatch_wall_seconds": action_phase_finished - action_phase_started,
                    "simulation_wait_wall_seconds": simulation_phase_finished - action_phase_finished,
                    "postprocess_wall_seconds": time.monotonic() - simulation_phase_finished,
                    "total_wall_seconds": time.monotonic() - step_wall_started,
                    "sim_progress_seconds": sim_progress_seconds,
                    "next_decision_seconds": next_decision_seconds,
                },
                "events": simulation_events,
                "reward_details": simulation_reward_details,
                "bottom_rewards": {
                    agent_type: {
                        "local_agent_rewards": dict(
                            info.get("local_agent_rewards", {})
                        ),
                        "local_reward_details": list(
                            info.get("local_reward_details", [])
                        ),
                    }
                    for agent_type, info in bottom_reward_infos.items()
                },
            })
        return rollout
