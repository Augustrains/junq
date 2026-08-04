"""Training-facing interfaces for the AFSIM island environment.

This module keeps RL code away from controller-specific details.  The main
environment remains responsible for AFSIM UDP traffic and scenario state; this
wrapper exposes stable state/action/reward views for bottom-level agents.
"""

from typing import Dict, Iterable, Mapping, Optional, Tuple, Union

import numpy as np


ActionInput = Union[int, Iterable[float], Mapping[str, object]]


class AFSIMRLInterface(object):
    """Unified interface used by RL training code.

    Agent types:
      - recon: always-on per-aircraft movement inside a static team context.
      - attack: always-on per-aircraft target/return actions inside a static team context.
      - landing: per-transport tactical actions inside an active landing group.
      - ground: per-ground-unit tactical actions inside an active ground group.

    Reward convention:
      The default reward is shared team reward from env.compute_reward().  Local
      task states and reward details are exposed for debugging or optional
      shaping, but cooperative red-side training should optimize the shared
      reward first.
    """

    AGENT_TYPES = ("recon", "attack", "landing", "ground")
    LOCAL_REWARD_WEIGHT = 0.2
    LOCAL_REWARD_WEIGHTS = {
        "recon": 0.2,
        "attack": 0.3,
        "landing": 0.2,
        "ground": 0.2,
    }
    POTENTIAL_GAMMA = 0.99

    def __init__(self, env, reward_profile="default"):
        self.env = env
        self.reward_profile = str(reward_profile or "default")

    def reset(self) -> Dict[str, object]:
        return self.reset_flat()

    def reset_flat(self) -> Dict[str, object]:
        """Reset the flat controller and create permanent team contexts."""
        self.env.reset()
        teams = self.env.initialize_bottom_teams()
        return {
            "control_mode": "flat_always_on",
            "teams": teams,
            "agents": {
                agent_type: self.get_persistent_agent_state(agent_type)
                for agent_type in ("recon", "attack", "landing", "ground")
            },
        }

    def reset_rule_driven(self) -> Dict[str, object]:
        """Compatibility wrapper for older training entry points."""
        return self.reset_flat()

    def initialize_bottom_teams(self):
        return self.env.initialize_bottom_teams()
    def step_flat(self) -> Tuple[float, bool, dict]:
        """Advance one flat multi-agent decision interval."""
        reward, done, info = self.env.step_flat()
        return self._team_reward_for_profile(reward), bool(done), info

    def step_rule_driven(self) -> Tuple[float, bool, dict]:
        """Compatibility wrapper for older training entry points."""
        return self.step_flat()

    def get_bottom_agent_specs(self) -> Dict[str, dict]:
        return self.get_agent_specs()

    def get_agent_specs(self) -> Dict[str, dict]:
        """Return observation/action dimensions and field/action metadata."""
        specs = {
            "recon": {
                "assignment_mode": "always_on_team",
                "teams": [
                    list(team) for team in self.env.recon_controller.fixed_recon_teams
                ],
                "reward_mode": "shared_team_plus_local_shaping",
                "obs_dim": len(self.env.recon_state_fields),
                "action_dim": int(self.env.recon_controller.continuous_action_dim),
                "action_type": "continuous",
                "action_shape": (int(self.env.recon_controller.continuous_action_dim),),
                "fields": list(self.env.recon_state_fields),
                "action_table": self.env.get_recon_action_table(),
            },
            "attack": {
                "assignment_mode": "always_on_team",
                "teams": [
                    list(team) for team in self.env.attack_controller.fixed_attack_teams
                ],
                "reward_mode": "shared_team_plus_local_shaping",
                "obs_dim": len(self.env.attack_state_fields),
                "action_dim": len(self.env.get_attack_action_table()),
                "fields": list(self.env.attack_state_fields),
                "action_table": self.env.get_attack_action_table(),
            },
            "landing": {
                "reward_mode": "shared_team_plus_local_shaping",
                "obs_dim": len(self.env.landing_state_fields),
                "action_dim": int(self.env.landing_controller.continuous_action_dim),
                "action_type": "continuous",
                "action_shape": (int(self.env.landing_controller.continuous_action_dim),),
                "fields": list(self.env.landing_state_fields),
                "action_table": self.env.get_landing_action_table(),
            },
            "ground": {
                "reward_mode": "shared_team_plus_local_shaping",
                "obs_dim": len(self.env.ground_state_fields),
                "action_dim": len(self.env.get_ground_action_table()),
                "fields": list(self.env.ground_state_fields),
                "action_table": self.env.get_ground_action_table(),
            },
        }
        global_fields = self.env.get_critic_global_state_fields()
        global_dim = len(global_fields)
        entity_count = len(self.env.platforms)
        entity_feature_dim = len(self.env.CRITIC_ENTITY_FEATURES)
        for agent_type in ("recon", "attack", "landing", "ground"):
            local_fields = list(specs[agent_type]["fields"])
            entity_names = list(self._persistent_entity_names(agent_type))
            specs[agent_type].update({
                "actor_state_scope": "self_task_team_and_red_known_enemy",
                "critic_state_scope": "omniscient_all_entities_plus_local_context",
                "critic_global_state_dim": global_dim,
                "critic_local_state_dim": len(local_fields),
                "critic_state_dim": global_dim + len(local_fields),
                "critic_state_fields": global_fields + [
                    "local." + field for field in local_fields
                ],
                "critic_entity_count": entity_count,
                "critic_entity_feature_dim": entity_feature_dim,
                "num_entities": len(entity_names),
                "entity_names": entity_names,
            })
        specs["global"] = {
            "scope": "omniscient_all_entities",
            "obs_dim": global_dim,
            "fields": global_fields,
        }
        return specs

    def get_critic_global_state(self) -> Dict[str, object]:
        """Return the omniscient fixed-slot state used only by centralized critics."""
        obs = self.env.get_critic_global_state()
        return {
            "scope": "omniscient_all_entities",
            "obs": obs,
            "fields": self.env.get_critic_global_state_fields(),
        }

    def get_agent_state(self, agent_type: str, group_id: Optional[str] = None) -> Dict[str, object]:
        """Return the numeric state and action mask for one agent type.

        For task agents, pass a group_id to read one active task. If group_id is
        omitted, the method returns all active groups of that type.
        """
        agent_type = self._normalize_agent_type(agent_type)
        if group_id is None:
            return {
                "agent_type": agent_type,
                "scope": "active_groups",
                "groups": {
                    active_group_id: self.get_agent_state(agent_type, active_group_id)
                    for active_group_id in self.get_active_group_ids(agent_type)
                },
            }

        state = self._task_state(agent_type, group_id)
        if state is None:
            return {
                "agent_type": agent_type,
                "scope": "task_group",
                "group_id": group_id,
                "available": False,
                "error": "group_not_found",
            }
        state = dict(state)
        state.update({
            "agent_type": agent_type,
            "scope": "task_group",
            "group_id": group_id,
            "available": True,
        })
        return state

    def step_task_agent(
        self,
        agent_type: str,
        group_id: str,
        actions: ActionInput,
        advance_sim: bool = True,
    ) -> Tuple[Dict[str, object], float, bool, dict]:
        """Apply one tactical action step for a task agent.

        actions can be either:
          - int: broadcast the same action to every entity in the fixed team.
          - dict: {entity_name: action_id} for per-unit actions.
        """
        agent_type = self._normalize_agent_type(agent_type)

        before_state = self.get_agent_state(agent_type, group_id)
        entity_names = self._entity_names(agent_type, before_state)
        action_map = self._normalize_action_map(actions, entity_names, agent_type)
        results = {}
        self.env.last_reward_events = []
        for entity_name, action_id in self._formation_ordered_action_items(agent_type, action_map):
            results[entity_name] = self._apply_action_with_hold_fallback(
                agent_type, group_id, entity_name, action_id
            )

        if advance_sim:
            self.env._drain_messages(timeout=self.env.decision_seconds)
            self.env.step_count += 1

        reward = self.env.compute_reward() if advance_sim else 0.0
        done = self.env.is_done()
        after_state = self.get_agent_state(agent_type, group_id)
        info = self._build_bottom_reward_info(
            agent_type,
            before_state,
            after_state,
            results,
            float(reward),
            {"agent_type": agent_type, "group_id": group_id},
        )
        return after_state, self._team_reward_for_profile(reward), bool(done), info

    def get_persistent_agent_state(self, agent_type: str) -> Dict[str, object]:
        """Return the fixed-team state for every configured entity."""
        agent_type = self._normalize_agent_type(agent_type)

        entities = self._persistent_entity_names(agent_type)
        active_groups = {
            group_id: self.get_agent_state(agent_type, group_id)
            for group_id in self.get_active_group_ids(agent_type)
        }

        agents = self._persistent_entity_states(agent_type, active_groups)
        missing = [name for name in entities if name not in agents]
        if missing:
            raise RuntimeError(
                "always-on team context missing for {0}: {1}".format(
                    agent_type, ", ".join(missing)
                )
            )

        return {
            "agent_type": agent_type,
            "scope": "persistent_agents",
            "agents": agents,
            "active_groups": active_groups,
            "task_contexts": self._persistent_task_contexts(agent_type, active_groups),
            "action_table": self._action_table_for(agent_type),
        }

    def step_persistent_agents(
        self,
        agent_type: str,
        actions: ActionInput,
        advance_sim: bool = True,
    ) -> Tuple[Dict[str, object], float, bool, dict]:
        """Apply one action for every configured entity in its fixed team."""
        agent_type = self._normalize_agent_type(agent_type)

        state = self.get_persistent_agent_state(agent_type)
        entity_names = list(state.get("agents", {}).keys())
        action_map = self._normalize_action_map(actions, entity_names, agent_type)
        group_by_entity = self._active_group_by_entity(agent_type)
        results = {}
        self.env.last_reward_events = []
        for entity_name, action in self._formation_ordered_action_items(agent_type, action_map):
            group_id = group_by_entity.get(entity_name)
            if group_id is None:
                raise RuntimeError(
                    "always-on entity has no fixed team: {0}".format(entity_name)
                )
            result = self._apply_action_with_hold_fallback(
                agent_type, group_id, entity_name, action
            )
            result["group_id"] = group_id
            results[entity_name] = result

        if advance_sim:
            self.env._drain_messages(timeout=self.env.decision_seconds)
            self.env.step_count += 1

        reward = self.env.compute_reward() if advance_sim else 0.0
        done = self.env.is_done()
        after_state = self.get_persistent_agent_state(agent_type)
        info = self._build_bottom_reward_info(
            agent_type,
            state,
            after_state,
            results,
            float(reward),
            {"agent_type": agent_type, "scope": "persistent_agents"},
        )
        return after_state, self._team_reward_for_profile(reward), bool(done), info

    def _formation_ordered_action_items(self, agent_type, action_map):
        """Apply leader actions before contextual follower HOLD actions."""
        ordered = []
        seen = set()
        controller = {
            "recon": self.env.recon_controller,
            "attack": self.env.attack_controller,
        }.get(agent_type)
        if controller is not None:
            for group in controller.active_groups.values():
                for platform in group.platforms:
                    if platform.name in action_map and platform.name not in seen:
                        ordered.append((platform.name, action_map[platform.name]))
                        seen.add(platform.name)
        ordered.extend((name, action) for name, action in action_map.items() if name not in seen)
        return ordered

    def build_post_step_bottom_reward_info(
        self,
        agent_type: str,
        before_state: Mapping[str, object],
        after_state: Mapping[str, object],
        action_results: Mapping[str, dict],
        team_reward: float,
        reward_events=None,
        reward_details=None,
    ) -> dict:
        """Attribute one completed simulation step to the bottom actions that caused it."""
        return self._build_bottom_reward_info(
            agent_type,
            before_state,
            after_state,
            action_results,
            team_reward,
            {"agent_type": agent_type, "scope": "persistent_agents", "post_step": True},
            reward_events=reward_events,
            reward_details=reward_details,
        )

    def _build_bottom_reward_info(
        self,
        agent_type: str,
        before_state: Mapping[str, object],
        after_state: Mapping[str, object],
        action_results: Mapping[str, dict],
        team_reward: float,
        base_info: Mapping[str, object],
        reward_events=None,
        reward_details=None,
    ) -> dict:
        events = list(self.env.last_reward_events if reward_events is None else reward_events)
        details_snapshot = list(self.env.last_reward_details if reward_details is None else reward_details)
        team_reward = self._team_reward_for_profile(team_reward)
        local_rewards, local_details = self._compute_local_agent_rewards(
            agent_type,
            before_state,
            after_state,
            action_results,
            reward_events=events,
        )
        local_reward_weight = self._local_reward_weight(agent_type)
        total_local = sum(float(v) for v in local_rewards.values())
        shared_reward = float(team_reward) + local_reward_weight * total_local
        agent_rewards = {name: shared_reward for name in local_rewards}
        info = dict(base_info)
        info.update({
            "actions": dict(action_results),
            "events": events,
            "reward_details": details_snapshot,
            "reward_mode": "shared_team_plus_local_shaping",
            "team_reward": float(team_reward),
            "local_reward_weight": local_reward_weight,
            "local_agent_rewards": local_rewards,
            "agent_rewards": agent_rewards,
            "local_reward_details": local_details,
        })
        contribution_rewards = {name: 0.0 for name in local_rewards}
        for detail in local_details:
            if str(detail.get("type", "")).startswith("contributed_target_"):
                name = str(detail.get("agent", ""))
                contribution_rewards[name] = (
                    contribution_rewards.get(name, 0.0)
                    + float(detail.get("value", 0.0))
                )
        info["target_contribution_rewards"] = contribution_rewards
        return info

    def _compute_local_agent_rewards(
        self,
        agent_type: str,
        before_state: Mapping[str, object],
        after_state: Mapping[str, object],
        action_results: Mapping[str, dict],
        reward_events=None,
    ) -> Tuple[Dict[str, float], list]:
        if agent_type in ("recon", "attack"):
            return self._compute_recon_attack_local_rewards(
                agent_type, before_state, after_state, action_results,
                reward_events=reward_events,
            )

        before_entities = self._collect_entity_states(agent_type, before_state)
        after_entities = self._collect_entity_states(agent_type, after_state)
        entity_names = set(before_entities) | set(after_entities) | set(action_results)
        local_rewards = {name: 0.0 for name in sorted(entity_names)}
        details = []
        if self.reward_profile == "recon_attack_stage":
            return local_rewards, details
        events = list(self.env.last_reward_events if reward_events is None else reward_events)
        for event in events:
            entity = event.get("actor") or event.get("shooter") or event.get("platform") or event.get("entity")
            if entity in local_rewards:
                self._add_local_detail(
                    local_rewards, details, entity, str(event.get("type", "")),
                    self._event_local_reward(event), event=event,
                )
        return local_rewards, details

    def _compute_recon_attack_local_rewards(
        self,
        agent_type: str,
        before_state: Mapping[str, object],
        after_state: Mapping[str, object],
        action_results: Mapping[str, dict],
        reward_events=None,
    ) -> Tuple[Dict[str, float], list]:
        before_entities = self._collect_entity_states(agent_type, before_state)
        after_entities = self._collect_entity_states(agent_type, after_state)
        entity_names = set(before_entities) | set(after_entities) | set(action_results)
        local_rewards = {name: 0.0 for name in sorted(entity_names)}
        details = []
        events = list(self.env.last_reward_events if reward_events is None else reward_events)

        register = getattr(self.env, "register_target_reward_contributor", None)
        if callable(register):
            for event in events:
                event_type = str(event.get("type", ""))
                target_name = str(event.get("target") or event.get("platform") or "")
                if event_type == "recon_task_detection":
                    register(target_name, event.get("observer", ""), "discover")
                elif event_type in ("new_attack_detection", "attack_target_detection"):
                    register(
                        target_name,
                        event.get("observer") or event.get("actor") or event.get("aircraft") or event.get("platform") or "",
                        "discover",
                    )
                elif event_type in ("attack_result", "weapon_fired"):
                    register(
                        target_name,
                        event.get("actor") or event.get("shooter") or event.get("aircraft") or event.get("platform") or "",
                        "attack",
                    )

        groups = {}
        for name in sorted(local_rewards):
            before_entity = before_entities.get(name, {})
            after_entity = after_entities.get(name, {})
            before_group = str(before_entity.get("group_id") or "")
            after_group = str(after_entity.get("group_id") or "")
            if before_group and before_group == after_group:
                fixed_team_id = self.env.reward_team_id_for_entity(agent_type, name)
                if fixed_team_id:
                    groups.setdefault((before_group, fixed_team_id), []).append(name)
        for (group_id, fixed_team_id), members in sorted(groups.items()):
            del group_id
            target = self.env.get_reward_target_for_team(agent_type, fixed_team_id)
            if not target:
                continue
            samples = []
            for name in members:
                before_pos = self._position_from_normalized_obs(
                    self._obs_by_name(before_entities[name])
                )
                after_pos = self._position_from_normalized_obs(
                    self._obs_by_name(after_entities[name])
                )
                if before_pos is None or after_pos is None:
                    continue
                target_lat = float(target["lat"])
                target_lon = float(target["lon"])
                before_distance, _ = self.env._distance_and_bearing(
                    before_pos[0], before_pos[1], target_lat, target_lon
                )
                after_distance, _ = self.env._distance_and_bearing(
                    after_pos[0], after_pos[1], target_lat, target_lon
                )
                samples.append((float(before_distance), float(after_distance)))
            if not samples:
                continue
            normalization = getattr(
                self.env, agent_type + "_state_config", {}
            ).get("normalization", {})
            max_distance = max(1.0, float(normalization.get("max_distance_m", 400000.0)))
            before_mean = sum(item[0] for item in samples) / len(samples)
            after_mean = sum(item[1] for item in samples) / len(samples)
            progress = max(0.0, (before_mean - after_mean) / max_distance)
            progress *= float(self.env.reward_movement_scale())
            if progress <= 0.0:
                continue
            for name in members:
                self._add_local_detail(
                    local_rewards,
                    details,
                    name,
                    "group_move_toward_{0}_target".format(agent_type),
                    progress,
                )
                details[-1].update({
                    "reward_target": target["name"],
                    "fixed_team_id": fixed_team_id,
                    "effective_priority": float(target.get("effective_priority", 0.0)),
                })
        contributors_for = getattr(self.env, "get_target_reward_contributors", None)
        if callable(contributors_for):
            for event in events:
                event_type = str(event.get("type", ""))
                if event_type not in ("damage_dealt", "target_destroyed"):
                    continue
                target_name = str(event.get("platform") or event.get("target") or "")
                for name in contributors_for(target_name).get("all", []):
                    if name not in local_rewards:
                        continue
                    value, reward_type = self._combat_positive_reward(event, name)
                    if value > 0.0:
                        self._add_local_detail(
                            local_rewards, details, name, reward_type, value, event=event
                        )

        if agent_type == "attack":
            for name in sorted(local_rewards):
                before_obs = self._obs_by_name(before_entities.get(name, {}))
                after_obs = self._obs_by_name(after_entities.get(name, {}))
                ammo_before = float(before_obs.get("aam_count_norm", 0.0)) + float(before_obs.get("agm_count_norm", 0.0))
                ammo_after = float(after_obs.get("aam_count_norm", 0.0)) + float(after_obs.get("agm_count_norm", 0.0))
                if (
                    float(before_obs.get("rearming", 0.0)) > 0.5
                    and float(after_obs.get("rearming", 0.0)) <= 0.5
                    and ammo_after > ammo_before
                ):
                    self._add_local_detail(
                        local_rewards, details, name, "rearm_completed", ammo_after - ammo_before
                    )
        return local_rewards, details
    def _combat_positive_reward(self, event: Mapping[str, object], contributor_name: str) -> Tuple[float, str]:
        """Return positive damage/kill credit for the direct attacker or recon contributors."""
        target_name = str(event.get("platform") or event.get("target") or "")
        target = self.env.platforms.get(target_name)
        target_role = str(event.get("role") or getattr(target, "role", ""))
        rules = getattr(getattr(self.env, "reward_manager", None), "rules", {})
        config = dict(rules.get("combat_positive_rewards", {}))
        role_values = dict(config.get("by_target_role", {}).get(target_role, {}))
        event_type = str(event.get("type", ""))
        if event_type == "damage_dealt":
            amount = max(0.0, float(event.get("amount", 0.0)))
            max_hp = float(event.get("max_hp", 0.0))
            value = float(role_values.get("damage_per_full_hp", 0.0)) * (
                amount / max_hp if max_hp > 0.0 else amount
            )
            suffix = "hp_loss"
        elif event_type == "target_destroyed":
            value = float(role_values.get("destroyed", 0.0))
            suffix = "destroyed"
        else:
            return 0.0, ""
        if value <= 0.0:
            return 0.0, ""

        contributor = self.env.platforms.get(str(contributor_name))
        contributor_role = str(getattr(contributor, "role", ""))
        direct_actor = str(event.get("actor") or event.get("shooter") or "")
        if contributor_role == "attack_aircraft" and str(contributor_name) == direct_actor:
            return value, "direct_target_{0}_{1}".format(target_role, suffix)
        if contributor_role == "recon_aircraft":
            fraction = max(0.0, float(config.get("recon_contributor_fraction", 0.0)))
            return value * fraction, "recon_contributed_target_{0}_{1}".format(target_role, suffix)
        return 0.0, ""
    def _team_reward_for_profile(self, value: float) -> float:
        return self._apply_negative_reward_switch(value)


    @property
    def negative_rewards_enabled(self) -> bool:
        return bool(getattr(self.env, "negative_rewards_enabled", False))

    def _apply_negative_reward_switch(self, value: float) -> float:
        value = float(value)
        if value < 0.0 and not self.negative_rewards_enabled:
            return 0.0
        return value

    def _local_reward_weight(self, agent_type: str) -> float:
        return float(self.LOCAL_REWARD_WEIGHTS.get(agent_type, self.LOCAL_REWARD_WEIGHT))

    def _collect_entity_states(self, agent_type: str, state: Mapping[str, object]) -> Dict[str, dict]:
        if not isinstance(state, Mapping):
            return {}
        if state.get("scope") == "persistent_agents":
            return {name: dict(value) for name, value in state.get("agents", {}).items()}
        container = self._state_container_name(agent_type)
        return {name: dict(value) for name, value in state.get(container, {}).items()}

    @staticmethod
    def _obs_by_name(entity_state: Mapping[str, object]) -> Mapping[str, float]:
        obs = entity_state.get("obs_by_name", {}) if isinstance(entity_state, Mapping) else {}
        return obs if isinstance(obs, Mapping) else {}

    def _position_from_normalized_obs(self, obs):
        if "lat_norm" not in obs or "lon_norm" not in obs:
            return None
        bounds = self.env.bounds
        lat_min = float(bounds.get("lat_min", 23.0))
        lat_max = float(bounds.get("lat_max", 26.0))
        lon_min = float(bounds.get("lon_min", 119.0))
        lon_max = float(bounds.get("lon_max", 123.0))
        return (
            lat_min + float(obs["lat_norm"]) * (lat_max - lat_min),
            lon_min + float(obs["lon_norm"]) * (lon_max - lon_min),
        )

    def _add_progress_delta(
        self,
        rewards: Dict[str, float],
        details: list,
        entity_name: str,
        before_obs: Mapping[str, float],
        after_obs: Mapping[str, float],
        field: str,
        scale: float,
        increasing: bool,
    ):
        if field not in before_obs or field not in after_obs:
            return
        self._add_numeric_progress(
            rewards,
            details,
            entity_name,
            field,
            float(before_obs.get(field, 0.0)),
            float(after_obs.get(field, 0.0)),
            scale,
            increasing,
        )

    def _add_numeric_progress(
        self,
        rewards: Dict[str, float],
        details: list,
        entity_name: str,
        field: str,
        before_value: float,
        after_value: float,
        scale: float,
        increasing: bool,
    ):
        before_phi = float(before_value) if increasing else 1.0 - float(before_value)
        after_phi = float(after_value) if increasing else 1.0 - float(after_value)
        value = (self.POTENTIAL_GAMMA * after_phi - before_phi) * float(scale)
        self._add_local_detail(rewards, details, entity_name, field + "_potential", value)

    def _add_transition_delta(
        self,
        rewards: Dict[str, float],
        details: list,
        entity_name: str,
        before_obs: Mapping[str, float],
        after_obs: Mapping[str, float],
        field: str,
        scale: float,
    ):
        if field not in before_obs or field not in after_obs:
            return
        delta = float(after_obs.get(field, 0.0)) - float(before_obs.get(field, 0.0))
        self._add_local_detail(rewards, details, entity_name, field + "_transition", delta * float(scale))
    def _add_local_detail(self, rewards: Dict[str, float], details: list, entity_name: str, reward_type: str, value: float, event=None):
        raw_value = float(value)
        value = self._apply_negative_reward_switch(raw_value)
        if raw_value == 0.0:
            return
        rewards[entity_name] = float(rewards.get(entity_name, 0.0)) + value
        detail = {"agent": entity_name, "type": reward_type, "value": value}
        if raw_value < 0.0 and value == 0.0:
            detail.update({"raw_value": raw_value,
                           "negative_reward_suppressed": True})
        if event is not None:
            detail["event"] = dict(event)
        details.append(detail)

    def _event_local_reward(self, event: Mapping[str, object]) -> float:
        event_type = event.get("type", "")
        if event_type == "damage_dealt":
            amount = max(0.0, float(event.get("amount", 0.0)))
            max_hp = float(event.get("max_hp", 0.0))
            damage_ratio = amount / max_hp if max_hp > 0.0 else amount
            return 6.0 * damage_ratio
        if event_type == "attack_result":
            # Accepted fire and probability results are not confirmed damage.
            return 0.0
        if event_type == "target_destroyed":
            return 5.0
        if event_type == "attack_failed":
            return -0.5
        if event_type in (
            "attack_action_masked",
            "recon_action_masked",
            "attack_target_slot_empty",
            "ground_action_masked",
            "landing_action_masked",
            "retreat_not_ready",
        ):
            return -0.5
        if event_type == "red_loss":
            return -2.0
        if event_type in ("task_rejected", "platform_not_ready", "udp_not_ready"):
            return -0.5
        return 0.0

    def _action_spec_by_id(self, agent_type: str, action_id: int) -> dict:
        table = self._action_table_for(agent_type)
        for action in table:
            if int(action.get("id", -1)) == int(action_id):
                return dict(action)
        return {}
    def get_active_group_ids(self, agent_type: str) -> Iterable[str]:
        agent_type = self._normalize_agent_type(agent_type)
        if agent_type == "recon":
            return list(self.env.recon_controller.active_groups.keys())
        if agent_type == "attack":
            return list(self.env.attack_controller.active_groups.keys())
        if agent_type == "landing":
            return list(self.env.landing_controller.active_groups.keys())
        if agent_type == "ground":
            return list(self.env.ground_controller.active_groups.keys())
        return []

    def get_last_reward_details(self) -> list:
        """Return reward terms from the last completed step without side effects."""
        return list(self.env.last_reward_details)

    def _task_state(self, agent_type: str, group_id: str):
        if agent_type == "recon":
            return self.env.get_recon_task_state(group_id)
        if agent_type == "attack":
            return self.env.get_attack_task_state(group_id)
        if agent_type == "landing":
            return self.env.get_landing_task_state(group_id)
        if agent_type == "ground":
            return self.env.get_ground_task_state(group_id)
        return None

    @staticmethod
    def _continuous_recon_action(action):
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        if values.size != 3 or not np.all(np.isfinite(values)):
            return [0.0, 0.0, 0.0]
        return np.clip(values, -1.0, 1.0).astype(np.float32).tolist()

    def _apply_task_action(self, agent_type: str, group_id: str, entity_name: str, action) -> bool:
        if agent_type == "recon":
            return bool(self.env.apply_recon_aircraft_continuous_action(
                group_id, entity_name, self._continuous_recon_action(action)
            ))
        if agent_type == "landing":
            return bool(self.env.apply_landing_ship_continuous_action(
                group_id, entity_name, self._continuous_recon_action(action)
            ))
        action_id = int(action)
        if agent_type == "attack":
            return bool(self.env.apply_attack_aircraft_action(group_id, entity_name, action_id))
        if agent_type == "landing":
            return bool(self.env.apply_landing_ship_action(group_id, entity_name, action_id))
        if agent_type == "ground":
            return bool(self.env.apply_ground_unit_action(group_id, entity_name, action_id))
        return False

    def _apply_action_with_hold_fallback(
        self, agent_type: str, group_id: str, entity_name: str, requested_action
    ) -> dict:
        if agent_type in ("recon", "landing"):
            action = self._continuous_recon_action(requested_action)
            requested_sent = self._apply_task_action(agent_type, group_id, entity_name, action)
            fallback = [0.0, 0.0, 0.0]
            sent = requested_sent or self._apply_task_action(
                agent_type, group_id, entity_name, fallback
            )
            return {
                "action": action if requested_sent else fallback,
                "requested_action": action,
                "executed_action": action if requested_sent else fallback,
                "requested_sent": bool(requested_sent),
                "sent": bool(sent),
                "fallback_to_hold": bool(not requested_sent),
            }
        requested_action_id = int(requested_action)
        requested_sent = self._apply_task_action(
            agent_type, group_id, entity_name, requested_action_id
        )
        fallback_to_hold = requested_action_id != 0 and not requested_sent
        executed_action_id = 0 if fallback_to_hold else requested_action_id
        sent = (
            self._apply_task_action(agent_type, group_id, entity_name, 0)
            if fallback_to_hold
            else requested_sent
        )
        return {
            "action_id": int(executed_action_id),
            "requested_action_id": requested_action_id,
            "executed_action_id": int(executed_action_id),
            "requested_sent": bool(requested_sent),
            "sent": bool(sent),
            "fallback_to_hold": bool(fallback_to_hold),
        }

    def _persistent_entity_names(self, agent_type: str) -> Iterable[str]:
        red = self.env.config.get("red", {})
        if agent_type == "recon":
            return list(red.get("commandable_recon_aircraft", red.get("recon_aircraft", [])))
        if agent_type == "attack":
            return list(red.get("commandable_attack_aircraft", red.get("attack_aircraft", [])))
        if agent_type == "landing":
            return list(red.get("commandable_transports", red.get("transports", [])))
        if agent_type == "ground":
            return list(red.get("commandable_ground_forces", red.get("ground_forces", [])))
        return []

    def _persistent_entity_states(self, agent_type: str, active_groups: Mapping[str, object]) -> Dict[str, dict]:
        agents = {}
        container = self._state_container_name(agent_type)
        for group_id, group_state in active_groups.items():
            for entity_name, entity_state in group_state.get(container, {}).items():
                copied = dict(entity_state)
                copied["group_id"] = group_id
                copied["task_context"] = self._group_task_context(agent_type, group_state)
                agents[entity_name] = copied
        return agents

    def _active_group_by_entity(self, agent_type: str) -> Dict[str, str]:
        mapping = {}
        for group_id in self.get_active_group_ids(agent_type):
            state = self.get_agent_state(agent_type, group_id)
            container = self._state_container_name(agent_type)
            for entity_name, entity_state in state.get(container, {}).items():
                mapping[entity_name] = group_id
        return mapping

    def _fields_for(self, agent_type: str) -> list:
        if agent_type == "recon":
            return list(self.env.recon_state_fields)
        if agent_type == "attack":
            return list(self.env.attack_state_fields)
        if agent_type == "landing":
            return list(self.env.landing_state_fields)
        if agent_type == "ground":
            return list(self.env.ground_state_fields)
        return []

    def _action_table_for(self, agent_type: str) -> list:
        if agent_type == "recon":
            return self.env.get_recon_action_table()
        if agent_type == "attack":
            return self.env.get_attack_action_table()
        if agent_type == "landing":
            return self.env.get_landing_action_table()
        if agent_type == "ground":
            return self.env.get_ground_action_table()
        return []

    @staticmethod
    def _state_container_name(agent_type: str) -> str:
        if agent_type in ("recon", "attack"):
            return "aircraft"
        if agent_type == "landing":
            return "ships"
        if agent_type == "ground":
            return "units"
        return "agents"

    def _persistent_task_contexts(self, agent_type: str, active_groups: Mapping[str, object]) -> Dict[str, dict]:
        return {
            group_id: self._group_task_context(agent_type, group_state)
            for group_id, group_state in active_groups.items()
        }

    @staticmethod
    def _group_task_context(agent_type: str, group_state: Mapping[str, object]) -> dict:
        team = dict(group_state.get("team", group_state.get("task", {})))
        context = {"active": True, "team_type": agent_type.upper()}
        context.update(team)
        return context
    @staticmethod
    def _entity_names(agent_type: str, state: Mapping[str, object]) -> Iterable[str]:
        if agent_type in ("recon", "attack"):
            return list(state.get("aircraft", {}).keys())
        if agent_type == "landing":
            return list(state.get("ships", {}).keys())
        if agent_type == "ground":
            return list(state.get("units", {}).keys())
        return []

    @staticmethod
    def _normalize_action_map(actions: ActionInput, entity_names: Iterable[str], agent_type: str = "") -> Dict[str, object]:
        names = list(entity_names)
        if isinstance(actions, Mapping):
            if agent_type in ("recon", "landing"):
                return {name: actions[name] for name in names if name in actions}
            return {name: int(actions[name]) for name in names if name in actions}
        if agent_type in ("recon", "landing"):
            return {name: actions for name in names}
        return {name: int(actions) for name in names}

    def _normalize_agent_type(self, agent_type: str) -> str:
        normalized = str(agent_type).strip().lower()
        if normalized not in self.AGENT_TYPES:
            raise ValueError("unknown agent_type: {0}".format(agent_type))
        return normalized


def make_rl_interface(env):
    """Small factory kept for training scripts."""
    return AFSIMRLInterface(env)

