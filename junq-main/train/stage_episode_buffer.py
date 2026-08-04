"""Task-instance on-policy buffer for staged HAPPO training."""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np


FLOAT_KEYS = {
    "obs", "next_obs", "action_mask", "global_state", "reward", "value",
    "next_value", "logprob", "done", "assigned", "task_done",
    "next_assigned", "team_reward", "shared_reward", "local_reward",
    "weighted_local_reward", "target_contribution_reward",
    "weighted_target_contribution_reward", "global_stage_reward", "action",
}
INT_KEYS = {"step_id", "episode_index", "policy_version", "worker_id"}


class TaskTrajectoryBuffer:
    """Collect completed high-level task instances instead of full episodes.

    A task instance is identified by the managed scenario index, bottom policy
    type, and the high-level group id. Only transitions produced while an
    entity is assigned are retained. Natural task completion terminates GAE;
    scenario termination force-closes the tasks that are still active.
    """

    FORMAT = "task_trajectory_v1"

    def __init__(self, agent_types: Iterable[str]):
        self.agent_types = tuple(agent_types)
        self.clear()

    def append_rollout(self, rollout):
        local_ids = {
            int(row.get("step_id", 0))
            for agent_type in self.agent_types
            for row in rollout.bottom[agent_type].rows
        }
        id_map = {
            old: self.next_step_id + offset
            for offset, old in enumerate(sorted(local_ids))
        }
        self.next_step_id += len(id_map)

        touched = {name: set() for name in self.agent_types}
        for agent_type in self.agent_types:
            for source in rollout.bottom[agent_type].rows:
                if not bool(source.get("assigned", False)):
                    delayed = float(source.get("weighted_target_contribution_reward", 0.0))
                    if delayed != 0.0:
                        self._apply_delayed_entity_reward(
                            agent_type,
                            str(source.get("entity_name", "")),
                            delayed,
                            float(source.get("target_contribution_reward", 0.0)),
                        )
                    continue
                group_id = str(source.get("group_id") or "")
                if not group_id:
                    continue
                task_id = self._task_id(agent_type, group_id)
                task = self.active_tasks[agent_type].setdefault(
                    task_id,
                    {
                        "task_id": task_id,
                        "agent_type": agent_type,
                        "group_id": group_id,
                        "episode_index": self.episode_index,
                        "policy_version": self.policy_version,
                        "rows": [],
                        "active_entities": set(),
                        "seen_entities": set(),
                        "start_step": None,
                        "end_reason": "",
                    },
                )
                row = dict(source)
                row["step_id"] = id_map[int(row.get("step_id", 0))]
                row["task_id"] = task_id
                row["episode_index"] = self.episode_index
                row["policy_version"] = self.policy_version
                row["assigned"] = 1.0
                row["task_done"] = float(bool(row.get("task_done", row.get("done", False))))
                row["done"] = row["task_done"]
                if row["done"] > 0.5:
                    row["next_value"] = 0.0
                task["rows"].append(row)
                entity_name = str(row.get("entity_name", ""))
                task["seen_entities"].add(entity_name)
                if task["start_step"] is None:
                    task["start_step"] = int(row["step_id"])
                if row["done"] > 0.5:
                    task["active_entities"].discard(entity_name)
                else:
                    task["active_entities"].add(entity_name)
                touched[agent_type].add(task_id)

        for agent_type in self.agent_types:
            for task_id in list(touched[agent_type]):
                task = self.active_tasks[agent_type].get(task_id)
                if task and not task["active_entities"] and any(
                    float(row.get("done", 0.0)) > 0.5 for row in task["rows"]
                ):
                    task["end_reason"] = "task_completed"
                    self._commit_task(agent_type, task_id)
    def _apply_delayed_entity_reward(
        self, agent_type: str, entity_name: str, weighted_reward: float, raw_reward: float
    ):
        """Attach delayed target credit to the contributor's latest retained action."""
        candidates = []
        for row in self.rows[agent_type]:
            if (
                str(row.get("entity_name", "")) == entity_name
                and int(row.get("episode_index", self.episode_index)) == self.episode_index
            ):
                candidates.append(row)
        for task in self.active_tasks[agent_type].values():
            if int(task.get("episode_index", self.episode_index)) != self.episode_index:
                continue
            for row in task.get("rows", []):
                if str(row.get("entity_name", "")) == entity_name:
                    candidates.append(row)
        if not candidates:
            return False
        row = max(candidates, key=lambda item: int(item.get("step_id", -1)))
        row["reward"] = float(row.get("reward", 0.0)) + float(weighted_reward)
        row["local_reward"] = float(row.get("local_reward", 0.0)) + float(raw_reward)
        row["weighted_local_reward"] = float(row.get("weighted_local_reward", 0.0)) + float(weighted_reward)
        row["target_contribution_reward"] = float(row.get("target_contribution_reward", 0.0)) + float(raw_reward)
        row["weighted_target_contribution_reward"] = float(row.get("weighted_target_contribution_reward", 0.0)) + float(weighted_reward)
        task_id = str(row.get("task_id", ""))
        return True
    def apply_delayed_target_rewards(
        self, agent_type: str, rewards: Mapping[str, float], weight: float
    ):
        """Apply target rewards produced after the final action step."""
        for entity_name, raw_reward in rewards.items():
            raw_reward = float(raw_reward)
            if raw_reward != 0.0:
                self._apply_delayed_entity_reward(
                    agent_type,
                    str(entity_name),
                    float(weight) * raw_reward,
                    raw_reward,
                )

    def apply_episode_global_reward(self, reward: float):
        """Give one stage-boundary reward to every participating entity."""
        reward = float(reward)
        if reward == 0.0:
            return
        latest = {}
        for agent_type in self.agent_types:
            candidates = list(self.rows[agent_type])
            for task in self.active_tasks[agent_type].values():
                if int(task.get("episode_index", self.episode_index)) == self.episode_index:
                    candidates.extend(task.get("rows", []))
            for row in candidates:
                if int(row.get("episode_index", self.episode_index)) != self.episode_index:
                    continue
                entity_name = str(row.get("entity_name", ""))
                task_key = str(row.get("task_id", row.get("group_id", "")))
                key = (agent_type, task_key, entity_name)
                if not entity_name:
                    continue
                previous = latest.get(key)
                if previous is None or int(row.get("step_id", -1)) > int(previous.get("step_id", -1)):
                    latest[key] = row
        for row in latest.values():
            row["reward"] = float(row.get("reward", 0.0)) + reward
            row["global_stage_reward"] = float(row.get("global_stage_reward", 0.0)) + reward


    def finish_episode(self, terminal_reward_bonus: float = 0.0, end_reason: str = "episode_terminal"):
        """Force-close only tasks still active at the scenario boundary."""
        for agent_type in self.agent_types:
            for task_id, task in list(self.active_tasks[agent_type].items()):
                rows = task["rows"]
                if not rows:
                    self.active_tasks[agent_type].pop(task_id, None)
                    continue
                active_entities = set(task["active_entities"])
                last_by_entity = {}
                for index, row in enumerate(rows):
                    entity_name = str(row.get("entity_name", ""))
                    if entity_name in active_entities:
                        last_by_entity[entity_name] = index
                for index in last_by_entity.values():
                    rows[index]["done"] = 1.0
                    rows[index]["task_done"] = 1.0
                    rows[index]["next_value"] = 0.0
                    rows[index]["reward"] = (
                        float(rows[index].get("reward", 0.0))
                        + float(terminal_reward_bonus)
                    )
                task["active_entities"].clear()
                task["end_reason"] = str(end_reason)
                self._commit_task(agent_type, task_id)
        self.episodes += 1
        self.episode_index += 1

    def _task_id(self, agent_type: str, group_id: str) -> str:
        return "episode_{0}:{1}:{2}".format(
            int(self.episode_index), str(agent_type), str(group_id)
        )

    def _commit_task(self, agent_type: str, task_id: str):
        task = self.active_tasks[agent_type].pop(task_id)
        rows = list(task.get("rows", []))
        if not rows:
            return
        self.rows[agent_type].extend(rows)

    def assigned_counts(self):
        return {name: len(rows) for name, rows in self.rows.items()}

    def active_counts(self):
        return {
            name: sum(len(task.get("rows", [])) for task in tasks.values())
            for name, tasks in self.active_tasks.items()
        }

    def ready(self, minimum_steps: Mapping[str, int]):
        steps = self.assigned_counts()
        return all(
            steps.get(name, 0) >= int(required)
            for name, required in minimum_steps.items()
        )

    def merge_completed_from(self, other, worker_id=0):
        """Move completed actor trajectories into this global learner buffer."""
        if tuple(other.agent_types) != self.agent_types:
            raise ValueError("actor and learner buffer agent types do not match")
        moved = {name: 0 for name in self.agent_types}
        prefix = "worker_{0}:".format(int(worker_id))
        for agent_type in self.agent_types:
            for source in other.rows.get(agent_type, []):
                row = dict(source)
                row["step_id"] = self.next_step_id
                self.next_step_id += 1
                row["task_id"] = prefix + str(row.get("task_id", ""))
                row["worker_id"] = int(worker_id)
                self.rows[agent_type].append(row)
                moved[agent_type] += 1
            other.rows[agent_type] = []
        return moved
    def to_batches(self):
        batches = {}
        for agent_type, rows in self.rows.items():
            batch = {}
            if rows:
                keys = sorted({key for row in rows for key in row})
                for key in keys:
                    values = [row.get(key) for row in rows]
                    if key in FLOAT_KEYS:
                        batch[key] = np.asarray(values, dtype=np.float32)
                    elif key in INT_KEYS:
                        batch[key] = np.asarray(values, dtype=np.int64)
                    else:
                        batch[key] = np.asarray(values, dtype=object)
            batches[agent_type] = batch
        return batches

    def clear_after_update(self):
        """Discard consumed rows and any pre-update prefix of active tasks."""
        self.rows = {name: [] for name in self.agent_types}
        self.policy_version += 1
        for tasks in self.active_tasks.values():
            for task in tasks.values():
                task["rows"] = []
                task["seen_entities"] = set(task.get("active_entities", set()))
                task["start_step"] = None
                task["policy_version"] = self.policy_version

    def clear(self):
        self.rows = {name: [] for name in self.agent_types}
        self.active_tasks = {name: {} for name in self.agent_types}
        self.next_step_id = 0
        self.episodes = 0
        self.episode_index = 1
        self.policy_version = 0

    def state_dict(self, include_active: bool = True):
        serial_active = {name: {} for name in self.agent_types}
        if include_active:
            for name, tasks in self.active_tasks.items():
                for task_id, task in tasks.items():
                    copied = dict(task)
                    copied["active_entities"] = sorted(task.get("active_entities", set()))
                    copied["seen_entities"] = sorted(task.get("seen_entities", set()))
                    serial_active[name][task_id] = copied
        return {
            "format": self.FORMAT,
            "agent_types": self.agent_types,
            "rows": self.rows,
            "active_tasks": serial_active,
            "next_step_id": self.next_step_id,
            "episodes": self.episodes,
            "episode_index": self.episode_index,
            "policy_version": self.policy_version,
        }

    def load_state_dict(self, state):
        state = dict(state or {})
        if not state:
            return
        if state.get("format") != self.FORMAT:
            print("resume_buffer_reset", "reason", "legacy_complete_episode_buffer")
            self.clear()
            return
        if tuple(state.get("agent_types", self.agent_types)) != self.agent_types:
            raise ValueError("stage buffer agent types do not match")
        self.rows = {
            name: list(state.get("rows", {}).get(name, []))
            for name in self.agent_types
        }
        self.active_tasks = {name: {} for name in self.agent_types}
        for name in self.agent_types:
            for task_id, task in state.get("active_tasks", {}).get(name, {}).items():
                copied = dict(task)
                copied["active_entities"] = set(task.get("active_entities", []))
                copied["seen_entities"] = set(task.get("seen_entities", []))
                self.active_tasks[name][task_id] = copied
        self.next_step_id = int(state.get("next_step_id", 0))
        self.episodes = int(state.get("episodes", 0))
        self.episode_index = int(state.get("episode_index", self.episodes + 1))
        self.policy_version = int(state.get("policy_version", 0))


# Compatibility import name used by older entry points and tests.
CompleteEpisodeBuffer = TaskTrajectoryBuffer
