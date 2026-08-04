"""AFSIM adapter for the official marlbenchmark/on-policy HAPPO code.

No PPO/HAPPO loss is implemented here. Networks, clipped updates, sequential
importance factors, and minibatch generation are provided by vendored upstream.
"""
from __future__ import annotations

import copy
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Mapping, MutableMapping

import numpy as np
import torch

UPSTREAM_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "mappo_on_policy"
if str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))

from onpolicy.algorithms.happo.happo_trainer import HAPPO as UpstreamHAPPO  # noqa: E402
from onpolicy.algorithms.happo.policy import HAPPO_Policy as UpstreamHAPPOPolicy  # noqa: E402
from onpolicy.utils.separated_buffer import SeparatedReplayBuffer  # noqa: E402

try:
    from .ppo_utils import add_gae_to_batch
except ImportError:  # pragma: no cover
    from ppo_utils import add_gae_to_batch

BOTTOM_AGENT_TYPES = ("recon", "attack", "landing", "ground")
UPSTREAM_REPOSITORY = "https://github.com/marlbenchmark/on-policy"
UPSTREAM_COMMIT = "de66d7a4b23fac2513f56f96f73b3f5cb96695ac"
UPSTREAM_COMPONENTS = (
    "onpolicy.algorithms.happo.policy.HAPPO_Policy",
    "onpolicy.algorithms.happo.happo_trainer.HAPPO",
    "onpolicy.utils.separated_buffer.SeparatedReplayBuffer",
)


class Box:
    def __init__(self, shape):
        self.shape = tuple(shape)


class Discrete:
    def __init__(self, n):
        self.n = int(n)
        self.shape = ()


@dataclass
class HAPPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 3.0e-4
    critic_learning_rate: float | None = None
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 256
    normalize_advantages: bool = True
    sequential_agent_order: tuple[str, ...] = BOTTOM_AGENT_TYPES
    use_huber_loss: bool = True
    huber_delta: float = 10.0
    use_feature_normalization: bool = True
    use_orthogonal: bool = True
    use_relu: bool = True
    gain: float = 0.01


class OfficialPolicy(UpstreamHAPPOPolicy):
    """Unchanged upstream policy plus a convenience parameter iterator."""
    def parameters(self):
        yield from self.actor.parameters()
        yield from self.critic.parameters()


    def train(self, mode=True):
        self.actor.train(mode)
        self.critic.train(mode)
        return self

    def eval(self):
        self.actor.eval()
        self.critic.eval()
        return self

class HAPPOTrainer:
    """AFSIM I/O adapter around official separated-policy HAPPO."""
    upstream_commit = UPSTREAM_COMMIT
    implementation = "official_marlbenchmark_on_policy"
    upstream_repository = UPSTREAM_REPOSITORY
    upstream_components = UPSTREAM_COMPONENTS

    def __init__(self, specs: Mapping[str, Mapping[str, object]], global_state_dim: int,
                 agent_types: Iterable[str] = BOTTOM_AGENT_TYPES, hidden_sizes=(128, 128),
                 config: HAPPOConfig | None = None, device: str | torch.device = "cpu",
                 trainable_agent_types: Iterable[str] | None = None):
        self.config = config or HAPPOConfig()
        self.device = torch.device(device)
        self.global_state_dim = int(global_state_dim)
        self.agent_types = tuple(agent_types)
        self.specs = {name: dict(specs[name]) for name in self.agent_types}
        sizes = tuple(int(size) for size in hidden_sizes)
        if not sizes or len(set(sizes)) != 1:
            raise ValueError("official on-policy MLP requires equal hidden layer widths")
        self.hidden_size = sizes[0]
        self.layer_N = max(0, len(sizes) - 1)
        requested_order = tuple(self.config.sequential_agent_order)
        self.update_order = tuple(x for x in requested_order if x in self.agent_types)
        self.update_order += tuple(x for x in self.agent_types if x not in self.update_order)

        self.entity_names: Dict[str, tuple[str, ...]] = {}
        self.entity_agent_type: Dict[str, str] = {}
        self.entity_policies: Dict[str, OfficialPolicy] = {}
        self.entity_trainers: Dict[str, UpstreamHAPPO] = {}
        self.entity_action_spaces: Dict[str, object] = {}
        self._policy_args: Dict[str, SimpleNamespace] = {}
        for agent_type in self.agent_types:
            spec = self.specs[agent_type]
            names = tuple(str(x) for x in spec.get("entity_names", ()) if str(x)) or (agent_type,)
            self.entity_names[agent_type] = names
            for entity_name in names:
                if entity_name in self.entity_policies:
                    raise ValueError("duplicate HAPPO entity name: {0}".format(entity_name))
                args = self._make_args()
                obs_space = Box((int(spec["obs_dim"]),))
                cent_obs_space = Box((self.global_state_dim + int(spec["obs_dim"]),))
                act_space = (
                    Box((int(spec["action_dim"]),))
                    if str(spec.get("action_type", "discrete")) == "continuous"
                    else Discrete(int(spec["action_dim"]))
                )
                policy = OfficialPolicy(args, obs_space, cent_obs_space, act_space, self.device)
                self.entity_policies[entity_name] = policy
                self.entity_trainers[entity_name] = UpstreamHAPPO(args, policy, self.device)
                self.entity_action_spaces[entity_name] = act_space
                self._policy_args[entity_name] = args
                self.entity_agent_type[entity_name] = agent_type
        self.policies = {x: self.entity_policies[self.entity_names[x][0]] for x in self.agent_types}
        self.optimizers = {x: self.policies[x].actor_optimizer for x in self.agent_types}
        self.set_trainable_agent_types(
            self.agent_types if trainable_agent_types is None else trainable_agent_types)

    def _make_args(self) -> SimpleNamespace:
        critic_lr = self.config.critic_learning_rate
        return SimpleNamespace(
            algorithm_name="happo", lr=float(self.config.learning_rate),
            critic_lr=float(self.config.learning_rate if critic_lr is None else critic_lr),
            opti_eps=1e-5, weight_decay=0.0, hidden_size=self.hidden_size,
            layer_N=self.layer_N, use_feature_normalization=bool(self.config.use_feature_normalization),
            use_orthogonal=bool(self.config.use_orthogonal), use_ReLU=bool(self.config.use_relu),
            stacked_frames=1, gain=float(self.config.gain), recurrent_N=1,
            use_recurrent_policy=False, use_naive_recurrent_policy=False,
            use_policy_active_masks=True, use_value_active_masks=True,
            use_popart=False, use_valuenorm=False, clip_param=float(self.config.clip_coef),
            ppo_epoch=int(self.config.update_epochs), num_mini_batch=1, data_chunk_length=1,
            value_loss_coef=float(self.config.value_coef), entropy_coef=float(self.config.entropy_coef),
            max_grad_norm=float(self.config.max_grad_norm), huber_delta=float(self.config.huber_delta),
            use_max_grad_norm=True, use_clipped_value_loss=True,
            use_huber_loss=bool(self.config.use_huber_loss), episode_length=1,
            n_rollout_threads=1, gamma=float(self.config.gamma),
            gae_lambda=float(self.config.gae_lambda), use_gae=True,
            use_proper_time_limits=False)

    def set_trainable_agent_types(self, agent_types: Iterable[str]):
        requested = tuple(dict.fromkeys(str(x) for x in agent_types))
        unknown = sorted(set(requested) - set(self.agent_types))
        if unknown:
            raise ValueError("unknown trainable HAPPO agent types: {0}".format(unknown))
        self.trainable_agent_types = tuple(x for x in self.update_order if x in requested)
        trainable = set(self.trainable_agent_types)
        for entity_name, policy in self.entity_policies.items():
            enabled = self.entity_agent_type[entity_name] in trainable
            for parameter in policy.parameters():
                parameter.requires_grad_(enabled)

    def is_trainable(self, agent_type: str) -> bool:
        return str(agent_type) in self.trainable_agent_types

    def _policy_id(self, agent_type: str, entity_name: str) -> str:
        if entity_name in self.entity_policies:
            if self.entity_agent_type[entity_name] != agent_type:
                raise KeyError("entity type mismatch: {0}".format(entity_name))
            return entity_name
        names = self.entity_names[agent_type]
        if len(names) == 1:
            return names[0]
        raise KeyError("no official HAPPO policy registered for {0}".format(entity_name))

    def _rnn_state(self, rows=1):
        return np.zeros((rows, 1, self.hidden_size), dtype=np.float32)

    def _central_obs(self, obs, global_state):
        local = np.asarray(obs, dtype=np.float32).reshape(-1)
        global_array = (np.zeros(self.global_state_dim, dtype=np.float32) if global_state is None
                        else np.asarray(global_state, dtype=np.float32).reshape(-1))
        if global_array.size != self.global_state_dim:
            raise ValueError("global state dimension mismatch")
        return np.concatenate((global_array, local), axis=0)

    def act_bottom(self, agent_type, entity_name, agent_state, global_state=None) -> dict:
        policy = self.entity_policies[self._policy_id(str(agent_type), str(entity_name))]
        policy.actor.eval(); policy.critic.eval()
        obs = np.asarray(agent_state["obs"], dtype=np.float32).reshape(1, -1)
        policy_id = self._policy_id(str(agent_type), str(entity_name))
        continuous = self.entity_action_spaces[policy_id].__class__.__name__ == "Box"
        available = None if continuous else np.asarray(
            agent_state["action_mask"], dtype=np.float32
        ).reshape(1, -1)
        cent_obs = self._central_obs(obs[0], global_state).reshape(1, -1)
        masks = np.ones((1, 1), dtype=np.float32)
        with torch.no_grad():
            value, action, logprob, _, _ = policy.get_actions(
                cent_obs, obs, self._rnn_state(), self._rnn_state(), masks, available)
        action_array = action.detach().cpu().numpy().reshape(-1)
        selected_action = action_array.astype(np.float32) if continuous else int(action_array[0])
        return {"action": selected_action,
                "logprob": float(logprob.detach().cpu().numpy().reshape(-1)[0]),
                "value": float(value.detach().cpu().numpy().reshape(-1)[0])}

    def value_bottom(self, agent_type, entity_name, agent_state, global_state=None) -> float:
        policy = self.entity_policies[self._policy_id(str(agent_type), str(entity_name))]
        cent_obs = self._central_obs(agent_state["obs"], global_state).reshape(1, -1)
        with torch.no_grad():
            value = policy.get_values(cent_obs, self._rnn_state(), np.ones((1, 1), np.float32))
        return float(value.detach().cpu().numpy().reshape(-1)[0])

    @staticmethod
    def _slice_batch(batch, indices):
        size = len(batch["action"]); result = {}
        for key, value in batch.items():
            try:
                if len(value) == size:
                    result[key] = np.asarray(value)[indices]; continue
            except (TypeError, ValueError):
                pass
            result[key] = value
        return result

    def _entity_batches(self, agent_type, batch):
        names = np.asarray(batch.get("entity_name", []), dtype=object)
        if names.size == 0:
            yield self.entity_names[agent_type][0], batch; return
        policy_names = self.entity_names[agent_type]
        if len(policy_names) == 1 and not np.any(names == policy_names[0]):
            # A synthetic policy id means every real entity shares one Actor.
            # Preserve real entity names in the batch for diagnostics.
            yield policy_names[0], batch
            return
        for entity_name in policy_names:
            indices = np.flatnonzero(names == entity_name)
            if indices.size:
                yield entity_name, self._slice_batch(batch, indices)

    def _make_buffer(self, policy_id, batch, factors):
        n = int(len(batch["action"])); args = copy.copy(self._policy_args[policy_id])
        args.episode_length = n; args.n_rollout_threads = 1
        agent_type = self.entity_agent_type[policy_id]; spec = self.specs[agent_type]
        obs_space = Box((int(spec["obs_dim"]),))
        share_space = Box((self.global_state_dim + int(spec["obs_dim"]),))
        act_space = self.entity_action_spaces[policy_id]
        buffer = SeparatedReplayBuffer(args, obs_space, share_space, act_space)
        if act_space.__class__.__name__ == "Box":
            # Upstream allocates log-probability with action width for Box,
            # while DiagGaussian correctly returns one joint log probability.
            buffer.action_log_probs = np.zeros((n, 1, 1), dtype=np.float32)
        obs = np.asarray(batch["obs"], np.float32)
        global_state = np.asarray(batch["global_state"], np.float32)
        buffer.obs[:-1, 0] = obs
        buffer.share_obs[:-1, 0] = np.concatenate((global_state, obs), axis=1)
        if "next_obs" in batch:
            buffer.obs[-1, 0] = np.asarray(batch["next_obs"][-1], np.float32)
            buffer.share_obs[-1, 0] = self._central_obs(batch["next_obs"][-1], global_state[-1])
        actions = np.asarray(batch["action"], np.float32)
        if act_space.__class__.__name__ == "Box":
            buffer.actions[:, 0, :] = actions.reshape(n, -1)
        else:
            buffer.actions[:, 0, 0] = actions.reshape(-1)
        buffer.action_log_probs[:, 0, 0] = np.asarray(batch["logprob"], np.float32)
        buffer.value_preds[:-1, 0, 0] = np.asarray(batch["value"], np.float32)
        buffer.value_preds[-1, 0, 0] = float(np.asarray(batch.get("next_value", [0.0]))[-1])
        buffer.rewards[:, 0, 0] = np.asarray(batch["reward"], np.float32)
        done = np.asarray(batch.get("done", np.zeros(n)), np.float32)
        buffer.masks[1:, 0, 0] = 1.0 - done
        buffer.active_masks[:-1, 0, 0] = 1.0
        if buffer.available_actions is not None:
            buffer.available_actions[:-1, 0] = np.asarray(batch["action_mask"], np.float32)
        buffer.returns[:-1, 0, 0] = np.asarray(batch["return"], np.float32)
        buffer.returns[-1, 0, 0] = buffer.value_preds[-1, 0, 0]
        buffer.factor = np.asarray(factors, np.float32).reshape(n, 1, 1)
        return buffer

    def _logprob_ratio(self, policy_id, batch):
        policy = self.entity_policies[policy_id]
        obs = np.asarray(batch["obs"], np.float32); n = len(obs)
        share = np.concatenate((np.asarray(batch["global_state"], np.float32), obs), axis=1)
        ones = np.ones((n, 1), np.float32)
        with torch.no_grad():
            _, new_logprob, _ = policy.evaluate_actions(
                share, obs, self._rnn_state(n), self._rnn_state(n),
                (np.asarray(batch["action"], np.float32).reshape(n, -1)
                 if self.entity_action_spaces[policy_id].__class__.__name__ == "Box"
                 else np.asarray(batch["action"], np.int64).reshape(n, 1)), ones,
                (None if self.entity_action_spaces[policy_id].__class__.__name__ == "Box"
                 else np.asarray(batch["action_mask"], np.float32)), ones)
        old = np.asarray(batch["logprob"], np.float32).reshape(-1)
        return np.exp(new_logprob.detach().cpu().numpy().reshape(-1) - old)

    def update_all(self, bottom_batches: Mapping[str, MutableMapping[str, object]]) -> Dict[str, dict]:
        prepared = {}; metrics = {}
        for agent_type in self.update_order:
            source = bottom_batches.get(agent_type)
            available = len(source.get("action", [])) if source else 0
            if not self.is_trainable(agent_type):
                metrics[agent_type] = {"samples": 0, "available_samples": int(available),
                                       "frozen": True, "correction_factor": 1.0}; continue
            if not source or not available:
                metrics[agent_type] = {"samples": 0, "empty_batch": True,
                                       "frozen": False}; continue
            batch = dict(source)
            if "return" not in batch:
                add_gae_to_batch(batch, gamma=self.config.gamma, lam=self.config.gae_lambda,
                                 normalize=False)
            prepared[agent_type] = batch

        update_entities = []
        for agent_type in self.update_order:
            if agent_type in prepared:
                update_entities.extend(self._entity_batches(agent_type, prepared[agent_type]))
        random.shuffle(update_entities)
        step_factors: Dict[int, float] = {}
        rows_by_type: Dict[str, list] = {x: [] for x in prepared}
        for policy_id, batch in update_entities:
            agent_type = self.entity_agent_type[policy_id]
            steps = np.asarray(batch.get("step_id", np.arange(len(batch["action"]))), np.int64)
            factors = np.asarray([step_factors.get(int(x), 1.0) for x in steps], np.float32)
            buffer = self._make_buffer(policy_id, batch, factors)
            trainer = self.entity_trainers[policy_id]
            trainer.num_mini_batch = max(1, min(len(batch["action"]),
                len(batch["action"]) // max(1, self.config.minibatch_size)))
            trainer.prep_training(); info = trainer.train(buffer, update_actor=True)
            ratios = self._logprob_ratio(policy_id, batch)
            grouped = {}; counts = {}
            for step, ratio in zip(steps, ratios):
                key = int(step); grouped[key] = grouped.get(key, 0.0) + float(ratio)
                counts[key] = counts.get(key, 0) + 1
            for key, total in grouped.items():
                step_factors[key] = step_factors.get(key, 1.0) * total / counts[key]
            row = {key: self._number(value) for key, value in info.items()}
            row.update({"entity": policy_id, "samples": int(len(batch["action"])),
                        "correction_factor": float(np.mean(factors))})
            rows_by_type[agent_type].append(row)

        for agent_type, rows in rows_by_type.items():
            if not rows: continue
            numeric = set.intersection(*[{k for k, v in row.items()
                                          if isinstance(v, (int, float))} for row in rows])
            metrics[agent_type] = {k: float(np.mean([row[k] for row in rows])) for k in numeric}
            item = metrics[agent_type]
            item["loss"] = (item.get("policy_loss", 0.0)
                + self.config.value_coef * item.get("value_loss", 0.0)
                - self.config.entropy_coef * item.get("dist_entropy", 0.0))

            metrics[agent_type]["samples"] = int(sum(row["samples"] for row in rows))
            metrics[agent_type]["entities_updated"] = len(rows)
            metrics[agent_type]["frozen"] = False
        return metrics

    @staticmethod
    def _number(value):
        return float(value.detach().cpu()) if torch.is_tensor(value) else float(value)

    def state_dict(self):
        policies = {}
        for name, policy in self.entity_policies.items():
            policies[name] = {"agent_type": self.entity_agent_type[name],
                "actor": policy.actor.state_dict(), "critic": policy.critic.state_dict(),
                "actor_optimizer": policy.actor_optimizer.state_dict(),
                "critic_optimizer": policy.critic_optimizer.state_dict()}
        return {"algorithm": "happo_official_on_policy", "implementation": self.implementation,
            "upstream_repository": self.upstream_repository,
            "upstream_components": self.upstream_components, "config": asdict(self.config),
            "upstream_commit": self.upstream_commit,
            "global_state_dim": self.global_state_dim, "agent_types": self.agent_types,
            "entity_names": self.entity_names,
            "trainable_agent_types": self.trainable_agent_types, "policies": policies}

    def load_state_dict(self, state: Mapping[str, object]):
        if state.get("algorithm") != "happo_official_on_policy":
            raise ValueError("legacy custom-HAPPO checkpoints are incompatible with official HAPPO")
        saved = state.get("policies", {})
        missing = sorted(set(self.entity_policies) - set(saved))
        if missing:
            raise ValueError("checkpoint missing entity policies: {0}".format(missing))
        for name, policy in self.entity_policies.items():
            packet = saved[name]
            policy.actor.load_state_dict(packet["actor"], strict=True)
            policy.critic.load_state_dict(packet["critic"], strict=True)
            policy.actor_optimizer.load_state_dict(packet["actor_optimizer"])
            policy.critic_optimizer.load_state_dict(packet["critic_optimizer"])

