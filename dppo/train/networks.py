"""Neural network building blocks for PPO/MAPPO training."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn

try:
    from .ppo_utils import masked_categorical
except ImportError:  # pragma: no cover - direct script execution fallback
    from ppo_utils import masked_categorical


def _mlp(input_dim: int, hidden_sizes: Sequence[int], output_dim: int, activation=nn.Tanh) -> nn.Sequential:
    layers = []
    last_dim = int(input_dim)
    for hidden_dim in hidden_sizes:
        layers.append(nn.Linear(last_dim, int(hidden_dim)))
        layers.append(activation())
        last_dim = int(hidden_dim)
    layers.append(nn.Linear(last_dim, int(output_dim)))
    return nn.Sequential(*layers)


class MLPActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: Iterable[int] = (128, 128)):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.net = _mlp(self.obs_dim, tuple(hidden_sizes), self.action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def distribution(self, obs: torch.Tensor, action_mask: torch.Tensor):
        return masked_categorical(self.forward(obs), action_mask)


class MLPCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: Iterable[int] = (128, 128)):
        super().__init__()
        self.input_dim = int(input_dim)
        self.net = _mlp(self.input_dim, tuple(hidden_sizes), 1)

    def forward(self, critic_input: torch.Tensor) -> torch.Tensor:
        return self.net(critic_input).squeeze(-1)


class EntityAttentionCritic(nn.Module):
    """Global-only critic for the single high-level commander.

    The flat transport tensor is split into commander context plus fixed entity
    slots. Categorical side/role/task ids are embedded, entity slots share an
    encoder, and self-attention produces a permutation-tolerant global summary.
    """

    def __init__(
        self,
        input_dim: int,
        commander_dim: int,
        entity_count: int,
        entity_feature_dim: int,
        hidden_sizes: Iterable[int] = (128, 128),
        entity_embedding_dim: int = 64,
        attention_heads: int = 4,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.commander_dim = int(commander_dim)
        self.entity_count = int(entity_count)
        self.entity_feature_dim = int(entity_feature_dim)
        expected = self.commander_dim + self.entity_count * self.entity_feature_dim
        if expected != self.input_dim:
            raise ValueError(
                "entity critic layout mismatch: input={0}, expected={1}".format(
                    self.input_dim, expected
                )
            )
        self.entity_embedding_dim = int(entity_embedding_dim)
        heads = max(1, int(attention_heads))
        while self.entity_embedding_dim % heads != 0 and heads > 1:
            heads -= 1
        self.side_embedding = nn.Embedding(2, 4)
        self.role_embedding = nn.Embedding(8, 8)
        self.task_embedding = nn.Embedding(10, 6)
        self.continuous_indices = tuple(
            index for index in range(self.entity_feature_dim)
            if index not in (3, 4, 19)
        )
        encoded_input_dim = len(self.continuous_indices) + 4 + 8 + 6
        self.entity_encoder = _mlp(
            encoded_input_dim,
            (self.entity_embedding_dim,),
            self.entity_embedding_dim,
        )
        self.entity_attention = nn.MultiheadAttention(
            self.entity_embedding_dim,
            heads,
            batch_first=True,
        )
        self.commander_encoder = _mlp(
            self.commander_dim,
            (self.entity_embedding_dim,),
            self.entity_embedding_dim,
        )
        self.value_head = _mlp(
            self.entity_embedding_dim * 3,
            tuple(hidden_sizes),
            1,
        )

    def forward(self, critic_input: torch.Tensor) -> torch.Tensor:
        if critic_input.ndim == 1:
            critic_input = critic_input.unsqueeze(0)
        commander = critic_input[:, :self.commander_dim]
        entities = critic_input[:, self.commander_dim:].reshape(
            -1, self.entity_count, self.entity_feature_dim
        )
        side_ids = entities[:, :, 3].round().long().clamp(0, 1)
        role_ids = entities[:, :, 4].round().long().clamp(0, 7)
        task_ids = entities[:, :, 19].round().long().clamp(0, 9)
        continuous = entities[:, :, self.continuous_indices]
        entity_input = torch.cat(
            [
                continuous,
                self.side_embedding(side_ids),
                self.role_embedding(role_ids),
                self.task_embedding(task_ids),
            ],
            dim=-1,
        )
        encoded = self.entity_encoder(entity_input)
        valid = entities[:, :, 0] > 0.5
        valid = torch.where(
            valid.any(dim=1, keepdim=True),
            valid,
            torch.ones_like(valid),
        )
        attended, _weights = self.entity_attention(
            encoded,
            encoded,
            encoded,
            key_padding_mask=~valid,
            need_weights=False,
        )
        attended = torch.tanh(attended + encoded)
        mask = valid.unsqueeze(-1).to(attended.dtype)
        mean_pool = (attended * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        max_pool = attended.masked_fill(~valid.unsqueeze(-1), -1.0e9).max(dim=1).values
        commander_embedding = self.commander_encoder(commander)
        return self.value_head(
            torch.cat([commander_embedding, mean_pool, max_pool], dim=-1)
        ).squeeze(-1)


class EntityAttentionLocalCritic(nn.Module):
    """Centralized bottom-level critic with a separate local-context encoder.

    Input layout is ``global commander summary + fixed global entity slots +
    local Actor observation``.  The global entity set is encoded with shared
    weights and attention instead of treating the roughly 2K input fields as
    unrelated columns in one flat MLP.
    """

    def __init__(
        self,
        input_dim: int,
        global_state_dim: int,
        commander_dim: int,
        entity_count: int,
        entity_feature_dim: int,
        local_dim: int,
        hidden_sizes: Iterable[int] = (128, 128),
        entity_embedding_dim: int = 64,
        attention_heads: int = 4,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.global_state_dim = int(global_state_dim)
        self.commander_dim = int(commander_dim)
        self.entity_count = int(entity_count)
        self.entity_feature_dim = int(entity_feature_dim)
        self.local_dim = int(local_dim)
        expected_global = self.commander_dim + self.entity_count * self.entity_feature_dim
        if expected_global != self.global_state_dim:
            raise ValueError(
                "bottom critic global layout mismatch: global={0}, expected={1}".format(
                    self.global_state_dim, expected_global
                )
            )
        if self.global_state_dim + self.local_dim != self.input_dim:
            raise ValueError(
                "bottom critic input layout mismatch: input={0}, expected={1}".format(
                    self.input_dim, self.global_state_dim + self.local_dim
                )
            )

        self.entity_embedding_dim = int(entity_embedding_dim)
        heads = max(1, int(attention_heads))
        while self.entity_embedding_dim % heads != 0 and heads > 1:
            heads -= 1
        self.side_embedding = nn.Embedding(2, 4)
        self.role_embedding = nn.Embedding(8, 8)
        self.task_embedding = nn.Embedding(10, 6)
        self.continuous_indices = tuple(
            index for index in range(self.entity_feature_dim)
            if index not in (3, 4, 19)
        )
        encoded_input_dim = len(self.continuous_indices) + 4 + 8 + 6
        self.entity_encoder = _mlp(
            encoded_input_dim,
            (self.entity_embedding_dim,),
            self.entity_embedding_dim,
        )
        self.entity_attention = nn.MultiheadAttention(
            self.entity_embedding_dim,
            heads,
            batch_first=True,
        )
        self.commander_encoder = _mlp(
            self.commander_dim,
            (self.entity_embedding_dim,),
            self.entity_embedding_dim,
        )
        self.local_encoder = _mlp(
            self.local_dim,
            (self.entity_embedding_dim,),
            self.entity_embedding_dim,
        )
        self.value_head = _mlp(
            self.entity_embedding_dim * 4,
            tuple(hidden_sizes),
            1,
        )

    def forward(self, critic_input: torch.Tensor) -> torch.Tensor:
        if critic_input.ndim == 1:
            critic_input = critic_input.unsqueeze(0)
        global_state = critic_input[:, :self.global_state_dim]
        local_state = critic_input[:, self.global_state_dim:]
        commander = global_state[:, :self.commander_dim]
        entities = global_state[:, self.commander_dim:].reshape(
            -1, self.entity_count, self.entity_feature_dim
        )
        side_ids = entities[:, :, 3].round().long().clamp(0, 1)
        role_ids = entities[:, :, 4].round().long().clamp(0, 7)
        task_ids = entities[:, :, 19].round().long().clamp(0, 9)
        continuous = entities[:, :, self.continuous_indices]
        entity_input = torch.cat(
            [
                continuous,
                self.side_embedding(side_ids),
                self.role_embedding(role_ids),
                self.task_embedding(task_ids),
            ],
            dim=-1,
        )
        encoded = self.entity_encoder(entity_input)
        valid = entities[:, :, 0] > 0.5
        valid = torch.where(
            valid.any(dim=1, keepdim=True),
            valid,
            torch.ones_like(valid),
        )
        attended, _weights = self.entity_attention(
            encoded,
            encoded,
            encoded,
            key_padding_mask=~valid,
            need_weights=False,
        )
        attended = torch.tanh(attended + encoded)
        mask = valid.unsqueeze(-1).to(attended.dtype)
        mean_pool = (attended * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        max_pool = attended.masked_fill(~valid.unsqueeze(-1), -1.0e9).max(dim=1).values
        commander_embedding = self.commander_encoder(commander)
        local_embedding = self.local_encoder(local_state)
        return self.value_head(
            torch.cat(
                [commander_embedding, mean_pool, max_pool, local_embedding],
                dim=-1,
            )
        ).squeeze(-1)


class ActorCriticPolicy(nn.Module):
    """Actor plus critic with the collector-compatible act() method."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        critic_input_dim: int | None = None,
        hidden_sizes: Iterable[int] = (128, 128),
        critic_global_only: bool = False,
        critic_module: nn.Module | None = None,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.critic_input_dim = int(critic_input_dim or obs_dim)
        self.critic_global_only = bool(critic_global_only)
        self.actor = MLPActor(self.obs_dim, self.action_dim, hidden_sizes=hidden_sizes)
        self.critic = critic_module or MLPCritic(self.critic_input_dim, hidden_sizes=hidden_sizes)
        self.capture_network_inputs = False
        self.last_actor_input = None
        self.last_critic_input = None

    def set_input_capture(self, enabled: bool = True):
        self.capture_network_inputs = bool(enabled)
        if not enabled:
            self.last_actor_input = None
            self.last_critic_input = None
        return self

    def _critic_input(self, obs: torch.Tensor, global_state: torch.Tensor | None = None) -> torch.Tensor:
        if self.critic_global_only:
            if global_state is None:
                raise ValueError("global-only critic requires global_state")
            return global_state
        if self.critic_input_dim == self.obs_dim or global_state is None:
            return obs
        return torch.cat([global_state, obs], dim=-1)

    def _capture_inputs(self, obs_t: torch.Tensor, critic_input: torch.Tensor):
        if not self.capture_network_inputs:
            return
        self.last_actor_input = obs_t.detach().cpu().clone()
        self.last_critic_input = critic_input.detach().cpu().clone()

    @torch.no_grad()
    def act(self, obs, mask, global_state=None, deterministic: bool = False, **_kwargs) -> dict:
        device = next(self.parameters()).device
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32), dtype=torch.float32, device=device).view(1, -1)
        mask_t = torch.as_tensor(np.asarray(mask, dtype=np.float32), dtype=torch.float32, device=device).view(1, -1)
        global_t = None
        if global_state is not None:
            global_t = torch.as_tensor(np.asarray(global_state, dtype=np.float32), dtype=torch.float32, device=device).view(1, -1)
        dist = self.actor.distribution(obs_t, mask_t)
        action_t = torch.argmax(dist.probs, dim=-1) if deterministic else dist.sample()
        logprob_t = dist.log_prob(action_t)
        critic_input = self._critic_input(obs_t, global_t)
        self._capture_inputs(obs_t, critic_input)
        value_t = self.critic(critic_input)
        return {
            "action": int(action_t.item()),
            "logprob": float(logprob_t.item()),
            "value": float(value_t.item()),
        }

    @torch.no_grad()
    def value(self, obs, global_state=None) -> float:
        device = next(self.parameters()).device
        obs_t = torch.as_tensor(
            np.asarray(obs, dtype=np.float32), dtype=torch.float32, device=device
        ).view(1, -1)
        global_t = None
        if global_state is not None:
            global_t = torch.as_tensor(
                np.asarray(global_state, dtype=np.float32), dtype=torch.float32, device=device
            ).view(1, -1)
        critic_input = self._critic_input(obs_t, global_t)
        self._capture_inputs(obs_t, critic_input)
        return float(self.critic(critic_input).item())

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        actions: torch.Tensor,
        global_state: torch.Tensor | None = None,
    ) -> dict:
        dist = self.actor.distribution(obs, action_mask)
        critic_input = self._critic_input(obs, global_state)
        self._capture_inputs(obs, critic_input)
        values = self.critic(critic_input)
        return {
            "logprob": dist.log_prob(actions),
            "entropy": dist.entropy(),
            "value": values,
        }
