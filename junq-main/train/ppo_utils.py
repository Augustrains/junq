"""Common PPO utilities shared by commander PPO and bottom MAPPO."""

from __future__ import annotations

from typing import Mapping, MutableMapping, Tuple

import numpy as np
import torch
from torch.distributions import Categorical


MASKED_LOGIT_VALUE = -1.0e9


def masked_logits(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """Set invalid action logits to a large negative value."""
    if action_mask.dtype != torch.bool:
        valid = action_mask > 0.0
    else:
        valid = action_mask
    return torch.where(valid, logits, torch.full_like(logits, MASKED_LOGIT_VALUE))


def masked_categorical(logits: torch.Tensor, action_mask: torch.Tensor) -> Categorical:
    """Return a categorical distribution that assigns zero mass to invalid actions."""
    return Categorical(logits=masked_logits(logits, action_mask))


def normalize_advantages(advantages: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    advantages = np.asarray(advantages, dtype=np.float32)
    if advantages.size == 0:
        return advantages
    return (advantages - float(advantages.mean())) / (float(advantages.std()) + eps)


def filter_assigned_batch(batch: Mapping[str, object]) -> dict:
    """Return only transitions produced by actively assigned entities.

    Older rollout formats do not contain an ``assigned`` field; those batches
    are already filtered and are returned unchanged. Every array-like field
    whose leading dimension matches ``assigned`` is filtered with the same
    mask so observations, actions, rewards, and metadata stay aligned.
    """
    if "assigned" not in batch:
        return dict(batch)
    assigned = np.asarray(batch["assigned"], dtype=np.float32)
    if assigned.ndim != 1:
        raise ValueError("assigned must be a one-dimensional rollout field")
    mask = assigned > 0.5
    filtered = {}
    for key, value in batch.items():
        array = np.asarray(value)
        if array.ndim > 0 and array.shape[0] == assigned.shape[0]:
            filtered[key] = array[mask]
        else:
            filtered[key] = value
    return filtered

def compute_gae(
    rewards,
    values,
    dones,
    last_value: float = 0.0,
    next_values=None,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute generalized advantage estimation.

    dones[t] is interpreted as the terminal flag after transition t.
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    if not (rewards.shape == values.shape == dones.shape):
        raise ValueError("rewards, values, and dones must have identical shapes")

    if next_values is not None:
        next_values = np.asarray(next_values, dtype=np.float32)
        if next_values.shape != rewards.shape:
            raise ValueError("next_values must have the same shape as rewards")

    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(rewards.shape[0])):
        if next_values is not None:
            next_value = float(next_values[t])
        elif t == rewards.shape[0] - 1:
            next_value = float(last_value)
        else:
            next_value = float(values[t + 1])
        next_nonterminal = 1.0 - float(dones[t])
        delta = float(rewards[t]) + float(gamma) * next_value * next_nonterminal - float(values[t])
        last_gae = delta + float(gamma) * float(lam) * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages.astype(np.float32), returns.astype(np.float32)


def add_gae_to_batch(
    batch: MutableMapping[str, object],
    last_value: float = 0.0,
    next_values=None,
    gamma: float = 0.99,
    lam: float = 0.95,
    normalize: bool = True,
) -> MutableMapping[str, object]:
    """Add advantage and return arrays to a rollout batch in-place."""
    required = ("reward", "value", "done")
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError("batch missing keys required for GAE: {0}".format(", ".join(missing)))
    rewards = np.asarray(batch["reward"], dtype=np.float32)
    values = np.asarray(batch["value"], dtype=np.float32)
    dones = np.asarray(batch["done"], dtype=np.float32)
    next_values = batch.get("next_value")
    entity_names = batch.get("entity_name")
    if entity_names is None:
        advantages, returns = compute_gae(
            rewards, values, dones, last_value=last_value, next_values=next_values,
            gamma=gamma, lam=lam,
        )
    else:
        entity_names = np.asarray(entity_names, dtype=object)
        if entity_names.shape != rewards.shape:
            raise ValueError("entity_name must have the same shape as reward")
        advantages = np.zeros_like(rewards, dtype=np.float32)
        returns = np.zeros_like(rewards, dtype=np.float32)
        next_values_array = None if next_values is None else np.asarray(next_values, dtype=np.float32)
        step_ids = batch.get("step_id")
        step_ids_array = None if step_ids is None else np.asarray(step_ids, dtype=np.int64)
        for entity_name in dict.fromkeys(entity_names.tolist()):
            entity_indices = np.flatnonzero(entity_names == entity_name)
            segments = []
            start = 0
            for offset in range(1, len(entity_indices)):
                previous = entity_indices[offset - 1]
                current = entity_indices[offset]
                step_gap = (
                    step_ids_array is not None
                    and int(step_ids_array[current]) != int(step_ids_array[previous]) + 1
                )
                if dones[previous] > 0.0 or step_gap:
                    segments.append(entity_indices[start:offset])
                    start = offset
            segments.append(entity_indices[start:])
            for indices in segments:
                entity_next_values = None if next_values_array is None else next_values_array[indices]
                entity_advantages, entity_returns = compute_gae(
                    rewards[indices], values[indices], dones[indices],
                    last_value=last_value, next_values=entity_next_values,
                    gamma=gamma, lam=lam,
                )
                advantages[indices] = entity_advantages
                returns[indices] = entity_returns
    batch["advantage"] = normalize_advantages(advantages) if normalize else advantages
    batch["return"] = returns
    return batch


def to_tensor_batch(batch: Mapping[str, object], device=None) -> dict:
    """Convert numeric numpy rollout fields to torch tensors."""
    device = device or torch.device("cpu")
    tensors = {}
    float_keys = {"obs", "next_obs", "action_mask", "global_state", "reward", "value", "next_value", "logprob", "done", "advantage", "return"}
    int_keys = {"action"}
    for key, value in batch.items():
        if key in float_keys:
            tensors[key] = torch.as_tensor(value, dtype=torch.float32, device=device)
        elif key in int_keys:
            tensors[key] = torch.as_tensor(value, dtype=torch.long, device=device)
    return tensors

