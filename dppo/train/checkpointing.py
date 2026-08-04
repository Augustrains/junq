"""Checkpoint and JSONL metric helpers for training scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import torch


def save_training_checkpoint(payload: Mapping[str, object], checkpoint_dir, prefix: str, update_id: int):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    numbered = checkpoint_dir / "{0}_update_{1:06d}.pt".format(prefix, int(update_id))
    latest = checkpoint_dir / "latest.pt"
    torch.save(dict(payload), numbered)
    torch.save(dict(payload), latest)
    return numbered, latest


def load_training_checkpoint(path):
    try:
        # Training checkpoints include trusted local optimizer, RNG, and
        # complete NumPy trajectory-buffer state.
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch before the weights_only argument existed.
        return torch.load(str(path), map_location="cpu")


def append_metrics_jsonl(path, row: Mapping[str, object]):
    if not path:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=True, sort_keys=True) + "\n")
