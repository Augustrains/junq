"""Recon-shared/attack-independent HAPPO parallel training entry point.

This wrapper preserves the legacy training entry point and enables only the
recon parameter-sharing topology required for leader succession. Attack actors
remain per-aircraft and all attack trajectories remain trainable.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.train_recon_attack_parallel_eval import main


def _recon_shared_defaults(argv):
    result = list(argv)
    if "--shared-recon-actor" not in result:
        result.append("--shared-recon-actor")
    if "--recon-init-checkpoint" not in result and "--attack-init-checkpoint" in result:
        attack_checkpoint = result[result.index("--attack-init-checkpoint") + 1]
        result.extend(["--recon-init-checkpoint", attack_checkpoint])
    if "--recon-init-policy" not in result:
        result.extend(["--recon-init-policy", "red_recon_1"])
    if "--checkpoint-dir" not in result:
        result.extend([
            "--checkpoint-dir",
            str(ROOT / "checkpoints" / "happo_recon_shared_attack_independent_parallel_eval"),
        ])
    return result


if __name__ == "__main__":
    sys.argv[1:] = _recon_shared_defaults(sys.argv[1:])
    raise SystemExit(main())