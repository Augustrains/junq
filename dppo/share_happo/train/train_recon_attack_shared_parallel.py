"""Four concurrent AFSIM scenarios with shared recon/attack HAPPO policies."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train.train_recon_attack_parallel_eval import main


def _shared_parallel_defaults(argv: list[str]) -> list[str]:
    result = list(argv)
    defaults = {
        "--workers": "4",
        "--simulation-clock-rate": "20",
        "--checkpoint-dir": str(ROOT / "checkpoints" / "happo_recon_attack_shared_parallel"),
    }
    for option, value in defaults.items():
        if option not in result:
            result.extend([option, value])
    if "--share-policy-by-type" not in result:
        result.append("--share-policy-by-type")
    return result


if __name__ == "__main__":
    sys.argv[1:] = _shared_parallel_defaults(sys.argv[1:])
    raise SystemExit(main())