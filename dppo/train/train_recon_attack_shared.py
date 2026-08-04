"""Train recon/attack HAPPO with one shared policy and pooled update per type."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train.train_recon_attack_stage import main


def _add_shared_defaults(argv: list[str]) -> list[str]:
    result = list(argv)
    if "--share-policy-by-type" not in result:
        result.append("--share-policy-by-type")
    if "--checkpoint-dir" not in result:
        result.extend(["--checkpoint-dir", str(ROOT / "checkpoints" / "recon_attack_shared")])
    return result


if __name__ == "__main__":
    sys.argv[1:] = _add_shared_defaults(sys.argv[1:])
    raise SystemExit(main())