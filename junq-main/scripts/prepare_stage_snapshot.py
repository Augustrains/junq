"""Select a dynamic snapshot and materialize its AFSIM scenario/config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.stage_snapshot_scenarios import materialize_snapshot
from train.stage_snapshots import StageSnapshotPool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("landing", "ground"), required=True)
    parser.add_argument("--pool", default=str(ROOT / "stage_snapshots"))
    parser.add_argument("--snapshot", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source-scenario", required=True)
    parser.add_argument("--output-scenario", required=True)
    parser.add_argument("--output-config", default="")
    parser.add_argument("--base-config", default="afsim_units.json")
    args = parser.parse_args()
    if args.snapshot:
        snapshot_path = Path(args.snapshot)
    else:
        snapshot_path, _snapshot = StageSnapshotPool(args.pool, args.stage, seed=args.seed).sample()
    result = materialize_snapshot(
        snapshot_path,
        args.source_scenario,
        args.output_scenario,
        output_config=args.output_config or None,
        base_config=args.base_config,
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
