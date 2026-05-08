"""Concatenate per-(model, dataset, mode) filter outputs into the released JSON.

Reads JSON files written by `scripts/run_three_step_filter.py` from a directory
tree like `predictions/<model>/<dataset>/<mode>.json` and emits the merged
`data/medground_bench/{direct,reasoning}.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for mode, out_name in [("direct", "direct.json"), ("cot", "reasoning.json")]:
        records: list[dict] = []
        for jf in args.predictions_root.glob(f"*/*/{mode}.json"):
            records.extend(json.load(open(jf)))
        out_path = args.out_dir / out_name
        json.dump(records, open(out_path, "w"), indent=2)
        print(f"wrote {len(records):>5d} records -> {out_path}")


if __name__ == "__main__":
    main()
