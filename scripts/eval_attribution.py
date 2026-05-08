"""Aggregate attribution outputs into a (method, dataset) IoU/F1/Prec/Recall table.

Reads any number of JSON files produced by `scripts/run_attribution.py` and
emits a single CSV row per (method, dataset) tuple, plus a Markdown table
identical in structure to Table 1 of the paper.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from medfocus.attribution.eval import compute_metrics_table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", type=Path, required=True,
                    help="Paths to JSON files produced by run_attribution.py.")
    ap.add_argument("--out", type=Path, default=Path("results/main_table.csv"))
    args = ap.parse_args()

    rows: list[dict] = []
    for jf in args.inputs:
        rows.extend(json.load(open(jf)))
    if not rows:
        raise SystemExit("No records found in --inputs.")

    table = compute_metrics_table(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(table.to_markdown(index=False, floatfmt=".2f"))
    print(f"\nwrote -> {args.out}")


if __name__ == "__main__":
    main()
