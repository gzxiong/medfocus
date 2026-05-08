"""One-shot extractor: build `data/medground_bench/{direct,reasoning}.json`.

Reads the original `predictions/collected_filtered_*_results.json` produced by
the 3-step causal filter, normalizes dataset / model identifiers to the
package's registry keys, and emits the released benchmark JSON.

Usage (run once):
    python scripts/extract_medground_bench.py \
        --predictions-dir /path/to/raw/predictions \
        --out-dir data/medground_bench
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASET_RENAME = {
    "ImaGenome-Attr": "imagenome",
    "VinDR-CXR-Attr": "vindr_cxr",
    "PadChest-GR-Attr": "padchest_gr",
}

MODEL_RENAME = {
    "Qwen/Qwen2.5-VL-3B-Instruct": "qwen2_5_vl_3b",
    "Qwen/Qwen2.5-VL-7B-Instruct": "qwen2_5_vl_7b",
    "google/gemma-3-4b-it": "gemma3_4b",
    "google/gemma-3-12b-it": "gemma3_12b",
    "google/medgemma-4b-it": "medgemma_4b",
    "google/medgemma-1.5-4b-it": "medgemma1_5_4b",
}


def normalize(records: list[dict]) -> list[dict]:
    out: list[dict] = []
    for r in records:
        rr = dict(r)
        rr["model"] = MODEL_RENAME.get(r["model"], r["model"])
        rr["dataset"] = DATASET_RENAME.get(r["dataset"], r["dataset"])
        out.append(rr)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    direct_in = args.predictions_dir / "collected_filtered_direct_results.json"
    cot_in = args.predictions_dir / "collected_filtered_cot_results.json"

    direct = normalize(json.load(open(direct_in)))
    cot = normalize(json.load(open(cot_in)))

    json.dump(direct, open(args.out_dir / "direct.json", "w"), indent=2)
    json.dump(cot, open(args.out_dir / "reasoning.json", "w"), indent=2)

    print(f"wrote {len(direct):,} direct + {len(cot):,} reasoning samples to {args.out_dir}")


if __name__ == "__main__":
    main()
