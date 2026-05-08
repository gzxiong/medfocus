"""Apply the 3-step causal filter for a (model, dataset) pair.

Reads RadEdit-generated images from `--radedit-dir`, runs four LVLM forwards
per sample (original / FG-edited / BG-edited / BG-edited-normal), keeps only
samples passing the causal gate, and writes JSON records compatible with
`data/medground_bench/{direct,reasoning}.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from benchmark.filter.three_step_filter import ThreeStepFilter
from medfocus.config import load_config
from medfocus.data.imagenome import load_imagenome
from medfocus.data.padchest_gr import load_padchest_gr
from medfocus.data.vindr_cxr import load_vindr_cxr
from medfocus.lvlm import load_lvlm


_LOADERS = {
    "imagenome": load_imagenome,
    "vindr_cxr": load_vindr_cxr,
    "padchest_gr": load_padchest_gr,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True, choices=list(_LOADERS))
    ap.add_argument("--mode", choices=["direct", "cot"], default="direct")
    ap.add_argument("--radedit-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--configs", type=Path, default=Path("configs"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.configs)
    suffixes = cfg.datasets.question_suffixes
    data_root = cfg.datasets.data_root

    samples = _LOADERS[args.dataset](data_root=data_root, suffixes=suffixes)
    if args.limit:
        samples = samples[: args.limit]

    adapter = load_lvlm(args.model, device=args.device)
    runner = ThreeStepFilter(adapter, model_key=args.model)

    out_records: list[dict] = []
    for sample in tqdm(samples, desc=f"filter:{args.model}/{args.dataset}/{args.mode}"):
        idx = sample.index
        edited = args.radedit_dir / f"{idx}_edited.png"
        kept = args.radedit_dir / f"{idx}_edited_kept.png"
        kept_normal = args.radedit_dir / f"{idx}_edited_kept_normal.png"
        if not (edited.exists() and kept.exists() and kept_normal.exists()):
            continue

        try:
            rec = runner.run_one(sample, edited, kept, kept_normal, mode=args.mode)
        except FileNotFoundError:
            continue
        if rec.passes():
            out_records.append(rec.to_dict(args.mode))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out_records, open(args.out, "w"), indent=2)
    print(f"wrote {len(out_records)} retained records -> {args.out}")


if __name__ == "__main__":
    main()
