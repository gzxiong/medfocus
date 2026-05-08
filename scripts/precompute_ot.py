"""Cache MedFocus concept bboxes per MedGround-Bench sample.

For each retained record, we run the UOT-based concept transfer once and
store the {concept: bbox} dictionary. Downstream attribution can then load
those caches and skip the expensive transfer step. This mirrors the
`preprocess/ot_medsam_v1.0/<dataset>/` cache used in the original notebooks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from medfocus.concepts.transfer import transfer_concept_masks
from medfocus.config import load_config
from medfocus.data.imagenome import load_imagenome
from medfocus.data.io import safe_open_image, resize_pad
from medfocus.data.medground_bench import attach_source_metadata, load_medground
from medfocus.data.padchest_gr import load_padchest_gr
from medfocus.data.vindr_cxr import load_vindr_cxr
from medfocus.medsam.client import MedSAMClient
from medfocus.ot.reference import ReferenceCXRPool


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["direct", "reasoning"], default="direct")
    ap.add_argument("--bench-dir", type=Path, default=Path("data/medground_bench"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--configs", type=Path, default=Path("configs"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.configs)
    image_size = int(cfg.datasets.image["resize_width"])

    records = load_medground(args.bench_dir, split=args.split)
    if args.limit:
        records = records[: args.limit]

    suffixes = cfg.datasets.question_suffixes
    sources = {
        "imagenome": load_imagenome(cfg.datasets.data_root, suffixes=suffixes),
        "vindr_cxr": load_vindr_cxr(cfg.datasets.data_root, suffixes=suffixes),
        "padchest_gr": load_padchest_gr(cfg.datasets.data_root, suffixes=suffixes),
    }
    records = attach_source_metadata(records, sources)

    medsam = MedSAMClient(cfg.medfocus.medsam.model_id)
    rp_cfg = cfg.reference_pool
    if rp_cfg is None:
        raise RuntimeError("reference_pool config is required for OT precomputation.")
    ref_pool = ReferenceCXRPool.from_directory(
        images_dir=rp_cfg.images_dir,
        masks_dir=rp_cfg.masks_dir,
        concepts=cfg.medfocus.concepts,
        candidates=rp_cfg.candidates or None,
        selection_grid=cfg.medfocus.ot.selection_grid,
        epsilon=cfg.medfocus.ot.epsilon,
        lambda_marginal=cfg.medfocus.ot.lambda_marginal,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for r in tqdm(records, desc=f"precompute_ot:{args.split}"):
        key = f"{r['dataset']}_{r['index']}"
        cache_path = args.out_dir / f"{key}.json"
        if cache_path.exists():
            continue
        img = safe_open_image(r["imgpath"])
        if img is None:
            continue
        img = resize_pad(img, image_size)
        ref, _cost = ref_pool.select_best(img, image_size=image_size)
        ref_masks = {c: ref.masks[c] for c in cfg.medfocus.concepts if c in ref.masks}
        bboxes = transfer_concept_masks(
            ref.image, img, ref_masks,
            medsam=medsam,
            image_size=image_size,
            transfer_grid=cfg.medfocus.ot.transfer_grid,
            epsilon=cfg.medfocus.ot.epsilon,
            lambda_marginal=cfg.medfocus.ot.lambda_marginal,
            mass_quantile=cfg.medfocus.ot.mass_quantile,
            sinkhorn_max_iter=cfg.medfocus.ot.sinkhorn_max_iter,
            sinkhorn_tol=cfg.medfocus.ot.sinkhorn_tol,
        )
        cache_path.write_text(json.dumps({
            "dataset": r["dataset"], "index": r["index"], "reference_path": ref.image_path,
            "concept_bboxes": {c: list(b) for c, b in bboxes.items()},
        }))


if __name__ == "__main__":
    main()
