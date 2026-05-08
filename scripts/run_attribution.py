"""Run any attribution method on (a slice of) MedGround-Bench.

For each retained sample matching `--model` and `--split`, we:
  1. open the source CXR (resize+pad to 224);
  2. run the chosen attribution method;
  3. convert the heatmap (or returned boxes) into evaluation boxes;
  4. write one JSON record per sample to `--out`.

Available methods (10 baselines + MedFocus):
    attention_head, attention_rollout, lrp, grad_weighted_attention,
    gradcam, gradcampp, integrated_gradients, occlusion, rise,
    prompting_medsam, medfocus
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from medfocus.attribution import MedFocus
from medfocus.attribution.baselines import METHODS
from medfocus.attribution.eval import bboxes_rescale, heatmap_to_bboxes_quantile
from medfocus.config import load_config
from medfocus.data.imagenome import load_imagenome
from medfocus.data.io import resize_pad, safe_open_image
from medfocus.data.medground_bench import attach_source_metadata, load_medground
from medfocus.data.padchest_gr import load_padchest_gr
from medfocus.data.vindr_cxr import load_vindr_cxr
from medfocus.lvlm import load_lvlm
from medfocus.medsam.client import MedSAMClient
from medfocus.ot.reference import ReferenceCXRPool


_DATASET_LOADERS = {
    "imagenome": load_imagenome,
    "vindr_cxr": load_vindr_cxr,
    "padchest_gr": load_padchest_gr,
}


def _heatmap_to_boxes(heatmap, target_size, native_size, q: float = 0.9) -> list[list[int]]:
    comps = heatmap_to_bboxes_quantile(heatmap, q=q)
    boxes = [list(c["bbox"]) for c in comps]
    return bboxes_rescale(boxes, orig_size=native_size, heatmap_size=target_size)


def _run_baseline(method: str, adapter, img, question, answer, *, medsam=None) -> list[list[int]]:
    fn, returns_heatmap = METHODS[method]
    if returns_heatmap:
        heatmap = fn(adapter, img, question, answer)
        return _heatmap_to_boxes(heatmap, target_size=img.size, native_size=img.size)
    if method == "prompting_medsam":
        return [list(b) for b in fn(adapter, img, question, answer, medsam=medsam)]
    return [list(b) for b in fn(adapter, img, question, answer)]


def _run_medfocus(mf: MedFocus, img, question, answer):
    res = mf.attribute(img, question, precomputed_answer=answer)
    return [list(res.bbox)], res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", choices=["direct", "reasoning"], default="direct")
    ap.add_argument("--method", required=True,
                    choices=list(METHODS.keys()) + ["medfocus"])
    ap.add_argument("--bench-dir", type=Path, default=Path("data/medground_bench"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--configs", type=Path, default=Path("configs"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--attn-impl", default="eager",
                    help='"eager" enables output_attentions=True (needed by attention/grad-weighted methods).')
    args = ap.parse_args()

    cfg = load_config(args.configs)
    image_size = int(cfg.datasets.image["resize_width"])

    records = load_medground(args.bench_dir, split=args.split)
    records = [r for r in records if r["model"] == args.model]
    if args.limit:
        records = records[: args.limit]

    suffixes = cfg.datasets.question_suffixes
    sources = {
        "imagenome": load_imagenome(cfg.datasets.data_root, suffixes=suffixes),
        "vindr_cxr": load_vindr_cxr(cfg.datasets.data_root, suffixes=suffixes),
        "padchest_gr": load_padchest_gr(cfg.datasets.data_root, suffixes=suffixes),
    }
    records = attach_source_metadata(records, sources)

    adapter = load_lvlm(args.model, device=args.device, attn_implementation=args.attn_impl)

    medsam = None
    mf = None
    if args.method == "medfocus" or args.method == "prompting_medsam":
        medsam = MedSAMClient(cfg.medfocus.medsam.model_id)
    if args.method == "medfocus":
        rp_cfg = cfg.reference_pool
        if rp_cfg is None:
            raise RuntimeError("reference_pool config required for medfocus.")
        ref_pool = ReferenceCXRPool.from_directory(
            images_dir=rp_cfg.images_dir,
            masks_dir=rp_cfg.masks_dir,
            concepts=cfg.medfocus.concepts,
            candidates=rp_cfg.candidates or None,
            selection_grid=cfg.medfocus.ot.selection_grid,
            epsilon=cfg.medfocus.ot.epsilon,
            lambda_marginal=cfg.medfocus.ot.lambda_marginal,
        )
        mf = MedFocus(
            adapter, ref_pool=ref_pool, medsam=medsam,
            concepts=cfg.medfocus.concepts,
            composites=cfg.medfocus.composites,
            tau=cfg.medfocus.intervention.tau,
            image_size=image_size,
            transfer_grid=cfg.medfocus.ot.transfer_grid,
            epsilon=cfg.medfocus.ot.epsilon,
            lambda_marginal=cfg.medfocus.ot.lambda_marginal,
            mass_quantile=cfg.medfocus.ot.mass_quantile,
            sinkhorn_max_iter=cfg.medfocus.ot.sinkhorn_max_iter,
            sinkhorn_tol=cfg.medfocus.ot.sinkhorn_tol,
            intervention_baseline=cfg.medfocus.intervention.baseline,
        )

    out_records: list[dict] = []
    for r in tqdm(records, desc=f"attribute:{args.method}"):
        img = safe_open_image(r["imgpath"])
        if img is None:
            continue
        img224 = resize_pad(img, image_size)
        question = r.get("question") or r.get("question_direct") or r.get("question_cot")
        answer = r.get("prediction_direct") or r.get("prediction_cot") or r["answer"]

        if args.method == "medfocus":
            pred_boxes, mf_result = _run_medfocus(mf, img224, question, answer)
            extras = {"concept": mf_result.concept, "fallback_used": mf_result.fallback_used}
        else:
            pred_boxes = _run_baseline(args.method, adapter, img224, question, answer, medsam=medsam)
            extras = {}

        out_records.append({
            "method": args.method,
            "model": args.model,
            "split": args.split,
            "dataset": r["dataset"],
            "index": r["index"],
            "pred_boxes": pred_boxes,
            "gt_boxes": r.get("locations", []),
            **extras,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out_records, open(args.out, "w"), indent=2)
    print(f"wrote {len(out_records)} records -> {args.out}")


if __name__ == "__main__":
    main()
