"""Generate RadEdit foreground and background counterfactuals for a dataset.

Outputs three images per sample:
    {idx}_edited.png            (foreground inpainted: "No {attribute}")
    {idx}_edited_kept.png       (background inpainted: "No {attribute}")
    {idx}_edited_kept_normal.png (background inpainted: "No abnormality")
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from benchmark.radedit.pipeline import RadEditConfig, RadEditPipeline
from medfocus.config import load_config
from medfocus.data.imagenome import load_imagenome
from medfocus.data.io import safe_open_image
from medfocus.data.padchest_gr import load_padchest_gr
from medfocus.data.vindr_cxr import load_vindr_cxr


_LOADERS = {
    "imagenome": load_imagenome,
    "vindr_cxr": load_vindr_cxr,
    "padchest_gr": load_padchest_gr,
}


def _attribute(sample) -> str:
    if sample.attribute:
        return sample.attribute
    # Fallback: parse from "Is there evidence of <attr> in the image?".
    q = sample.question
    return q.split("evidence of ")[1].split(" in the image")[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=list(_LOADERS))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--configs", type=Path, default=Path("configs"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None, help="Process at most N samples.")
    args = ap.parse_args()

    cfg = load_config(args.configs)
    suffixes = cfg.datasets.question_suffixes
    data_root = cfg.datasets.data_root

    samples = _LOADERS[args.dataset](data_root=data_root, suffixes=suffixes)
    if args.limit:
        samples = samples[: args.limit]

    radedit = RadEditPipeline(cfg=RadEditConfig(image_size=cfg.datasets.image["resize_width"]),
                              device=args.device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for sample in tqdm(samples, desc=f"radedit:{args.dataset}"):
        idx = sample.index
        edited_path = args.out_dir / f"{idx}_edited.png"
        kept_path = args.out_dir / f"{idx}_edited_kept.png"
        kept_normal_path = args.out_dir / f"{idx}_edited_kept_normal.png"
        if edited_path.exists() and kept_path.exists() and kept_normal_path.exists():
            continue

        img = safe_open_image(sample.imgpath)
        if img is None:
            continue
        boxes = sample.locations
        attr = _attribute(sample)
        prompt_attr = f"No {attr}"

        if not edited_path.exists():
            radedit.inpaint(img, boxes, prompt_attr, mode="fg").save(edited_path)
        if not kept_path.exists():
            radedit.inpaint(img, boxes, prompt_attr, mode="bg").save(kept_path)
        if not kept_normal_path.exists():
            radedit.inpaint(img, boxes, "No abnormality", mode="bg").save(kept_normal_path)


if __name__ == "__main__":
    main()
