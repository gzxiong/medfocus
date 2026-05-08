"""VinDR-CXR loader.

Reads the test-split annotations CSV, drops "No finding" rows, groups multiple
boxes per (image, class), and yields one :class:`Sample` per (image, class)."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from medfocus.data.sample import Sample


def load_vindr_cxr(
    data_root: str | Path,
    annotations_relpath: str = "vindr-cxr/1.0.0/annotations/annotations_test.csv",
    images_dir_relpath: str = "vindr-cxr/1.0.0/test",
    suffixes: dict[str, str] | None = None,
) -> list[Sample]:
    suffixes = suffixes or {}
    df = pd.read_csv(os.path.join(data_root, annotations_relpath))
    df = df[df["class_name"] != "No finding"].copy()
    df["locations"] = df.apply(
        lambda r: [round(r["x_min"]), round(r["y_min"]), round(r["x_max"]), round(r["y_max"])], axis=1
    )
    df = df.groupby(["image_id", "class_name"]).agg({"locations": list}).reset_index()

    samples: list[Sample] = []
    for idx in range(len(df)):
        item = df.iloc[idx].to_dict()
        image_id = item["image_id"]
        imgpath = os.path.join(data_root, images_dir_relpath, f"{image_id}.dicom")
        attribute = item["class_name"]
        question = f"Is there evidence of {attribute} in the image?"
        samples.append(
            Sample(
                index=idx,
                dataset="vindr_cxr",
                imgpath=imgpath,
                question=question + suffixes.get("open", ""),
                question_direct=question + suffixes.get("direct", ""),
                question_cot=question + suffixes.get("cot", ""),
                answer="Yes",
                locations=item["locations"],
                attribute=attribute,
            )
        )
    return samples
