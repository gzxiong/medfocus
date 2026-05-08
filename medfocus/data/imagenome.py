"""ImaGenome (Chest ImaGenome) loader.

Reads `gold_object_attribute_with_coordinates.txt`, keeps anatomical-finding /
disease entries, groups by (patient, study, image, label) and emits one
:class:`Sample` per (image, attribute) pair following the binary-VQA template.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from medfocus.data.sample import Sample


def load_imagenome(
    data_root: str | Path,
    annotations_relpath: str = "chest-imagenome/1.0.0/gold_dataset/gold_object_attribute_with_coordinates.txt",
    suffixes: dict[str, str] | None = None,
) -> list[Sample]:
    suffixes = suffixes or {}
    df = pd.read_csv(os.path.join(data_root, annotations_relpath), sep="\t")
    df = df[
        (df["relation"] == 1)
        & ((df["categoryID"] == "anatomicalfinding") | (df["categoryID"] == "disease"))
    ].reset_index(drop=True)
    df = df[["patient_id", "study_id", "image_id", "label_name", "coord_original", "bbox", "categoryID"]]
    df = df.groupby(
        ["patient_id", "study_id", "image_id", "label_name", "categoryID"]
    ).agg({"coord_original": list, "bbox": list}).reset_index()

    samples: list[Sample] = []
    for idx in range(len(df)):
        item = df.iloc[idx].to_dict()
        patient_id = str(item["patient_id"])
        study_id = str(item["study_id"])
        image_id = str(item["image_id"])
        imgpath = os.path.join(
            data_root,
            f"mimic-cxr-jpg/2.1.0/files/p{patient_id[:2]}/p{patient_id}/s{study_id}/{image_id.replace('.dcm', '.jpg')}",
        )
        attribute = item["label_name"]
        question = f"Is there evidence of {attribute} in the image?"
        locations = [list(eval(s)) for s in item["coord_original"]]
        samples.append(
            Sample(
                index=idx,
                dataset="imagenome",
                imgpath=imgpath,
                question=question + suffixes.get("open", ""),
                question_direct=question + suffixes.get("direct", ""),
                question_cot=question + suffixes.get("cot", ""),
                answer="Yes",
                locations=locations,
                attribute=attribute,
            )
        )
    return samples
