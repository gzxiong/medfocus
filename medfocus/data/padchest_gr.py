"""PadChest-GR loader.

Builds binary-VQA samples from the abnormality findings of the first 2,000
patients in `filtered_studies.json`, joining the relative bbox coordinates with
each image's pixel size."""

from __future__ import annotations

import json
import os
from pathlib import Path

from tqdm import tqdm

from medfocus.data.io import safe_open_image
from medfocus.data.sample import Sample


def load_padchest_gr(
    data_root: str | Path,
    studies_json_relpath: str = "filtered_studies.json",
    images_dir_relpath: str = "PadChest_GR",
    split: str = "test",
    suffixes: dict[str, str] | None = None,
) -> list[Sample]:
    suffixes = suffixes or {}
    data = json.load(open(os.path.join(data_root, studies_json_relpath)))
    if split == "test":
        num_samples = 2000
        data_selected = data[:num_samples]
        save_path = os.path.join(data_root, f"filtered_data_{num_samples}.json")
    elif split == "train":
        num_samples = 100
        data_selected = data[-num_samples:][::-1]
        save_path = os.path.join(data_root, f"filtered_data_train_{num_samples}.json")
    else:
        raise ValueError("split must be 'train' or 'test'")

    if not os.path.exists(save_path):
        items = []
        for item in tqdm(data_selected, desc="building padchest_gr items"):
            imgpath_rel = os.path.join(images_dir_relpath, item["ImageID"])
            img = None
            for q_item in item["findings"]:
                if not q_item["abnormal"]:
                    continue
                if not q_item["boxes"] and not q_item["extra_boxes"]:
                    continue
                if img is None:
                    img = safe_open_image(os.path.join(data_root, imgpath_rel))
                    if img is None:
                        break
                attribute = " or ".join(q_item["labels"])
                question = f"Is there evidence of {attribute} in the image?"
                locations = [
                    [
                        round(loc[0] * img.size[0]),
                        round(loc[1] * img.size[1]),
                        round(loc[2] * img.size[0]),
                        round(loc[3] * img.size[1]),
                    ]
                    for loc in q_item["boxes"] + q_item["extra_boxes"]
                ]
                items.append({
                    "imgpath": imgpath_rel,
                    "question": question,
                    "attribute": attribute,
                    "answer": "Yes",
                    "locations": locations,
                })
        json.dump(items, open(save_path, "w"), indent=4)
    else:
        items = json.load(open(save_path))

    samples: list[Sample] = []
    for idx, it in enumerate(items):
        question = it["question"]
        samples.append(
            Sample(
                index=idx,
                dataset="padchest_gr",
                imgpath=os.path.join(data_root, it["imgpath"]),
                question=question + suffixes.get("open", ""),
                question_direct=question + suffixes.get("direct", ""),
                question_cot=question + suffixes.get("cot", ""),
                answer=it["answer"],
                locations=it["locations"],
                attribute=it.get("attribute"),
            )
        )
    return samples
