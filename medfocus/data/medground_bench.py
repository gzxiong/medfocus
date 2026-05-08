"""MedGround-Bench loader.

The released `data/medground_bench/{direct,reasoning}.json` files store one
record per (model, dataset, source_index) tuple together with the four
predictions used by the 3-step causal filter:

```
{
    "model": "Qwen/Qwen2.5-VL-3B-Instruct",
    "dataset": "imagenome",
    "index": 43,
    "answer": "Yes",
    "prediction_direct": "Yes",
    "prediction_edited_direct": "No",
    "prediction_edited_kept_direct": "Yes",
    "prediction_edited_kept_normal_direct": "Yes",
    "imgpath": "...",          # populated by `attach_source_metadata`
    "question": "Is there evidence of cardiomegaly in the image?",
    "locations": [[x1,y1,x2,y2], ...],
    "attribute": "cardiomegaly"
}
```

`load_medground` reads the JSON; `attach_source_metadata` joins to the original
dataset loader to recover image path / question / boxes for samples that don't
have those fields baked in (the released JSON ships them denormalized).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Literal

Split = Literal["direct", "reasoning"]


def load_medground(
    json_dir: str | Path = "data/medground_bench",
    split: Split = "direct",
) -> list[dict]:
    path = Path(json_dir) / f"{split}.json"
    with open(path, "r") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise TypeError(f"{path} must contain a JSON list")
    return records


def attach_source_metadata(
    records: Iterable[dict],
    source_loaders: dict[str, list],
) -> list[dict]:
    """Backfill `imgpath / question / locations / attribute` from the source dataset.

    `source_loaders` maps dataset key -> list[Sample] so we can index by `record["index"]`.
    Records that already have these fields are left unchanged.
    """
    out = []
    for r in records:
        if all(k in r for k in ("imgpath", "question", "locations")):
            out.append(r)
            continue
        src = source_loaders.get(r["dataset"])
        if src is None or r["index"] >= len(src):
            out.append(r)
            continue
        s = src[r["index"]]
        merged = dict(r)
        merged.setdefault("imgpath", s.imgpath)
        merged.setdefault("question", s.question_direct or s.question)
        merged.setdefault("locations", s.locations)
        merged.setdefault("attribute", s.attribute)
        out.append(merged)
    return out
