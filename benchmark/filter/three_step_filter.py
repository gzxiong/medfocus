"""3-step causal filter for MedGround-Bench.

A sample is *retained* iff:
  1. the LVLM answers the original VQA correctly (`yes` for our positive set);
  2. its answer flips (`yes` -> `no`) on the foreground-edited counterfactual;
  3. its answer remains unchanged on both background counterfactuals
     (one with the same "No {attribute}" prompt and one with "No abnormality").

This module assumes RadEdit-edited images and original predictions already exist.
For each (model, sample) pair we issue four LVLM calls and collect the four
predictions. `ThreeStepFilter` then applies the gates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Optional

from PIL import Image

from medfocus.data.io import safe_open_image
from medfocus.data.sample import Sample
from medfocus.lvlm.adapters import LVLMAdapter
from medfocus.lvlm.generation import generate_answer

Mode = Literal["direct", "cot"]


def is_yes(text: str) -> bool:
    """Heuristic yes/no parser used in `1_llm_inference.ipynb`."""
    if text is None:
        return False
    t = text.strip().lower()
    return bool(re.search(r"\byes\b", t)) and not re.search(r"\bno\b\s*$", t)


def is_no(text: str) -> bool:
    if text is None:
        return False
    t = text.strip().lower()
    return bool(re.search(r"\bno\b", t)) and not re.search(r"\byes\b\s*$", t)


@dataclass
class FilterRecord:
    """One filter row, mirroring the released MedGround-Bench schema."""
    model: str
    dataset: str
    index: int
    answer: str
    prediction: str
    prediction_edited: str
    prediction_edited_kept: str
    prediction_edited_kept_normal: str

    def passes(self) -> bool:
        return (
            is_yes(self.prediction)
            and is_no(self.prediction_edited)
            and is_yes(self.prediction_edited_kept)
            and is_yes(self.prediction_edited_kept_normal)
        )

    def to_dict(self, mode: Mode) -> dict:
        suffix = mode  # "direct" or "cot"
        return {
            "model": self.model,
            "dataset": self.dataset,
            "index": self.index,
            "answer": self.answer,
            f"prediction_{suffix}": self.prediction,
            f"prediction_edited_{suffix}": self.prediction_edited,
            f"prediction_edited_kept_{suffix}": self.prediction_edited_kept,
            f"prediction_edited_kept_normal_{suffix}": self.prediction_edited_kept_normal,
        }


class ThreeStepFilter:
    """Run the four LVLM forwards and apply the 3-step gate."""

    def __init__(self, adapter: LVLMAdapter, model_key: str):
        self.adapter = adapter
        self.model_key = model_key

    def run_one(
        self,
        sample: Sample,
        edited_path: str | Path,
        bg_kept_path: str | Path,
        bg_kept_normal_path: str | Path,
        *,
        mode: Mode = "direct",
    ) -> FilterRecord:
        question = sample.question_direct if mode == "direct" else sample.question_cot
        if question is None:
            question = sample.question

        original = safe_open_image(sample.imgpath)
        edited = safe_open_image(edited_path)
        bg_kept = safe_open_image(bg_kept_path)
        bg_kept_normal = safe_open_image(bg_kept_normal_path)
        if any(im is None for im in (original, edited, bg_kept, bg_kept_normal)):
            raise FileNotFoundError(f"Missing image(s) for sample {sample.index} ({sample.dataset})")

        return FilterRecord(
            model=self.model_key,
            dataset=sample.dataset,
            index=sample.index,
            answer=sample.answer,
            prediction=generate_answer(self.adapter, original, question),
            prediction_edited=generate_answer(self.adapter, edited, question),
            prediction_edited_kept=generate_answer(self.adapter, bg_kept, question),
            prediction_edited_kept_normal=generate_answer(self.adapter, bg_kept_normal, question),
        )
