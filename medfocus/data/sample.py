"""Typed Sample container shared across dataset loaders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Sample:
    """One CXR-VQA item.

    Fields:
      index: position within the source dataset (stable across runs).
      dataset: source-dataset key ("imagenome" | "vindr_cxr" | "padchest_gr").
      imgpath: absolute path to the CXR image.
      question: full question text including the suffix variant in use.
      answer: gold yes/no answer (always "Yes" for the abnormality VQA).
      locations: list of [x1, y1, x2, y2] bboxes giving expert-annotated evidence.
      attribute: parsed attribute name (e.g., "cardiomegaly"); used by RadEdit.
      question_direct / question_cot: pre-built question variants for the two output modes.
    """

    index: int
    dataset: str
    imgpath: str
    question: str
    answer: str
    locations: list[list[int]] = field(default_factory=list)
    attribute: Optional[str] = None
    question_direct: Optional[str] = None
    question_cot: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "dataset": self.dataset,
            "imgpath": self.imgpath,
            "question": self.question,
            "answer": self.answer,
            "locations": self.locations,
            "attribute": self.attribute,
            "question_direct": self.question_direct,
            "question_cot": self.question_cot,
        }
