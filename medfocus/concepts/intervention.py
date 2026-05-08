"""Causal intervention scoring for MedFocus.

Given the LVLM's original answer to (image, question), we score each concept
by zero-masking its bounding box and measuring the cumulative drop in
teacher-forced answer log-probabilities:

    Δ_c = Σ_t max(0, log p(y_t | x, q, y_<t) - log p(y_t | x̃_c, q, y_<t))

A higher Δ_c means concept c is more causally responsible for the prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from medfocus.data.io import mask_image_regions
from medfocus.lvlm.adapters import LVLMAdapter


# ----------------------------- batched scoring ------------------------------

@torch.inference_mode()
def batched_teacher_forced_logprobs(
    adapter: LVLMAdapter,
    images: list[Image.Image],
    question: str,
    answer: str,
    *,
    batch_size: int = 4,
) -> torch.Tensor:
    """Return per-token log-probs of `answer` for each image in `images`.

    Shape (N, T) where T is the answer-token count under the adapter's
    processor. Uses one teacher-forced forward pass per image (batched
    internally) and avoids regenerating, so each call is O(N * forward).
    """
    if not images:
        return torch.empty((0, 0))

    # Build a reference (user, full) input pair on the first image to discover
    # the answer span; assume all images share the same prompt structure.
    user_inputs, ref_inputs = adapter.build_teacher_forced_inputs(images[0], question, answer)
    ts = int(user_inputs["input_ids"].shape[1])
    te = int(ref_inputs["input_ids"].shape[1])
    target_ids = ref_inputs["input_ids"][:, ts:te]  # (1, T)

    out_logprobs: list[torch.Tensor] = []
    for i in range(0, len(images), batch_size):
        chunk = images[i : i + batch_size]
        batched: list[dict] = []
        for img in chunk:
            _, full_inputs = adapter.build_teacher_forced_inputs(img, question, answer)
            batched.append(full_inputs)
        # Concatenate along batch dim (assumes identical text -> identical lengths).
        keys = batched[0].keys()
        merged = {k: torch.cat([b[k] for b in batched], dim=0) for k in keys}

        out = adapter.model(**merged, return_dict=True, use_cache=False)
        logp = F.log_softmax(out.logits[:, ts - 1 : te - 1, :], dim=-1)  # (B, T, V)
        gathered = logp.gather(-1, target_ids.expand(logp.shape[0], -1).unsqueeze(-1)).squeeze(-1)
        out_logprobs.append(gathered.float().detach().cpu())
    return torch.cat(out_logprobs, dim=0)  # (N, T)


# ----------------------------- intervention API -----------------------------

@dataclass
class InterventionScores:
    """Output of :class:`ConceptInterventionScorer.score`.

    deltas: concept_or_group -> Δ (cumulative log-prob drop, >= 0).
    per_token: concept_or_group -> per-token max(0, drop) of shape (T,).
    original_logprobs: (T,) baseline log-probs on the unperturbed image.
    """
    deltas: dict[str, float] = field(default_factory=dict)
    per_token: dict[str, np.ndarray] = field(default_factory=dict)
    original_logprobs: np.ndarray = field(default_factory=lambda: np.zeros((0,)))


class ConceptInterventionScorer:
    """Score concepts by zero-masking and measuring teacher-forced log-prob drops."""

    def __init__(
        self,
        adapter: LVLMAdapter,
        *,
        baseline: str = "zero",
        batch_size: int = 4,
    ):
        if baseline not in ("zero", "mean"):
            raise ValueError("baseline must be 'zero' or 'mean'")
        self.adapter = adapter
        self.baseline = baseline
        self.batch_size = int(batch_size)

    def _make_baseline_image(self, img: Image.Image) -> Image.Image:
        if self.baseline == "zero":
            return Image.new(img.mode, img.size, 0)
        arr = np.array(img.convert("L"))
        return Image.new("L", img.size, int(arr.mean()))

    def _occlude(self, img: Image.Image, boxes: Iterable[tuple[int, int, int, int]]) -> Image.Image:
        """Replace pixels inside `boxes` with the baseline image's pixels."""
        # For "zero" baseline this is identical to mask_image_regions.
        if self.baseline == "zero":
            return mask_image_regions(img, boxes)
        baseline_img = self._make_baseline_image(img)
        out = img.copy()
        out_arr = np.array(out)
        base_arr = np.array(baseline_img)
        for x1, y1, x2, y2 in boxes:
            out_arr[y1:y2, x1:x2] = base_arr[y1:y2, x1:x2]
        return Image.fromarray(out_arr)

    def score(
        self,
        img: Image.Image,
        question: str,
        answer: str,
        concept_bboxes: dict[str, tuple[int, int, int, int]],
        composite_groups: Optional[dict[str, list[str]]] = None,
    ) -> InterventionScores:
        """Compute Δ_c for every concept and composite group."""
        composite_groups = composite_groups or {}

        names: list[str] = ["__original__"] + list(concept_bboxes.keys()) + list(composite_groups.keys())
        images: list[Image.Image] = [img]
        for c in concept_bboxes:
            images.append(self._occlude(img, [concept_bboxes[c]]))
        for g, members in composite_groups.items():
            boxes = [concept_bboxes[m] for m in members if m in concept_bboxes]
            if not boxes:
                # No member resolved; reuse original (Δ will be 0).
                images.append(img)
            else:
                images.append(self._occlude(img, boxes))

        all_logp = batched_teacher_forced_logprobs(
            self.adapter, images, question, answer, batch_size=self.batch_size
        )  # (N, T)
        original = all_logp[0].numpy()
        out = InterventionScores(original_logprobs=original)
        for i, name in enumerate(names[1:], start=1):
            drop = (original - all_logp[i].numpy()).clip(min=0.0)
            out.per_token[name] = drop
            out.deltas[name] = float(drop.sum())
        return out
