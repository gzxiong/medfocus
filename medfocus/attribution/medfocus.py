"""MedFocus attribution orchestrator.

Top-level usage:

    from medfocus import MedFocus, load_lvlm
    from medfocus.medsam import MedSAMClient
    from medfocus.ot import ReferenceCXRPool

    adapter = load_lvlm("medgemma1_5_4b")
    medsam = MedSAMClient()
    ref_pool = ReferenceCXRPool.from_directory(
        images_dir="data/reference_cxrs/images",
        masks_dir="data/reference_cxrs/concept_masks",
        concepts=concepts,
    )
    mf = MedFocus(adapter, ref_pool=ref_pool, medsam=medsam,
                  concepts=concepts, composites=composites)
    result = mf.attribute(image, question)
    print(result.concept, result.bbox, result.fallback_used)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
from PIL import Image

from medfocus.concepts.intervention import (
    ConceptInterventionScorer,
    InterventionScores,
)
from medfocus.concepts.transfer import transfer_concept_masks
from medfocus.lvlm.adapters import LVLMAdapter
from medfocus.lvlm.generation import generate_answer
from medfocus.medsam.client import MedSAMClient
from medfocus.ot.reference import ReferenceCXR, ReferenceCXRPool


@dataclass
class AttributionResult:
    """MedFocus output for a single (image, question) pair."""

    answer: str                                  # the model's generated answer text
    concept: str                                 # winning concept or composite group, or "__image__"
    bbox: tuple[int, int, int, int]              # (x1, y1, x2, y2) in target-image pixels
    delta: float                                 # Δ for the chosen concept
    deltas: dict[str, float] = field(default_factory=dict)
    per_token: dict[str, np.ndarray] = field(default_factory=dict)
    concept_bboxes: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)
    reference_path: Optional[str] = None
    fallback_used: bool = False                  # True when no concept exceeded τ


class MedFocus:
    """Concept-based causal attribution for medical LVLM reasoning.

    Parameters
    ----------
    adapter : LVLMAdapter
        Wrapped LVLM (e.g., from `load_lvlm("medgemma1_5_4b")`).
    ref_pool : ReferenceCXRPool
        Pool of normal CXRs used as anatomical templates.
    medsam : MedSAMClient
        For bbox-prompt mask refinement after UOT transfer.
    concepts, composites :
        Concept vocabulary and clinically meaningful unions.
    tau : float
        Whole-image fallback threshold. When all r_c = exp(-Δ_c) ≥ τ, no
        single concept dominates and we report the whole image as attribution.
    image_size : int
        Resolution at which UOT is computed (concept transfer is done at
        `transfer_grid` × `transfer_grid` after downsampling from `image_size`).
    """

    def __init__(
        self,
        adapter: LVLMAdapter,
        *,
        ref_pool: ReferenceCXRPool,
        medsam: MedSAMClient,
        concepts: list[str],
        composites: Optional[dict[str, list[str]]] = None,
        tau: float = 0.75,
        image_size: int = 224,
        transfer_grid: int = 56,
        epsilon: float = 0.05,
        lambda_marginal: float = 0.1,
        mass_quantile: float = 0.75,
        sinkhorn_max_iter: int = 500,
        sinkhorn_tol: float = 1e-6,
        intervention_baseline: str = "zero",
        batch_size: int = 4,
    ):
        self.adapter = adapter
        self.ref_pool = ref_pool
        self.medsam = medsam
        self.concepts = list(concepts)
        self.composites = dict(composites or {})
        self.tau = float(tau)
        self.image_size = int(image_size)
        self.transfer_grid = int(transfer_grid)
        self.epsilon = float(epsilon)
        self.lambda_marginal = float(lambda_marginal)
        self.mass_quantile = float(mass_quantile)
        self.sinkhorn_max_iter = int(sinkhorn_max_iter)
        self.sinkhorn_tol = float(sinkhorn_tol)
        self.scorer = ConceptInterventionScorer(
            adapter, baseline=intervention_baseline, batch_size=batch_size
        )

    # ----------------------------- main entry point -----------------------

    def attribute(
        self,
        image: Image.Image,
        question: str,
        *,
        precomputed_answer: Optional[str] = None,
        precomputed_concept_bboxes: Optional[dict[str, tuple[int, int, int, int]]] = None,
    ) -> AttributionResult:
        """Run MedFocus on a single (image, question) pair.

        `precomputed_answer` and `precomputed_concept_bboxes` short-circuit
        the corresponding stages, useful when batching reproductions.
        """
        # 1) Generate the answer to attribute.
        answer = precomputed_answer or generate_answer(self.adapter, image, question)

        # 2) Localize anatomical concepts on the target image.
        if precomputed_concept_bboxes is not None:
            concept_bboxes = dict(precomputed_concept_bboxes)
            ref_path = None
        else:
            ref, _cost = self.ref_pool.select_best(image, image_size=self.image_size)
            ref_path = ref.image_path
            ref_masks = {c: ref.masks[c] for c in self.concepts if c in ref.masks}
            concept_bboxes = transfer_concept_masks(
                ref.image,
                image,
                ref_masks,
                medsam=self.medsam,
                image_size=self.image_size,
                transfer_grid=self.transfer_grid,
                epsilon=self.epsilon,
                lambda_marginal=self.lambda_marginal,
                mass_quantile=self.mass_quantile,
                sinkhorn_max_iter=self.sinkhorn_max_iter,
                sinkhorn_tol=self.sinkhorn_tol,
            )

        # 3) Score each concept + composite group via causal intervention.
        scores: InterventionScores = self.scorer.score(
            image, question, answer, concept_bboxes, self.composites
        )

        # 4) Argmax + τ-thresholded whole-image fallback.
        if not scores.deltas:
            return self._whole_image_fallback(image, answer, ref_path, scores, concept_bboxes)

        best = max(scores.deltas, key=scores.deltas.get)
        # r_c = exp(-Δ_c); fall back when min_c r_c >= tau, i.e. max_c Δ_c <= -ln(tau).
        if math.exp(-scores.deltas[best]) >= self.tau:
            return self._whole_image_fallback(image, answer, ref_path, scores, concept_bboxes)

        bbox = self._resolve_bbox(best, concept_bboxes, image.size)
        return AttributionResult(
            answer=answer,
            concept=best,
            bbox=bbox,
            delta=scores.deltas[best],
            deltas=dict(scores.deltas),
            per_token=dict(scores.per_token),
            concept_bboxes=concept_bboxes,
            reference_path=ref_path,
            fallback_used=False,
        )

    # ----------------------------- helpers --------------------------------

    def _resolve_bbox(
        self,
        name: str,
        concept_bboxes: dict[str, tuple[int, int, int, int]],
        image_size: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        if name in concept_bboxes:
            return concept_bboxes[name]
        # Composite group → tight box of the union of its member boxes.
        members = self.composites.get(name, [])
        boxes = [concept_bboxes[m] for m in members if m in concept_bboxes]
        if not boxes:
            return (0, 0, image_size[0], image_size[1])
        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[2] for b in boxes)
        y2 = max(b[3] for b in boxes)
        return (x1, y1, x2, y2)

    def _whole_image_fallback(
        self,
        image: Image.Image,
        answer: str,
        ref_path: Optional[str],
        scores: InterventionScores,
        concept_bboxes: dict[str, tuple[int, int, int, int]],
    ) -> AttributionResult:
        W, H = image.size
        return AttributionResult(
            answer=answer,
            concept="__image__",
            bbox=(0, 0, W, H),
            delta=0.0,
            deltas=dict(scores.deltas),
            per_token=dict(scores.per_token),
            concept_bboxes=concept_bboxes,
            reference_path=ref_path,
            fallback_used=True,
        )
