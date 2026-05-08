# MedFocus algorithm

MedFocus produces a **spatial bbox**, **concept name**, and **per-token
attribution** for any LVLM answer to a CXR-VQA question. It does not need
access to the LVLM's gradients or attentions — only its forward pass.

## Pipeline

```
                     reference CXRs (16 normals,
                     all 11 concepts annotated)
                              │
target CXR ──► UOT ──► concept transfer ──► MedSAM ──► concept bboxes
                              │                            │
                              │                            ▼
              question, answer ──► zero-mask each bbox, run one
                                   teacher-forced forward pass per
                                   concept, score by Σ_t max(0, Δ log p)
                                                          │
                                                          ▼
                                             argmax → bbox + concept
                                             (or whole-image fallback if
                                             min_c r_c ≥ tau)
```

## Concept vocabulary

11 anatomical regions defined in ImaGenome:

```
cardiac silhouette, left lung, right lung,
mediastinum, upper mediastinum,
left clavicle, right clavicle,
left hilar structures, right hilar structures,
left costophrenic angle, right costophrenic angle
```

Plus 4 clinically meaningful unions evaluated alongside individual concepts:
`{left, right} lung`, `{left, right} clavicle`,
`{left, right} hilar structures`, `{left, right} costophrenic angle`.

## Reference pool

A "reference" CXR is a normal image with masks for all 11 concepts. We use
16 normal cases from ImaGenome whose annotations cover the full vocabulary.
At inference time the pool's `select_best(target_image)` picks the reference
whose UOT cost to the target is minimized (computed at the coarse 14×14 grid).

To extract the pool from your local ImaGenome copy, see
`scripts/extract_reference_pool.py` (a single-shot utility that filters
ImaGenome normals to those with all 11 concept annotations).

## Hyperparameters

`configs/medfocus.yaml` defaults (matching the paper):

| Parameter | Value |
|-----------|-------|
| ε (entropic regularization) | 0.05 |
| λ (marginal relaxation, both sides) | 0.1 |
| Reference selection grid | 14×14 |
| Concept transfer grid | 56×56 |
| Sinkhorn iterations | up to 500 (tol 1e-6) |
| Per-concept mass quantile | 0.75 |
| Intervention baseline | zero |
| Whole-image fallback τ | 0.75 |

## Public API

```python
from medfocus import MedFocus

mf = MedFocus(adapter, ref_pool=ref_pool, medsam=medsam,
              concepts=concepts, composites=composites,
              tau=0.75)
result = mf.attribute(image, question)
# result.answer            : str  — model's generated answer
# result.concept           : str  — winning concept (e.g. "cardiac silhouette")
# result.bbox              : (x1, y1, x2, y2)
# result.delta             : Δ for the winning concept
# result.deltas            : dict[concept, Δ] for all candidates
# result.per_token         : dict[concept, np.ndarray of shape (T,)]
# result.fallback_used     : True iff no concept exceeded τ
# result.reference_path    : which reference CXR was selected
# result.concept_bboxes    : per-concept localized bboxes on the target image
```

You can short-circuit expensive stages with `precomputed_answer=...` (skip
generation) or `precomputed_concept_bboxes=...` (skip UOT + MedSAM, e.g.
when reading from a `precompute_ot.py` cache).
