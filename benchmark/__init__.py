"""MedGround-Bench construction.

Two stages:
  1. `benchmark.radedit.pipeline` — generate foreground/background counterfactuals
     using the RadEdit inpainting pipeline.
  2. `benchmark.filter.three_step_filter` — apply correctness + flip + stay
     gates to retain only causally-grounded samples.

The user-facing CLIs that drive these stages live under `scripts/`.
"""
