# Reproducing paper results

Three reproduction tiers, fastest to slowest.

## 1. Smoke test

Validates that the package end-to-end works on your machine. Runs MedFocus
+ two baselines on 6 samples (2 per dataset).

```bash
bash scripts/e2e_smoke.sh
```

Expected output: a small CSV in `results/smoke/main_table.csv` with non-empty
rows for `medfocus`, `gradcam`, `attention_rollout`. IoUs may be noisy at
this sample size — the smoke test asserts shape, not paper numbers.

## 2. 100-sample sanity

Reproduces headline MedFocus IoU within ±0.02 of paper Table 1.

```bash
python scripts/run_attribution.py --model medgemma1_5_4b --split direct \
    --method medfocus --limit 100 --out results/sanity/medfocus.json
python scripts/eval_attribution.py --inputs results/sanity/medfocus.json
```

## 3. Full reproduction

Reproduces Tables 1, 2, and Figure 4 in full. Run for both splits and all
six LVLMs. Caching the UOT step once amortizes the cost across the 11
attribution methods.

```bash
# (a) Cache UOT concept bboxes per sample.
python scripts/precompute_ot.py --split direct  --out-dir preprocess/direct/
python scripts/precompute_ot.py --split reasoning --out-dir preprocess/reasoning/

# (b) Run all (method × model × split) combinations.
SPLITS="direct reasoning"
MODELS="qwen2_5_vl_3b qwen2_5_vl_7b gemma3_4b gemma3_12b medgemma_4b medgemma1_5_4b"
METHODS="attention_head attention_rollout lrp grad_weighted_attention \
         gradcam gradcampp integrated_gradients occlusion rise \
         prompting_medsam medfocus"

for split in $SPLITS; do
  for m in $MODELS; do
    for method in $METHODS; do
      python scripts/run_attribution.py --model $m --split $split \
          --method $method --out predictions/$split/$m/$method.json
    done
  done
done

# (c) Aggregate to Tables 1/2.
python scripts/eval_attribution.py \
    --inputs predictions/direct/*/*.json \
    --out results/main_table_direct.csv
python scripts/eval_attribution.py \
    --inputs predictions/reasoning/*/*.json \
    --out results/main_table_reasoning.csv
```

## Mapping to paper artifacts

| Artifact            | Reproduction command                                                       |
|---------------------|-----------------------------------------------------------------------------|
| Table 1 (direct)    | `eval_attribution.py --inputs predictions/direct/*/*.json`                  |
| Figure 4 (reasoning)| `eval_attribution.py --inputs predictions/reasoning/*/*.json`               |
| Figure 5 (case)     | `notebooks/02_qualitative_figures.ipynb`                                    |
| Ablation tables     | flip `MedFocus(... mass_quantile=..., medsam_refine=...)` and rerun         |
