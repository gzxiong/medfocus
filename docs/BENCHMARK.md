# Building MedGround-Bench

The benchmark applies a 3-step **causal filter** that keeps only the
binary-VQA samples for which the expert-annotated region is *causally*
responsible for the LVLM's prediction.

## Pipeline

For each (image, attribute) sample:

1. **Correctness.** Ask the LVLM the question on the original CXR. Keep only
   if the answer is correct (`Yes`).
2. **Foreground counterfactual.** Use RadEdit (`microsoft/radedit`) to inpaint
   the annotated bounding box with the prompt `"No {attribute}"`. Keep only
   if the LVLM **flips** to `No`.
3. **Background counterfactual.** Inpaint the *complement* of the bbox under
   the same prompt and again under `"No abnormality"`. Keep only if the LVLM
   answer **stays** `Yes` in both cases.

A sample passes iff all three gates pass for the same model. Per-model
filtering produces the (model, dataset, index, predictions) tuples shipped in
`data/medground_bench/{direct,reasoning}.json`.

## Rebuilding the benchmark

Required: source datasets in place (`docs/DATASETS.md`), a GPU, a HuggingFace
token with access to the LVLMs and the RadEdit pipeline.

```bash
# (a) Generate counterfactual images. Each pass writes 3 PNGs per sample.
python scripts/run_radedit.py --dataset imagenome \
    --out-dir data/generated_images/imagenome/

# (b) Run the LVLM forwards and apply the 3 gates per (model, dataset, mode).
for m in qwen2_5_vl_3b qwen2_5_vl_7b gemma3_4b gemma3_12b medgemma_4b medgemma1_5_4b; do
  for d in imagenome vindr_cxr padchest_gr; do
    for mode in direct cot; do
      python scripts/run_three_step_filter.py \
          --model $m --dataset $d --mode $mode \
          --radedit-dir data/generated_images/$d/ \
          --out predictions/$m/$d/$mode.json
    done
  done
done

# (c) Concatenate per-(model, dataset) outputs into the released JSON.
python scripts/build_medground_bench.py \
    --predictions-root predictions \
    --out-dir data/medground_bench
```

## Schema

See [`data/medground_bench/SCHEMA.md`](../data/medground_bench/SCHEMA.md) for
field-by-field documentation, including how to join records back to the source
dataset via `(dataset, index)`.

## Retention rates

The retention rates reported in the paper are roughly 1.5%–20% per
(model, dataset, mode). The strictness comes from gate (3): many correct
predictions rely on cues outside the annotated region. See Appendix
`tab:per_dataset_model_breakdown` of the paper for the full breakdown.
