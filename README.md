# MedFocus / MedGround-Bench

Code for **"Rethinking Visual Attribution for Chest X-ray Reasoning in Large
Vision Language Models"** (NeurIPS 2026).

The release ships two artifacts:

- **MedGround-Bench** — a causally validated CXR-VQA evaluation set
  (1,880 direct + 2,060 reasoning samples across 6 LVLMs and 3 source datasets).
- **MedFocus** — a concept-based causal attribution method that beats 10 baselines
  on the benchmark.

## Install

```bash
git clone <repo> medfocus && cd medfocus
pip install -e .
export MEDFOCUS_DATA_ROOT=/path/to/physionet.org/files     # or wherever the source data lives
```

Datasets are not redistributed here. See [`docs/DATASETS.md`](docs/DATASETS.md) for
PhysioNet credentials and PadChest-GR access.

## Quick start

```python
from PIL import Image
from medfocus import MedFocus, load_lvlm
from medfocus.config import load_config
from medfocus.medsam import MedSAMClient
from medfocus.ot.reference import ReferenceCXRPool

cfg = load_config()                                       # reads configs/*.yaml
adapter = load_lvlm("medgemma1_5_4b")                     # any of 6 paper LVLMs
medsam  = MedSAMClient(cfg.medfocus.medsam.model_id)
ref_pool = ReferenceCXRPool.from_directory(
    images_dir=cfg.reference_pool.images_dir,
    masks_dir=cfg.reference_pool.masks_dir,
    concepts=cfg.medfocus.concepts,
)

mf = MedFocus(adapter,
              ref_pool=ref_pool,
              medsam=medsam,
              concepts=cfg.medfocus.concepts,
              composites=cfg.medfocus.composites,
              tau=cfg.medfocus.intervention.tau)

img = Image.open("my_cxr.png").convert("L")
result = mf.attribute(img, "Is there evidence of cardiomegaly in the image?")
print(result.answer)         # "Yes" / "No"
print(result.concept)        # "cardiac silhouette"
print(result.bbox)           # (x1, y1, x2, y2)
print(result.fallback_used)  # True iff no concept exceeded tau
```

## Reproducing paper results

```bash
# 1. Pre-compute UOT concept caches once.
python scripts/precompute_ot.py --split direct --out-dir preprocess/direct/

# 2. Run every method on every model. METHODS lists the 10 baselines + MedFocus.
for m in qwen2_5_vl_3b qwen2_5_vl_7b gemma3_4b gemma3_12b medgemma_4b medgemma1_5_4b; do
  for method in attention_head attention_rollout lrp grad_weighted_attention \
                gradcam gradcampp integrated_gradients occlusion rise \
                prompting_medsam medfocus; do
    python scripts/run_attribution.py --model $m --split direct \
        --method $method --out predictions/direct/$m/$method.json
  done
done

# 3. Aggregate into Table 1.
python scripts/eval_attribution.py \
    --inputs predictions/direct/*/*.json \
    --out results/main_table.csv
```

For the reasoning split, replace `--split direct` with `--split reasoning`.
See [`docs/REPRODUCE.md`](docs/REPRODUCE.md) for full instructions, smoke
testing, and the equivalent for figures.

## Building MedGround-Bench from scratch

The released JSON in `data/medground_bench/` is sufficient to reproduce all
attribution results. To rebuild from raw datasets — for example to extend the
benchmark with a new modality — see [`docs/BENCHMARK.md`](docs/BENCHMARK.md):

```bash
python scripts/run_radedit.py --dataset imagenome --out-dir data/generated_images/imagenome/
python scripts/run_three_step_filter.py --model medgemma1_5_4b --dataset imagenome \
    --mode direct --radedit-dir data/generated_images/imagenome/ \
    --out predictions/medgemma1_5_4b/imagenome/direct.json
python scripts/build_medground_bench.py --predictions-root predictions \
    --out-dir data/medground_bench
```

## Repository layout

```
configs/         Five YAMLs: models, datasets, medfocus, radedit, reference_pool.
medfocus/        Importable package.
  attribution/   MedFocus orchestrator + 10 baselines + IoU/F1 evaluation.
  concepts/      UOT-based concept transfer + intervention scorer.
  data/          Image I/O + the four dataset loaders.
  lvlm/          Adapters covering Qwen-VL and Gemma image-text formats.
  medsam/        MedSAM HF wrapper for box-prompt mask refinement.
  ot/            Sinkhorn UOT + reference-image pool.
benchmark/       MedGround-Bench construction (RadEdit + 3-step filter).
scripts/         CLIs that drive the benchmark build, attribution, evaluation.
notebooks/       Demo + qualitative-figure regeneration.
data/            Released benchmark JSON and reference-CXR pool stubs.
```
