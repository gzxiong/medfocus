# MedFocus / MedGround-Bench

Code for **"Rethinking Visual Attribution for Chest X-ray Reasoning in Large
Vision Language Models"**.

The release ships two artifacts:

- **MedGround-Bench** — a causally validated CXR-VQA attribution evaluation set
  (1,880 direct + 2,060 reasoning samples across 6 LVLMs and 3 source datasets).
- **MedFocus** — a concept-based causal attribution method that beats 10 baselines
  on the benchmark.

## Install

```bash
git clone https://github.com/gzxiong/medfocus.git && cd medfocus
pip install -e .
export MEDFOCUS_DATA_ROOT=/path/to/cxr/datasets
```

The CXR datasets are not redistributed. Download
[ImaGenome](https://physionet.org/content/chest-imagenome/) and
[VinDR-CXR](https://physionet.org/content/vindr-cxr/1.0.0/) from PhysioNet
(credentialing required), and PadChest-GR from [Kaggle](https://www.kaggle.com/datasets/fatihasubha/padchest-full), then place them under
`$MEDFOCUS_DATA_ROOT` following the layout in `configs/datasets.yaml`.

## Quick start

```python
from PIL import Image
from medfocus import MedFocus, load_lvlm
from medfocus.config import load_config
from medfocus.medsam import MedSAMClient
from medfocus.ot.reference import ReferenceCXRPool

cfg = load_config()
adapter = load_lvlm("medgemma1_5_4b")
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

See `notebooks/01_medfocus_walkthrough.ipynb` for an end-to-end run on a
single MedGround-Bench sample.

## Reproducing the main results

```bash
# 1. Pre-compute UOT concept caches once.
python scripts/precompute_ot.py --split direct --out-dir preprocess/direct/

# 2. Run every method on every model.
for m in qwen2_5_vl_3b qwen2_5_vl_7b gemma3_4b gemma3_12b medgemma_4b medgemma1_5_4b; do
  for method in attention_head attention_rollout lrp grad_weighted_attention \
                gradcam gradcampp integrated_gradients occlusion rise \
                prompting_medsam medfocus; do
    python scripts/run_attribution.py --model $m --split direct \
        --method $method --out predictions/direct/$m/$method.json
  done
done

# 3. Aggregate.
python scripts/eval_attribution.py \
    --inputs predictions/direct/*/*.json \
    --out results/main_table.csv
```

Replace `--split direct` with `--split reasoning` for the reasoning results.

## Rebuilding MedGround-Bench

The released JSON in `data/medground_bench/` is sufficient for reproduction.
To rebuild from raw datasets:

```bash
python scripts/run_radedit.py --dataset imagenome --out-dir data/generated_images/imagenome/
python scripts/run_three_step_filter.py --model medgemma1_5_4b --dataset imagenome \
    --mode direct --radedit-dir data/generated_images/imagenome/ \
    --out predictions/medgemma1_5_4b/imagenome/direct.json
python scripts/build_medground_bench.py --predictions-root predictions \
    --out-dir data/medground_bench
```

## Citation
If you use `MedFocus` or `MedGround-Bench`, please consider citing
```
@article{xiong2026rethinking,
    title={Rethinking Visual Attribution for Chest X-ray Reasoning in Large Vision Language Models}, 
    author={Guangzhi Xiong and Qiao Jin and Sanchit Sinha and Zhiyong Lu and Aidong Zhang},
    journal={arXiv preprint arXiv:2605.20158},
    year={2026}
}
```
