# Installation

## Python environment

The package targets Python 3.10+ and was tested on PyTorch 2.4 / 2.9 with
CUDA 12. Other versions usually work; the only hard pin is `transformers==4.57.3`,
which is the version we used to evaluate every LVLM in the paper. Newer
transformers releases occasionally change the chat-template API for Gemma /
Qwen-VL — pin to 4.57.3 if you want exact reproduction.

```bash
git clone <repo> medfocus
cd medfocus
pip install -e .
```

## GPU memory

| Pipeline stage          | 1× GPU mem | Notes                                      |
|------------------------|-----------:|--------------------------------------------|
| Qwen2.5-VL-3B inference |  ~12 GB    | bf16, attn_implementation="eager"          |
| Qwen2.5-VL-7B          |  ~22 GB    |                                            |
| Gemma-3-12B            |  ~30 GB    | use device_map="auto" to shard if needed   |
| MedSAM box-prompt      |   ~3 GB    | shared across all attribution calls        |
| RadEdit inpainting     |  ~14 GB    | `microsoft/radedit` UNet + BiomedVLP       |

Attribution methods using `output_attentions=True` (Attention Head, Rollout,
LRP, Gradient-weighted Attention) require the eager attention kernel. We load
all models with `attn_implementation="eager"` by default; pass
`--attn-impl sdpa` to `scripts/run_attribution.py` if you only run MedFocus
or gradient-only methods.

## Reference-CXR pool

MedFocus needs a directory of normal CXRs annotated with the 11 anatomical
concepts. We ship the *configuration* but not the images themselves; you build
the pool once via:

```bash
python scripts/extract_reference_pool.py            # extracts 16 ImaGenome normals
```

(See [`docs/MEDFOCUS.md`](MEDFOCUS.md#reference-pool) for details.)
