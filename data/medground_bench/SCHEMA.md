# MedGround-Bench JSON schema

## Files
- `direct.json` — 1,880 records (direct yes/no mode).
- `reasoning.json` — 2,060 records (chain-of-thought mode).

## Record format

Each record is the (model, dataset, source-index, answer, predictions) tuple
that survived the 3-step causal filter:

```json
{
  "model": "qwen2_5_vl_3b",
  "dataset": "imagenome",
  "index": 43,
  "answer": "Yes",
  "prediction_direct": "Yes",
  "prediction_edited_direct": "No",
  "prediction_edited_kept_direct": "Yes",
  "prediction_edited_kept_normal_direct": "Yes"
}
```

Reasoning records use the `prediction_cot*` family of keys instead. The four
prediction fields correspond to:

| Field                                  | Image used                                                          |
|----------------------------------------|---------------------------------------------------------------------|
| `prediction_*`                         | original CXR                                                        |
| `prediction_edited_*`                  | RadEdit foreground inpainting (attribute removed inside annotation) |
| `prediction_edited_kept_*`             | RadEdit background inpainting with same prompt                       |
| `prediction_edited_kept_normal_*`      | RadEdit background inpainting with "no abnormality" prompt           |

A record passes the 3-step filter when:
1. `prediction_*` matches `answer`,
2. `prediction_edited_*` flips,
3. both `prediction_edited_kept_*` keep the original answer.

## Joining to source data

`(dataset, index)` is a stable key into the source dataset loader output:

```python
from medfocus.data.imagenome import load_imagenome
from medfocus.data.medground_bench import load_medground, attach_source_metadata

direct = load_medground("data/medground_bench", split="direct")
samples = load_imagenome(data_root="/data/physionet.org/files")
records = attach_source_metadata(
    direct,
    source_loaders={"imagenome": samples},
)
```

## Allowed values

| Field      | Values                                                                          |
|------------|---------------------------------------------------------------------------------|
| `model`    | `qwen2_5_vl_3b`, `qwen2_5_vl_7b`, `gemma3_4b`, `gemma3_12b`, `medgemma_4b`, `medgemma1_5_4b` |
| `dataset`  | `imagenome`, `vindr_cxr`, `padchest_gr`                                          |
| `answer`   | `"Yes"` (the benchmark currently contains only abnormality-positive samples)    |
