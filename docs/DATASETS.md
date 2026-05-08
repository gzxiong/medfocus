# Datasets

MedGround-Bench is built from three publicly available CXR datasets. The
release ships only the filtered VQA samples; raw images must be downloaded
from the original sources under their respective licenses.

## ImaGenome (Chest ImaGenome)

- Source: PhysioNet, https://physionet.org/content/chest-imagenome/
- Underlying images: MIMIC-CXR-JPG (https://physionet.org/content/mimic-cxr-jpg/2.1.0/).
- Access: PhysioNet credentialing required.
- Layout under `$MEDFOCUS_DATA_ROOT`:
  ```
  chest-imagenome/1.0.0/gold_dataset/gold_object_attribute_with_coordinates.txt
  mimic-cxr-jpg/2.1.0/files/p<2>/p<patient>/s<study>/<image>.jpg
  ```

## VinDR-CXR

- Source: PhysioNet, https://physionet.org/content/vindr-cxr/1.0.0/
- Access: PhysioNet credentialing.
- Layout:
  ```
  vindr-cxr/1.0.0/annotations/annotations_test.csv
  vindr-cxr/1.0.0/test/<image_id>.dicom
  ```

## PadChest-GR

- Source: Kaggle (license requires acceptance from original data providers).
- Place the unzipped images under `<root>/PadChest_GR/` and the accompanying
  `filtered_studies.json` directly in `<root>/`.

## Setting up

Set `MEDFOCUS_DATA_ROOT` to the parent directory containing all three trees:

```bash
export MEDFOCUS_DATA_ROOT=/data/cxr
```

`configs/datasets.yaml` resolves all paths against this variable.

## Released filtered JSON

After all three datasets are in place, the package can already run on the
released benchmark in `data/medground_bench/`. The 4 prediction fields per
record (the four LVLM forwards used by the 3-step filter) are sufficient to
reproduce every attribution baseline; the rebuild scripts under
`scripts/run_radedit.py` / `scripts/run_three_step_filter.py` are only needed
to extend the benchmark.
