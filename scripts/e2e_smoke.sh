#!/usr/bin/env bash
# End-to-end smoke test: 6 samples, 3 attribution methods, 1 LVLM.

set -euo pipefail

MODEL=${MODEL:-qwen2_5_vl_3b}
LIMIT=${LIMIT:-2}
OUT_DIR=${OUT_DIR:-results/smoke}

mkdir -p "${OUT_DIR}"

echo "== MedFocus =="
python scripts/run_attribution.py --model "${MODEL}" --split direct \
    --method medfocus --limit "${LIMIT}" --out "${OUT_DIR}/medfocus.json"

echo "== gradcam =="
python scripts/run_attribution.py --model "${MODEL}" --split direct \
    --method gradcam --limit "${LIMIT}" --out "${OUT_DIR}/gradcam.json"

echo "== attention_rollout =="
python scripts/run_attribution.py --model "${MODEL}" --split direct \
    --method attention_rollout --limit "${LIMIT}" --out "${OUT_DIR}/attention_rollout.json"

echo "== eval =="
python scripts/eval_attribution.py \
    --inputs "${OUT_DIR}/medfocus.json" "${OUT_DIR}/gradcam.json" "${OUT_DIR}/attention_rollout.json" \
    --out "${OUT_DIR}/main_table.csv"

echo "smoke OK"
