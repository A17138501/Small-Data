#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-openai:gpt-4o-mini}"
SPLIT="splits/test.csv"
OUT_DIR="experiments/zero_shot"

python -m baselines.zero_shot \
  --split "${SPLIT}" \
  --model "${MODEL}" \
  --out_dir "${OUT_DIR}" \
  --max_new_tokens 256 \
  --temperature 0

# 评测
python -m evaluation.eval_calls \
  --gold "${SPLIT}" \
  --pred "experiments/zero_shot/${MODEL//:/_}/pred_test.csv" \
  --run_name "zero_shot_${MODEL//:/_}" \
  --out_dir reports/baselines
