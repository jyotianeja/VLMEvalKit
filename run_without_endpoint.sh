#!/usr/bin/env bash
set -euo pipefail

# MODEL="Qwen3-VL-8B-Instruct-API"
MODEL="Qwen3-VL-8B-Instruct-API"
API_NPROC=32

BENCHMARKS=(
  # MathVista_MINI
  # MathVerse_MINI
  # MathVision_MINI
  # AI2D_TEST
  # BLINK
  # ChartQA_TEST
  # DocVQA_VAL
  # HallusionBench
  # MMMU_DEV_VAL
  # MMStar
  # # MUIRBench
  # OCRBench
  # # OlympiadBench
  # ScreenSpot_v2_Desktop
  # ScreenSpot_v2_Mobile
  # ScreenSpot_v2_Web
  # # ScreenSpot_Pro
  # WeMath
  # WildVision
  ZEROBench_sub
)

for DATA in "${BENCHMARKS[@]}"; do
  echo "=============================================="
  echo "Running benchmark: ${DATA}"
  echo "Model: ${MODEL}"
  echo "=============================================="

  python run.py \
    --data "${DATA}" \
    --model "${MODEL}" \
    --api-nproc "${API_NPROC}" \
    --judge "gpt-4o" \
    # --reuse \
    # --mode eval --verbose

  echo "Finished benchmark: ${DATA}" 
  echo
done

echo "✅ All benchmarks completed successfully."