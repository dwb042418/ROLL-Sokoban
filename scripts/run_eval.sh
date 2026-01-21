#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-/home/batchcom/.cache/modelscope/hub/models/Qwen/Qwen2-1.5B-Instruct}"
NUM_EPISODES="${2:-200}"
OUTPUT_JSON="${3:-output/evals/sokoban_sft_baseline_metrics.json}"

export ROLL_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROLL_HOME}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "======================================"
echo "Sokoban SFT Evaluation"
echo "======================================"
echo "Model: ${MODEL_PATH}"
echo "Episodes: ${NUM_EPISODES}"
echo "Output: ${OUTPUT_JSON}"
echo "======================================"
echo ""

python "${ROLL_HOME}/evaluations/eval_sokoban.py" \
  --model "${MODEL_PATH}" \
  --env-config "${ROLL_HOME}/configs/eval/sokoban_eval.yaml" \
  --num-episodes "${NUM_EPISODES}" \
  --seed 2025 \
  --output-json "${OUTPUT_JSON}" \
  --template qwen2_5
