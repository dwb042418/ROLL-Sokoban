#!/usr/bin/env bash
set -euo pipefail

EXP_NAME="${1:-sokoban_sft_baseline}"
CONFIG_NAME="${2:-qwen3_sokoban}"
CONFIG_PATH="${3:-configs/sft}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export ROLL_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROLL_HOME}:${PYTHONPATH:-}"

# Disable DeepSpeed CUDA ops compilation
export DS_BUILD_OPS=0
export DS_BUILD_SPARSE_ATTN=0
export DS_BUILD_TRANSFORMER_INFERENCE=0

mkdir -p "${ROLL_HOME}/output/logs/${EXP_NAME}"
mkdir -p "${ROLL_HOME}/output/checkpoints/${EXP_NAME}"

python examples/start_sft_pipeline.py \
  --config_path "${CONFIG_PATH}" \
  --config_name "${CONFIG_NAME}"