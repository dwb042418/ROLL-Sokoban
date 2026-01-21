#!/usr/bin/env bash
set -euo pipefail

# GRPO训练脚本 - Sokoban with Llama-3.2-3B
# 独立于任务一的SFT训练，使用不同的输出目录

# ================================
# 设置 HuggingFace 缓存目录和 Ray 临时目录
# ================================
export HF_HOME="$(pwd)/hf_cache"
export RAY_TEMP_DIR="$(pwd)/ray_tmp"

mkdir -p "$HF_HOME"
mkdir -p "$RAY_TEMP_DIR"

# ================================
# 实验参数
# ================================
EXP_NAME="${1:-sokoban_grpo_llama32_3b}"
CONFIG_NAME="${2:-sokoban_grpo_llama}"
CONFIG_PATH="${3:-configs/rl}"
export HF_TOKEN={YOUR_HF_TOKEN}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export ROLL_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROLL_HOME}:${PYTHONPATH:-}"

# 禁用DeepSpeed CUDA ops编译
export DS_BUILD_OPS=0
export DS_BUILD_SPARSE_ATTN=0
export DS_BUILD_TRANSFORMER_INFERENCE=0

# 使用 HuggingFace 下载模型（而不是 ModelScope）
export MODEL_DOWNLOAD_TYPE="HUGGINGFACE_HUB"
# 使用 HuggingFace 镜像（如果网络受限）
export HF_ENDPOINT="https://hf-mirror.com"
# 设置 HuggingFace token（用于访问 gated 模型）
# 请将 YOUR_HF_TOKEN 替换为你的实际 token
export HF_TOKEN="${HF_TOKEN:-YOUR_HF_TOKEN}"

# ================================
# 创建输出目录
# ================================
mkdir -p "${ROLL_HOME}/output/rl_runs/logs/${EXP_NAME}"
mkdir -p "${ROLL_HOME}/output/rl_runs/checkpoints/${EXP_NAME}"
mkdir -p "${ROLL_HOME}/output/rl_runs/tensorboard/${EXP_NAME}"

echo "=========================================="
echo "Sokoban GRPO Training"
echo "=========================================="
echo "实验名称: ${EXP_NAME}"
echo "配置文件: ${CONFIG_PATH}/${CONFIG_NAME}.yaml"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "输出目录: ${ROLL_HOME}/output/rl_runs/"
echo "HF缓存目录: ${HF_HOME}"
echo "Ray临时目录: ${RAY_TEMP_DIR}"
echo "=========================================="
echo ""

# 检查数据文件是否存在
TRAIN_DATA="data/rl/sokoban_train_prompts.jsonl"
VAL_DATA="data/rl/sokoban_val_prompts.jsonl"

if [ ! -f "${TRAIN_DATA}" ]; then
    echo "警告: 训练数据不存在 ${TRAIN_DATA}"
    echo "请先运行: python scripts/rl/prepare_rl_data.py"
    echo ""
fi

# ================================
# 启动训练
# ================================
MODEL_DOWNLOAD_TYPE="HUGGINGFACE_HUB" \
HF_HUB_ENABLE_HF_TRANSFER=1 \
python "${ROLL_HOME}/examples/start_rlvr_pipeline.py" \
  --config_path "../${CONFIG_PATH}" \
  --config_name "${CONFIG_NAME}" \
  2>&1 | tee "${ROLL_HOME}/output/rl_runs/logs/${EXP_NAME}/train.log"

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo "日志: ${ROLL_HOME}/output/rl_runs/logs/${EXP_NAME}/train.log"
echo "Checkpoint: ${ROLL_HOME}/output/rl_runs/checkpoints/${EXP_NAME}/"
echo "TensorBoard: ${ROLL_HOME}/output/rl_runs/tensorboard/${EXP_NAME}/"
echo ""
echo "查看TensorBoard:"
echo "  tensorboard --logdir ${ROLL_HOME}/output/rl_runs/tensorboard/${EXP_NAME}"
echo "=========================================="