#!/usr/bin/env bash
set -euo pipefail

# 任务二完整流程：从数据采集到GRPO训练
# 一键运行所有步骤

export ROLL_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROLL_HOME}"

# 配置参数
NUM_EPISODES="${1:-100}"           # 采集episode数量（默认100，快速测试）
POLICY="${2:-bfs}"                  # 采集策略
MAX_SAMPLES="${3:-}"                # GRPO训练最大样本数（可选）

echo "=========================================="
echo "任务二：Sokoban GRPO完整训练流程"
echo "=========================================="
echo "采集Episodes: ${NUM_EPISODES}"
echo "采集策略: ${POLICY}"
echo "=========================================="
echo ""

# ===========================
# 步骤1：采集轨迹
# ===========================
echo "[步骤 1/5] 采集Sokoban轨迹..."
echo "----------------------------------------"

TRAJECTORY_DIR="artifacts/sokoban/${POLICY}_${NUM_EPISODES}ep"

python data_pipeline/collect_sokoban.py \
  --policy ${POLICY} \
  --num-episodes ${NUM_EPISODES} \
  --dim-room 10 10 \
  --num-boxes 4 \
  --max-steps 20 \
  --search_depth 300 \
  --output-dir "${TRAJECTORY_DIR}"

echo "✓ 轨迹采集完成: ${TRAJECTORY_DIR}"
echo ""

# ===========================
# 步骤2：过滤轨迹
# ===========================
echo "[步骤 2/5] 过滤轨迹数据..."
echo "----------------------------------------"

FILTERED_DIR="artifacts/sokoban/filtered_${NUM_EPISODES}ep"

python data_pipeline/filter_sokoban.py \
  --input-dir "${TRAJECTORY_DIR}" \
  --output-dir "${FILTERED_DIR}" \
  --require-success false \
  --min-total-reward -20 \
  --max-length 80

echo "✓ 轨迹过滤完成: ${FILTERED_DIR}"
echo ""

# ===========================
# 步骤3：准备GRPO数据
# ===========================
echo "[步骤 3/5] 准备GRPO训练数据..."
echo "----------------------------------------"

mkdir -p data/rl

PREPARE_CMD="python scripts/rl/prepare_rl_data.py \
  --input-dir ${FILTERED_DIR} \
  --output-train data/rl/sokoban_train_prompts.jsonl \
  --output-val data/rl/sokoban_val_prompts.jsonl \
  --val-ratio 0.1 \
  --seed 42"

# 如果指定了最大样本数，添加到命令
if [ -n "${MAX_SAMPLES}" ]; then
    PREPARE_CMD="${PREPARE_CMD} --max-samples ${MAX_SAMPLES}"
fi

eval ${PREPARE_CMD}

echo "✓ GRPO数据准备完成"
echo ""

# ===========================
# 步骤4：检查配置
# ===========================
echo "[步骤 4/5] 检查配置文件..."
echo "----------------------------------------"

CONFIG_FILE="configs/rl/sokoban_grpo_llama.yaml"

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "错误: 配置文件不存在 ${CONFIG_FILE}"
    exit 1
fi

echo "配置文件: ${CONFIG_FILE}"
echo ""
echo "关键配置:"
grep -E "pretrain:|rollout_batch_size:|num_return_sequences_in_group:|learning_rate:" "${CONFIG_FILE}" | head -10
echo ""

# ===========================
# 步骤5：运行GRPO训练
# ===========================
echo "[步骤 5/5] 启动GRPO训练..."
echo "----------------------------------------"

# 检查数据文件是否存在
if [ ! -f "data/rl/sokoban_train_prompts.jsonl" ]; then
    echo "错误: 训练数据不存在，请检查前面的步骤"
    exit 1
fi

# 运行训练
bash scripts/rl/run_grpo_sokoban.sh

echo ""
echo "=========================================="
echo "✓ 任务二训练流程完成！"
echo "=========================================="
echo ""
echo "查看结果："
echo "  日志: output/rl_runs/logs/sokoban_grpo_llama32_3b/train.log"
echo "  TensorBoard: tensorboard --logdir output/rl_runs/tensorboard/sokoban_grpo_llama32_3b"
echo "  Checkpoint: output/rl_runs/checkpoints/sokoban_grpo_llama32_3b/"
echo ""
echo "=========================================="
