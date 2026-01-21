#!/usr/bin/env bash
set -euo pipefail

# 准备Sokoban GRPO训练数据
# 从采集的轨迹生成prompts

export ROLL_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

INPUT_DIR="${1:-artifacts/sokoban/filtered}"
OUTPUT_TRAIN="${2:-data/rl/sokoban_train_prompts.jsonl}"
OUTPUT_VAL="${3:-data/rl/sokoban_val_prompts.jsonl}"
VAL_RATIO="${4:-0.1}"
MAX_SAMPLES="${5:-}"  # 可选，限制样本数用于快速测试

echo "=========================================="
echo "准备Sokoban GRPO训练数据"
echo "=========================================="
echo "输入目录: ${INPUT_DIR}"
echo "训练输出: ${OUTPUT_TRAIN}"
echo "验证输出: ${OUTPUT_VAL}"
echo "验证比例: ${VAL_RATIO}"
echo "=========================================="
echo ""

# 检查输入目录
if [ ! -d "${INPUT_DIR}" ]; then
    echo "错误: 输入目录不存在 ${INPUT_DIR}"
    echo ""
    echo "请先采集轨迹数据："
    echo "  python data_pipeline/collect_sokoban.py --policy bfs --num-episodes 300 --output-dir artifacts/sokoban/bfs_300ep"
    echo ""
    echo "然后过滤数据："
    echo "  python data_pipeline/filter_sokoban.py --input-dir artifacts/sokoban/bfs_300ep --output-dir artifacts/sokoban/filtered --require-success true"
    exit 1
fi

# 构建命令
CMD="python ${ROLL_HOME}/scripts/rl/prepare_rl_data.py \
  --input-dir ${INPUT_DIR} \
  --output-train ${OUTPUT_TRAIN} \
  --output-val ${OUTPUT_VAL} \
  --val-ratio ${VAL_RATIO} \
  --seed 42"

# 如果指定了最大样本数，添加到命令中
if [ -n "${MAX_SAMPLES}" ]; then
    CMD="${CMD} --max-samples ${MAX_SAMPLES}"
fi

# 运行数据准备
eval ${CMD}

echo ""
echo "=========================================="
echo "数据准备完成！"
echo "=========================================="
echo "下一步：运行GRPO训练"
echo "  bash scripts/rl/run_grpo_sokoban.sh"
echo "=========================================="
