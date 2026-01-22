# Sokoban SFT Evaluation Pipeline

This directory contains the evaluation pipeline for the Sokoban SFT (Supervised Fine-Tuning) baseline.

## Directory Structure

```
ROLL/
├── evaluations/
│   ├── eval_sokoban.py          # Main evaluation script
│   └── README.md                 # This file
├── configs/eval/
│   └── sokoban_eval.yaml        # Evaluation configuration
├── scripts/
│   └── run_eval.sh              # Convenience script for running evaluation
└── output/evals/                 # Directory for evaluation results
```

## Quick Start

### Option 1: Using the wrapper script (Recommended)

```bash
# Run evaluation with default settings
./scripts/run_eval.sh

# Run evaluation with custom model and episodes
./scripts/run_eval.sh /path/to/model 100 output/evals/custom_results.json
```

### Option 2: Using Python directly

```bash
python evaluations/eval_sokoban.py \
  --model /home/batchcom/.cache/modelscope/hub/models/Qwen/Qwen2-1.5B-Instruct \
  --env-config configs/eval/sokoban_eval.yaml \
  --num-episodes 200 \
  --seed 2025 \
  --output-json output/evals/sokoban_sft_baseline_metrics.json \
  --template qwen2_5
```

## Configuration

The evaluation is configured via `configs/eval/sokoban_eval.yaml`:

```yaml
env:
  env_id: sokoban
  dim_room: [10, 10]      # Room dimensions
  num_boxes: 4            # Number of boxes
  max_steps: 20           # Maximum steps per episode
  search_depth: 300       # Depth for room generation
  render_mode: text

inference:
  batch_size: 1
  max_new_tokens: 50
  temperature: 1.0
  top_p: 1.0
  do_sample: false
```

## Output Format

Results are saved as JSON with the following structure:

```json
{
  "evaluation": {
    "success_rate": 45.5,
    "avg_steps": 12.3,
    "avg_reward": 0.234,
    "total_episodes": 200,
    "successful_episodes": 91
  },
  "failure_modes": {
    "max_steps_reached": 109,
    "invalid_actions": 0
  },
  "config": {
    "model_path": "...",
    "env_config": "...",
    "num_episodes": 200,
    "seed": 2025,
    "template": "qwen2_5"
  },
  "hardware": {
    "torch_version": "...",
    "cuda_available": true,
    "cuda_version": "...",
    "gpu_count": 1,
    "gpu_name": "...",
    "gpu_memory_total_gb": 79.25
  },
  "timestamp": "2026-01-18T..."
}
```

## Metrics

### Primary Metrics

1. **Success Rate**: Percentage of episodes where all boxes reached their targets
2. **Avg Steps**: Average number of steps taken per episode

### Secondary Metrics

3. **Avg Reward**: Average cumulative reward per episode
4. **Failure Modes**:
   - `max_steps_reached`: Episodes that hit the step limit
   - `invalid_actions`: Episodes with invalid action parsing

## Reproducibility

The evaluation pipeline ensures reproducibility through:

- **Fixed Seed**: Default seed is 2025, configurable via `--seed`
- **Deterministic Environment**: Same seed produces identical room layouts
- **Hardware Recording**: GPU information is logged for documentation
- **Timestamp**: Evaluation run time is recorded

## Evaluation Process

For each episode:

1. **Initialize Environment**: Reset Sokoban environment with `seed + episode_idx`
2. **Generate Prompt**: Format observation with environment instructions
3. **Model Inference**: Generate action using the fine-tuned model
4. **Execute Action**: Parse response and execute in environment
5. **Track Metrics**: Record success, steps, rewards
6. **Repeat**: Until episode terminates (success or max steps)

## Troubleshooting

### Model Not Found

If you get "model not found" errors:

```bash
# Find cached model location
find ~/.cache/modelscope -name "Qwen2-1.5B-Instruct" -type d

# Use the full path
python evaluations/eval_sokoban.py --model /full/path/to/model ...
```

### CUDA Out of Memory

If you encounter OOM errors:

1. Reduce `max_new_tokens` in config
2. Use a smaller model
3. Set `CUDA_VISIBLE_DEVICES` to use specific GPU

### Environment Errors

If the Sokoban environment fails to initialize:

```bash
# Check if gym-sokoban is installed
pip install gym-sokoban
```

## Advanced Usage

### Evaluating Different Checkpoints

```bash
# Evaluate base model
python evaluations/eval_sokoban.py --model Qwen/Qwen2-1.5B-Instruct ...

# Evaluate fine-tuned model (once checkpoints are saved)
python evaluations/eval_sokoban.py --model output/checkpoints/sokoban_sft_baseline/checkpoint-2000 ...
```

### Batch Evaluation

```bash
# Evaluate multiple seeds
for seed in 2025 2026 2027; do
  python evaluations/eval_sokoban.py \
    --model /path/to/model \
    --seed $seed \
    --output-json output/evals/seed_${seed}.json
done
```

### Custom Environment Parameters

Edit `configs/eval/sokoban_eval.yaml`:

```yaml
env:
  dim_room: [8, 8]      # Smaller room
  num_boxes: 2          # Fewer boxes
  max_steps: 30         # More steps
```

## Implementation Details

- **Model Loading**: Uses HuggingFace Transformers with ModelScope support
- **Chat Template**: Applies Qwen2.5 chat template for prompt formatting
- **Action Parsing**: Extracts actions from `<answer>Up/Down/Left/Right</answer>` format
- **Progress Tracking**: Real-time progress bar with current metrics
- **Error Handling**: Robust error handling with detailed logging

## Dependencies

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.x
- Gym-Sokoban
- tqdm (for progress bar)
- PyYAML

## Notes

- The evaluation requires a GPU for inference
- Model checkpoints from training should be saved in `output/checkpoints/`
- Currently uses base Qwen2-1.5B-Instruct model as baseline
- Fine-tuned checkpoint evaluation will be added after training completion
