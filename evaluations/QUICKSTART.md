# Quick Reference: Sokoban Evaluation

## 🚀 Quick Start

```bash
# 1. Run validation test
python evaluations/test_eval_setup.py

# 2. Run evaluation (default 200 episodes)
./scripts/run_eval.sh

# 3. View results
cat output/evals/sokoban_sft_baseline_metrics.json
```

## 📊 Command-Line Options

```bash
python evaluations/eval_sokoban.py \
  --model <path>              # Model checkpoint path
  --env-config <path>         # Config file (default: configs/eval/sokoban_eval.yaml)
  --num-episodes <int>        # Number of episodes (default: 200)
  --seed <int>                # Random seed (default: 2025)
  --output-json <path>        # Output file (default: auto-generated)
  --template <str>            # Chat template (default: qwen2_5)
```

## 📁 File Locations

```
ROLL/
├── evaluations/
│   ├── eval_sokoban.py              # Main script
│   ├── test_eval_setup.py           # Validation test
│   ├── README.md                    # Full documentation
│   └── IMPLEMENTATION_SUMMARY.md    # Implementation details
├── configs/eval/
│   └── sokoban_eval.yaml           # Configuration
├── scripts/
│   └── run_eval.sh                 # Wrapper script
└── output/evals/                   # Results directory
```

## 📏 Metrics

| Metric | Description |
|--------|-------------|
| Success Rate | % episodes with all boxes on targets |
| Avg Steps | Average steps per episode |
| Avg Reward | Average cumulative reward |
| Failure Modes | Max steps reached, invalid actions |

## 🔧 Configuration

Edit `configs/eval/sokoban_eval.yaml`:

```yaml
env:
  dim_room: [10, 10]    # Room size
  num_boxes: 4          # Number of boxes
  max_steps: 20         # Steps per episode

inference:
  max_new_tokens: 50    # Max generation tokens
  temperature: 1.0      # Sampling temperature
```

## 🎯 Common Tasks

### Test Different Episode Counts
```bash
./scripts/run_eval.sh /path/to/model 50 output/evals/test_50.json
```

### Evaluate Multiple Seeds
```bash
for seed in 2025 2026 2027; do
  python evaluations/eval_sokoban.py --model /path/to/model \
    --seed $seed --output-json output/evals/seed_$seed.json
done
```

### Compare Base vs Fine-tuned
```bash
# Base model
python evaluations/eval_sokoban.py --model Qwen/Qwen2-1.5B-Instruct \
  --output-json output/evals/base_model.json

# Fine-tuned model
python evaluations/eval_sokoban.py --model output/checkpoints/sft_baseline \
  --output-json output/evals/finetuned.json
```

## ✅ Validation Checklist

- [ ] Run `python evaluations/test_eval_setup.py` - should pass all tests
- [ ] Check model path exists (use cached path if no internet)
- [ ] Verify GPU available: `nvidia-smi`
- [ ] Check output directory: `mkdir -p output/evals`

## 🐛 Troubleshooting

**Model not found?**
```bash
# Find cached model
find ~/.cache/modelscope -name "Qwen2-1.5B-Instruct" -type d

# Use full path
python evaluations/eval_sokoban.py --model /full/path/to/model
```

**CUDA OOM?**
- Reduce `max_new_tokens` in config
- Use smaller model
- Set `CUDA_VISIBLE_DEVICES=0`

**Environment errors?**
```bash
pip install gym-sokoban
```

## 📝 Example Output

```json
{
  "evaluation": {
    "success_rate": 45.5,
    "avg_steps": 12.3,
    "total_episodes": 200,
    "successful_episodes": 91
  },
  "hardware": {
    "gpu_name": "NVIDIA A100",
    "gpu_memory_total_gb": 79.25
  }
}
```

## 🔗 Related Files

- Training config: [configs/sft/qwen3_sokoban.yaml](../configs/sft/qwen3_sokoban.yaml)
- Training script: [scripts/run_sft.sh](../scripts/run_sft.sh)
- Training logs: `output/logs/sokoban_sft_baseline/`
