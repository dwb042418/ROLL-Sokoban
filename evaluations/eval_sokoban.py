#!/usr/bin/env python3
"""
Evaluation script for Sokoban SFT baseline.
Evaluates a fine-tuned model on the Sokoban environment and outputs metrics.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

# Add ROLL to path
ROLL_HOME = Path(__file__).parent.parent
sys.path.insert(0, str(ROLL_HOME))

from transformers import AutoModelForCausalLM, AutoTokenizer
from roll.pipeline.agentic.env.sokoban.env import SokobanEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Sokoban SFT model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--env-config",
        type=str,
        default="configs/eval/sokoban_eval.yaml",
        help="Path to the environment config file",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=200,
        help="Number of episodes to evaluate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to output JSON file with metrics",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="qwen2_5",
        help="Chat template to use",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_hardware_info() -> Dict[str, str]:
    """Get hardware information for reproducibility."""
    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0) if info["gpu_count"] > 0 else "N/A"
        info["gpu_memory_total_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
        ) if info["gpu_count"] > 0 else "N/A"

    return info


def format_prompt(observation: str, env_instruction: str) -> str:
    """Format the observation and instruction into a prompt."""
    return f"{env_instruction}\n\nCurrent state:\n{observation}\n\nWhat's your next action?"


def extract_action_from_response(response: str) -> str:
    """Extract the action from the model's response.

    The model should respond with <answer>Up/Down/Left/Right</answer>
    We extract the full response as the action for the environment to parse.
    """
    return response


def evaluate_episode(
    model,
    tokenizer,
    env: SokobanEnv,
    chat_template_func,
    seed: int,
    max_new_tokens: int = 50,
) -> Dict[str, Any]:
    """Evaluate a single episode.

    Returns:
        Dictionary containing episode metrics:
        - success: bool, whether all boxes reached targets
        - steps: int, number of steps taken
        - total_reward: float, cumulative reward
        - terminated: bool, whether episode ended naturally
        - truncated: bool, whether episode was truncated
    """
    obs, info = env.reset(seed=seed)
    env_instruction = info.get("env_instruction", "")

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        # Format prompt
        user_input = format_prompt(obs, env_instruction)

        # Build conversation
        conversation = [{"role": "user", "content": user_input}]

        # Apply chat template
        prompt = chat_template_func(conversation)

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", padding=False)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                top_p=1.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode response
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        action = extract_action_from_response(response)

        # Step environment
        obs, reward, terminated, truncated, step_info = env.step(action)

        total_reward += reward
        steps += 1

        # Check if success
        if step_info.get("metrics", {}).get("success", False):
            break

    success = step_info.get("metrics", {}).get("success", False)

    return {
        "success": success,
        "steps": steps,
        "total_reward": total_reward,
        "terminated": terminated,
        "truncated": truncated,
    }


def main():
    args = parse_args()

    # Load configuration
    if os.path.exists(args.env_config):
        config = load_config(args.env_config)
        env_config = config.get("env", {})
        inference_config = config.get("inference", {})
    else:
        print(f"Warning: Config file {args.env_config} not found, using defaults")
        env_config = {}
        inference_config = {}

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    print(f"Loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model from {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()

    # Create chat template function
    def chat_template_func(conversation):
        """Apply chat template for Qwen2.5"""
        return tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

    # Create environment
    print("Creating Sokoban environment...")
    env = SokobanEnv(
        dim_room=tuple(env_config.get("dim_room", [10, 10])),
        num_boxes=env_config.get("num_boxes", 4),
        max_steps=env_config.get("max_steps", 20),
        search_depth=env_config.get("search_depth", 300),
        render_mode=env_config.get("render_mode", "text"),
    )

    # Run evaluation
    print(f"\nEvaluating {args.num_episodes} episodes...")
    print(f"Random seed: {args.seed}")
    print("=" * 60)

    all_results = []
    success_count = 0
    total_steps = 0
    total_rewards = 0.0

    failure_modes = {
        "max_steps_reached": 0,
        "invalid_actions": 0,
    }

    pbar = tqdm(range(args.num_episodes), desc="Evaluating")

    for episode_idx in pbar:
        # Use seed + episode_idx for reproducibility
        episode_seed = args.seed + episode_idx

        try:
            result = evaluate_episode(
                model=model,
                tokenizer=tokenizer,
                env=env,
                chat_template_func=chat_template_func,
                seed=episode_seed,
                max_new_tokens=inference_config.get("max_new_tokens", 50),
            )

            all_results.append(result)

            # Update statistics
            if result["success"]:
                success_count += 1
            total_steps += result["steps"]
            total_rewards += result["total_reward"]

            if result["truncated"] and not result["terminated"]:
                failure_modes["max_steps_reached"] += 1

            # Update progress bar
            current_success_rate = success_count / (episode_idx + 1) * 100
            current_avg_steps = total_steps / (episode_idx + 1)

            pbar.set_postfix({
                "Success Rate": f"{current_success_rate:.1f}%",
                "Avg Steps": f"{current_avg_steps:.1f}",
            })

        except Exception as e:
            print(f"\nError in episode {episode_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Calculate final metrics
    success_rate = success_count / args.num_episodes * 100
    avg_steps = total_steps / args.num_episodes
    avg_reward = total_rewards / args.num_episodes

    print("\n" + "=" * 60)
    print("Evaluation Results:")
    print(f"  Success Rate: {success_rate:.2f}%")
    print(f"  Avg Steps: {avg_steps:.2f}")
    print(f"  Avg Reward: {avg_reward:.4f}")
    print(f"  Total Episodes: {args.num_episodes}")
    print(f"  Successful Episodes: {success_count}")
    print("=" * 60)

    # Prepare output
    output = {
        "evaluation": {
            "success_rate": round(success_rate, 2),
            "avg_steps": round(avg_steps, 2),
            "avg_reward": round(avg_reward, 4),
            "total_episodes": args.num_episodes,
            "successful_episodes": success_count,
        },
        "failure_modes": failure_modes,
        "config": {
            "model_path": args.model,
            "env_config": args.env_config,
            "num_episodes": args.num_episodes,
            "seed": args.seed,
            "template": args.template,
        },
        "hardware": get_hardware_info(),
        "timestamp": datetime.now().isoformat(),
    }

    # Save to JSON
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {output_path}")

    # Close environment
    env.close()


if __name__ == "__main__":
    main()
