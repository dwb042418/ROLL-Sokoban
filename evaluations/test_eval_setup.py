#!/usr/bin/env python3
"""
Simple test to validate the evaluation setup without loading the full model.
Tests environment creation and prompt formatting.
"""

import sys
from pathlib import Path

# Add ROLL to path
ROLL_HOME = Path(__file__).parent.parent
sys.path.insert(0, str(ROLL_HOME))

from roll.pipeline.agentic.env.sokoban.env import SokobanEnv


def test_environment():
    """Test that the Sokoban environment can be created and reset."""
    print("Testing Sokoban environment...")

    env = SokobanEnv(
        dim_room=(10, 10),
        num_boxes=4,
        max_steps=20,
        search_depth=300,
        render_mode="text",
    )

    # Test reset
    obs, info = env.reset(seed=2025)
    env_instruction = info.get("env_instruction", "")

    print(f"✓ Environment created successfully")
    print(f"✓ Environment reset successfully")
    print(f"\nEnvironment Instruction:\n{env_instruction}")
    print(f"\nSample Observation (first 200 chars):\n{str(obs)[:200]}...")

    # Test step with valid action
    action_text = "<answer>Right</answer>"
    obs, reward, terminated, truncated, step_info = env.step(action_text)

    print(f"\n✓ Action executed successfully")
    print(f"  Reward: {reward}")
    print(f"  Terminated: {terminated}")
    print(f"  Truncated: {truncated}")
    print(f"  Metrics: {step_info.get('metrics', {})}")

    env.close()
    print("\n✓ All environment tests passed!")


def test_prompt_formatting():
    """Test prompt formatting logic."""
    print("\n" + "="*60)
    print("Testing prompt formatting...")

    from transformers import AutoTokenizer

    # Create a mock observation
    mock_obs = """##########
#P       #
#   X    #
#   O    #
##########"""

    mock_instruction = "You are solving the Sokoban puzzle."

    # Format prompt
    user_input = f"{mock_instruction}\n\nCurrent state:\n{mock_obs}\n\nWhat's your next action?"

    print(f"✓ Prompt formatted successfully")
    print(f"\nSample prompt (first 300 chars):\n{user_input[:300]}...")

    print("\n✓ Prompt formatting test passed!")


if __name__ == "__main__":
    print("="*60)
    print("Evaluation Setup Validation")
    print("="*60)

    try:
        test_environment()
        test_prompt_formatting()
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60)
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
