#!/usr/bin/env python3
"""
准备Sokoban GRPO训练数据

功能：
1. 从采集的轨迹中提取初始状态作为prompts
2. 生成prompts JSONL文件用于GRPO训练
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict
import random


def parse_args():
    parser = argparse.ArgumentParser(description="准备Sokoban RL训练数据")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="采集的轨迹目录（artifacts/sokoban/filtered）"
    )
    parser.add_argument(
        "--output-train",
        type=str,
        default="data/rl/sokoban_train_prompts.jsonl",
        help="输出训练prompts文件"
    )
    parser.add_argument(
        "--output-val",
        type=str,
        default="data/rl/sokoban_val_prompts.jsonl",
        help="输出验证prompts文件"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="验证集比例"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最大样本数（用于快速测试）"
    )
    return parser.parse_args()


def load_trajectories(data_dir: str) -> List[Dict]:
    """加载轨迹数据"""
    trajectories = []

    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"错误: 目录不存在 {data_dir}")
        return trajectories

    # 查找所有episode文件
    episode_files = sorted(data_path.glob("episode_*.json"))

    print(f"找到 {len(episode_files)} 个轨迹文件")

    for episode_file in episode_files:
        try:
            with open(episode_file, 'r') as f:
                traj = json.load(f)
                trajectories.append(traj)
        except Exception as e:
            print(f"警告: 无法加载 {episode_file}: {e}")
            continue

    return trajectories


def trajectory_to_prompt(trajectory: Dict, template: str = "llama3_1") -> Dict:
    """
    将轨迹转换为prompt格式

    对于GRPO，我们需要：
    1. 初始状态观测
    2. 环境指令
    3. 任务描述
    """
    # 提取初始观测
    initial_obs = trajectory.get('initial_observation', '')

    # 构建环境指令
    env_instruction = (
        "You are solving the Sokoban puzzle. "
        "You are the player and you need to push all boxes (X) to targets (O). "
        "When you are right next to a box, you can push it by moving in the same direction. "
        "You cannot push a box through a wall, and you cannot pull a box. "
        "Generate a sequence of actions to solve the puzzle.\n\n"
        "Map symbols:\n"
        "# = wall\n"
        "_ = empty floor\n"
        "O = target\n"
        "X = box\n"
        "P = player\n"
        "√ = box on target\n"
        "S = player on target\n\n"
        "Available actions: Up, Down, Left, Right\n"
        "Format: Generate each action as <answer>Action</answer>, for example:\n"
        "<answer>Right</answer>\n"
        "<answer>Down</answer>\n"
        "etc.\n\n"
    )

    # 构建完整prompt
    prompt_content = f"{env_instruction}Current state:\n{initial_obs}\n\nWhat actions should you take to solve this puzzle?"

    # 构建messages格式（Llama 3格式）
    messages = [
        {"role": "user", "content": prompt_content}
    ]

    return {
        "messages": messages,
        "tag": "sokoban",  # 顶层tag，用于ROLL框架识别domain
        "metadata": {
            "episode_id": trajectory.get('episode_id'),
            "seed": trajectory.get('seed'),
            "success": trajectory.get('success', False),
            "total_reward": trajectory.get('total_reward', 0),
            "length": trajectory.get('length', 0),
            "tag": "sokoban"  # 用于reward worker识别
        }
    }


def prepare_prompts(trajectories: List[Dict], seed: int, val_ratio: float = 0.1):
    """准备训练和验证prompts"""
    random.seed(seed)

    # 打乱数据
    random.shuffle(trajectories)

    # 划分训练集和验证集
    val_size = int(len(trajectories) * val_ratio)
    val_trajectories = trajectories[:val_size]
    train_trajectories = trajectories[val_size:]

    print(f"训练集: {len(train_trajectories)} 样本")
    print(f"验证集: {len(val_trajectories)} 样本")

    # 转换为prompts
    train_prompts = [trajectory_to_prompt(traj) for traj in train_trajectories]
    val_prompts = [trajectory_to_prompt(traj) for traj in val_trajectories]

    return train_prompts, val_prompts


def save_prompts(prompts: List[Dict], output_file: str):
    """保存prompts到JSONL文件"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        for prompt in prompts:
            f.write(json.dumps(prompt, ensure_ascii=False) + '\n')

    print(f"✓ 已保存 {len(prompts)} 个prompts到 {output_file}")


def main():
    args = parse_args()

    print("="*60)
    print("Sokoban GRPO训练数据准备")
    print("="*60)
    print(f"输入目录: {args.input_dir}")
    print(f"训练输出: {args.output_train}")
    print(f"验证输出: {args.output_val}")
    print(f"验证集比例: {args.val_ratio}")
    print("="*60)
    print()

    # 1. 加载轨迹
    print("1. 加载轨迹数据...")
    trajectories = load_trajectories(args.input_dir)

    if len(trajectories) == 0:
        print("错误: 没有找到轨迹数据")
        return

    # 限制样本数（用于测试）
    if args.max_samples and args.max_samples < len(trajectories):
        print(f"限制样本数: {args.max_samples}")
        trajectories = trajectories[:args.max_samples]

    print(f"   成功加载 {len(trajectories)} 条轨迹")
    print()

    # 2. 转换为prompts
    print("2. 转换为prompts...")
    train_prompts, val_prompts = prepare_prompts(
        trajectories,
        args.seed,
        args.val_ratio
    )
    print()

    # 3. 保存prompts
    print("3. 保存prompts...")
    save_prompts(train_prompts, args.output_train)
    save_prompts(val_prompts, args.output_val)
    print()

    # 4. 统计信息
    print("="*60)
    print("数据准备完成！")
    print("="*60)
    print(f"训练样本: {len(train_prompts)}")
    print(f"验证样本: {len(val_prompts)}")

    # 显示示例
    if len(train_prompts) > 0:
        print("\n示例prompt:")
        print(json.dumps(train_prompts[0], ensure_ascii=False, indent=2)[:500] + "...")


if __name__ == "__main__":
    main()
