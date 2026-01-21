#!/usr/bin/env python3
"""
简单的 Sokoban 环境 smoke test。

用法示例：
    python scripts/test_sokoban_env.py --seed 0 --steps 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 如果脚本位置较深，可在此手动加入仓库根目录到 sys.path
REPO_ROOT_HINT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_HINT))

try:
    from roll.pipeline.agentic.env.sokoban import SokobanEnv
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "无法导入 SokobanEnv，请确认：\n"
        "1) 当前目录在 ROLL 仓库根目录；或\n"
        "2) 已执行 `export PYTHONPATH=/path/to/ROLL:$PYTHONPATH`。\n"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sokoban 环境 smoke test")
    parser.add_argument("--seed", type=int, default=0, help="环境 reset 的随机种子")
    parser.add_argument("--steps", type=int, default=5, help="最多执行的 step 数")
    parser.add_argument(
        "--dim-room",
        type=int,
        nargs=2,
        default=(6, 6),
        metavar=("H", "W"),
        help="棋盘尺寸 (高度, 宽度)",
    )
    parser.add_argument("--num-boxes", type=int, default=1, help="箱子数量")
    parser.add_argument("--max-steps", type=int, default=50, help="环境最大步数")
    parser.add_argument(
        "--render-mode",
        type=str,
        default="text",
        choices=("text", "rgb_array"),
        help="渲染模式（默认为 text）",
    )
    parser.add_argument(
        "--search-depth", type=int, default=10, help="房间生成时的搜索深度"
    )
    return parser.parse_args()


def format_action(action_name: str) -> str:
    """根据环境要求封装动作文本。"""
    return f"<answer>{action_name}</answer>"


def main() -> None:
    args = parse_args()

    env = SokobanEnv(
        dim_room=tuple(args.dim_room),
        num_boxes=args.num_boxes,
        max_steps=args.max_steps,
        render_mode=args.render_mode,
        search_depth=args.search_depth,
    )

    try:
        obs, info = env.reset(seed=args.seed)
        print("=== Reset 完成 ===")
        print("初始观测:")
        print(obs)
        print("info:", info)
        print()

        for step_idx in range(args.steps):
            action_name = env.sample_random_action()
            action_text = format_action(action_name)
            obs, reward, terminated, truncated, info = env.step(action_text)

            print(
                f"[step {step_idx:02d}] action={action_name!s:>5} "
                f"reward={reward:+.2f} terminated={terminated} truncated={truncated}"
            )
            print("下一观测:")
            print(obs)
            print("info:", info)
            print("-" * 40)

            if terminated or truncated:
                print("Episode 结束。")
                break

    finally:
        env.close()


if __name__ == "__main__":
    main()