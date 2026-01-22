#!/usr/bin/env python3
"""
随机采集 Sokoban 轨迹，并保存为 JSON。

示例：
    python data_pipeline/collect_sokoban.py \
        --num-episodes 50 \
        --output-dir artifacts/sokoban/random_trajs
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import numpy as np
from tqdm import trange

# 确保可以导入 roll 包（脚本直接运行时需要）
import sys
from pathlib import Path as _Path

REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from roll.pipeline.agentic.env.sokoban import SokobanEnv


@dataclass
class StepRecord:
    observation: str
    action: str
    reward: float
    terminated: bool
    truncated: bool
    info: dict


@dataclass
class EpisodeRecord:
    steps: List[StepRecord]
    success: bool
    total_reward: float
    length: int
    seed: int


def format_action(action_name: str) -> str:
    """将动作名包装成环境要求的 <answer>...</answer> 格式。"""
    return f"<answer>{action_name}</answer>"


def collect(
    output_dir: str,
    num_episodes: int,
    max_steps: int,
    seed_offset: int,
    dim_room: tuple[int, int],
    num_boxes: int,
    render_mode: str,
    search_depth: int,
) -> None:
    env = SokobanEnv(
        dim_room=dim_room,
        num_boxes=num_boxes,
        max_steps=max_steps,
        render_mode=render_mode,
        search_depth=search_depth,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "success": 0}

    try:
        for ep_idx in trange(num_episodes, desc="Collecting"):
            seed = seed_offset + ep_idx
            obs, info = env.reset(seed=seed)

            episode = EpisodeRecord(
                steps=[],
                success=False,
                total_reward=0.0,
                length=0,
                seed=seed,
            )

            for _ in range(max_steps):
                action_name = env.sample_random_action()
                action_text = format_action(action_name)

                next_obs, reward, terminated, truncated, step_info = env.step(action_text)

                episode.steps.append(
                    StepRecord(
                        observation=obs,
                        action=action_name,
                        reward=float(reward),
                        terminated=terminated,
                        truncated=truncated,
                        info=step_info,
                    )
                )

                episode.total_reward += reward
                episode.length += 1
                obs = next_obs

                if terminated or truncated:
                    episode.success = bool(step_info["metrics"]["success"])
                    break

            stats["total"] += 1
            stats["success"] += int(episode.success)

            with open(out_dir / f"episode_{ep_idx:05d}.json", "w", encoding="utf-8") as f:
                json.dump(asdict(episode), f, ensure_ascii=False, indent=2)

        summary = {
            "num_episodes": stats["total"],
            "success": stats["success"],
            "success_rate": stats["success"] / max(stats["total"], 1),
            "dim_room": dim_room,
            "num_boxes": num_boxes,
            "max_steps": max_steps,
            "seed_offset": seed_offset,
        }

        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print("Done:", summary)
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="随机采集 Sokoban 轨迹")
    parser.add_argument(
    "--policy",
    type=str,
    default="random",
    choices=("random", "bfs"),
    help="采集策略：随机或 BFS 规划。",
    )
    parser.add_argument(
        "--policy-epsilon",
        type=float,
        default=0.05,
        help="对于 BFS 策略，按该概率跳过规划直接随机。",
    )
    parser.add_argument("--bfs-max-depth", type=int, default=80)
    parser.add_argument("--bfs-max-nodes", type=int, default=20000)
    parser.add_argument("--output-dir", type=str, default="artifacts/sokoban/random_trajs")
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--dim-room", type=int, nargs=2, default=(6, 6), metavar=("H", "W"))
    parser.add_argument("--num-boxes", type=int, default=1)
    parser.add_argument(
        "--render-mode",
        type=str,
        default="text",
        choices=("text", "rgb_array"),
        help="随机策略采集通常用 text，必要时可以换 rgb_array",
    )
    parser.add_argument("--search-depth", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collect(
        output_dir=args.output_dir,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        seed_offset=args.seed_offset,
        dim_room=tuple(args.dim_room),
        num_boxes=args.num_boxes,
        render_mode=args.render_mode,
        search_depth=args.search_depth,
    )


if __name__ == "__main__":
    main()