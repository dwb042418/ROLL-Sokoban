#!/usr/bin/env python3
"""
过滤 Sokoban 轨迹并输出统计报告。

示例：
    python data_pipeline/filter_sokoban.py \
        --input-dir artifacts/sokoban/raw \
        --output-dir artifacts/sokoban/filtered \
        --min-total-reward -10 \
        --require-success true \
        --max-length 80 \
        --max-repeat-action-ratio 0.95
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger(__name__)


def str2bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {"true", "t", "yes", "y", "1"}:
        return True
    if value in {"false", "f", "no", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"无法解析布尔值: {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="过滤 Sokoban 轨迹并生成统计报告"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="原始 episode JSON 文件所在目录",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="过滤后 episode JSON 输出目录（会自动创建）",
    )
    parser.add_argument(
        "--min-total-reward",
        type=float,
        default=None,
        help="累计奖励阈值，低于该值的样本会被过滤",
    )
    parser.add_argument(
        "--max-total-reward",
        type=float,
        default=None,
        help="累计奖励上限（可选），超过该值的样本会被过滤",
    )
    parser.add_argument(
        "--require-success",
        type=str2bool,
        default=False,
        help="是否要求 success == True",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="样本最大长度限制（步数），超过则过滤",
    )
    parser.add_argument(
        "--max-repeat-action-ratio",
        type=float,
        default=None,
        help=(
            "连续重复动作比例上限（0~1）。"
            "例如 0.9 表示若超过 90% 的相邻动作相同，则过滤。"
        ),
    )
    parser.add_argument(
        "--report-filename",
        type=str,
        default="filter_report.json",
        help="统计报告文件名",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅输出报告，不落盘过滤后的 episode",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="日志等级",
    )
    return parser.parse_args()


def load_episode(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("读取失败 %s: %s", path, exc)
        return None


def compute_repeat_action_ratio(steps: List[Dict[str, Any]]) -> Optional[float]:
    if not steps or len(steps) < 2:
        return 0.0
    actions = [step.get("action") for step in steps]
    total_pairs = len(actions) - 1
    repeats = sum(
        1 for prev, curr in zip(actions[:-1], actions[1:]) if prev == curr
    )
    if total_pairs <= 0:
        return 0.0
    return repeats / total_pairs


def evaluate_episode(
    episode: Dict[str, Any],
    *,
    min_total_reward: Optional[float],
    max_total_reward: Optional[float],
    require_success: bool,
    max_length: Optional[int],
    max_repeat_action_ratio: Optional[float],
) -> List[str]:
    reasons: List[str] = []

    success = bool(episode.get("success", False))
    total_reward = episode.get("total_reward")
    length = episode.get("length") or len(episode.get("steps", []))

    if require_success and not success:
        reasons.append("require_success")

    if min_total_reward is not None:
        if total_reward is None or total_reward < min_total_reward:
            reasons.append("min_total_reward")

    if max_total_reward is not None:
        if total_reward is None or total_reward > max_total_reward:
            reasons.append("max_total_reward")

    if max_length is not None:
        if length is None or length > max_length:
            reasons.append("max_length")

    if max_repeat_action_ratio is not None:
        steps = episode.get("steps") or []
        ratio = compute_repeat_action_ratio(steps)
        if ratio is None or ratio > max_repeat_action_ratio:
            reasons.append("max_repeat_action_ratio")

    return reasons


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_files = sorted(input_dir.glob("episode_*.json"))
    if not episode_files:
        LOGGER.warning("输入目录无匹配 episode_*.json 文件: %s", input_dir)
        return

    LOGGER.info("发现 %d 个候选样本", len(episode_files))

    rejected_counter: Counter[str] = Counter()
    retained_rewards: List[float] = []
    retained_lengths: List[int] = []
    retained_success = 0

    total_processed = 0
    total_retained = 0

    for episode_path in episode_files:
        data = load_episode(episode_path)
        total_processed += 1

        if data is None:
            rejected_counter["load_error"] += 1
            continue

        reasons = evaluate_episode(
            data,
            min_total_reward=args.min_total_reward,
            max_total_reward=args.max_total_reward,
            require_success=args.require_success,
            max_length=args.max_length,
            max_repeat_action_ratio=args.max_repeat_action_ratio,
        )

        if reasons:
            rejected_counter.update(reasons)
            continue

        total_retained += 1
        success = bool(data.get("success", False))
        if success:
            retained_success += 1

        total_reward = data.get("total_reward")
        if isinstance(total_reward, (int, float)):
            retained_rewards.append(float(total_reward))
        length = data.get("length") or len(data.get("steps", []))
        if isinstance(length, int):
            retained_lengths.append(length)

        if not args.dry_run:
            target_path = output_dir / episode_path.name
            with target_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    pass_rate = (
        total_retained / total_processed if total_processed else 0.0
    )
    retained_success_rate = (
        retained_success / total_retained if total_retained else 0.0
    )

    def summarize(values: List[float]) -> Dict[str, Optional[float]]:
        if not values:
            return {"min": None, "max": None, "mean": None, "median": None}
        return {
            "min": min(values),
            "max": max(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
        }

    report: Dict[str, Any] = {
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "filters": {
            "min_total_reward": args.min_total_reward,
            "max_total_reward": args.max_total_reward,
            "require_success": args.require_success,
            "max_length": args.max_length,
            "max_repeat_action_ratio": args.max_repeat_action_ratio,
        },
        "counters": {
            "total_processed": total_processed,
            "total_retained": total_retained,
            "total_rejected": total_processed - total_retained,
            "pass_rate": pass_rate,
            "retained_success_count": retained_success,
            "retained_success_rate": retained_success_rate,
        },
        "rejection_reasons": dict(rejected_counter.most_common()),
        "stats": {
            "retained_total_reward": summarize(retained_rewards),
            "retained_length": summarize([float(x) for x in retained_lengths]),
        },
        "dry_run": args.dry_run,
    }

    report_path = output_dir / args.report_filename
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    LOGGER.info(
        "过滤完成：保留 %d / %d (pass_rate=%.2f%%)，报告已写入 %s",
        total_retained,
        total_processed,
        pass_rate * 100,
        report_path,
    )


if __name__ == "__main__":
    main()