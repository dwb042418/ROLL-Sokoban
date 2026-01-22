#!/usr/bin/env python3
"""
将过滤后的 Sokoban 轨迹转换为 Qwen 指令调优所需的对话格式（JSONL）。

示例：
    python data_pipeline/convert_to_sft_format.py \
        --input-dir artifacts/sokoban/filtered_20260115_221722 \
        --output-jsonl data/sft/sokoban_train.jsonl \
        --template configs/templates/qwen_instruct.json \
        --include-think false \
        --max-steps 80 \
        --val-ratio 0.1 \
        --val-output-jsonl data/sft/sokoban_val.jsonl \
        --stats-output data/sft/sokoban_stats.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
        description="将 Sokoban 轨迹转换为指令式 SFT 数据（JSONL）"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="过滤后的 episode JSON 文件目录",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        required=True,
        help="训练集 JSONL 输出路径",
    )
    parser.add_argument(
        "--template",
        type=str,
        required=True,
        help="模板配置文件 (JSON)",
    )
    parser.add_argument(
        "--include-think",
        type=str2bool,
        default=False,
        help="是否在 assistant 回复中加入 <think> 思维链段落",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="样本最多保留的步数（超出则截断）",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.0,
        help="验证集比例（0~1），>0 时需要同时指定 --val-output-jsonl",
    )
    parser.add_argument(
        "--val-output-jsonl",
        type=str,
        default=None,
        help="验证集 JSONL 输出路径（val_ratio>0 时必填）",
    )
    parser.add_argument(
        "--stats-output",
        type=str,
        default=None,
        help="额外输出统计信息 JSON 文件路径（可选）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（用于划分训练/验证集）",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="日志等级",
    )
    return parser.parse_args()


def safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("JSON 解析失败 %s: %s", path, exc)
        return None


def load_template(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_initial_board(episode: Dict[str, Any]) -> Optional[str]:
    """
    根据常见字段推断初始棋盘文本。
    若无法找到，返回 None。
    """
    candidates = [
        episode.get("initial_observation"),
        episode.get("initial_state"),
        episode.get("initial_board"),
        episode.get("board"),
        episode.get("state"),
    ]

    # 有些数据里 steps 内部包含起始状态
    steps = episode.get("steps") or []
    if steps:
        first_step = steps[0]
        candidates.extend(
            [
                first_step.get("observation_before"),
                first_step.get("observation"),
                first_step.get("state"),
                first_step.get("board_before"),
            ]
        )

    for item in candidates:
        if isinstance(item, str) and item.strip():
            return item.strip()
        if isinstance(item, dict):
            text = item.get("text") or item.get("board")
            if isinstance(text, str) and text.strip():
                return text.strip()
    return None


def resolve_action_name(
    step: Dict[str, Any],
    idx: int,
    template: Dict[str, Any],
) -> Optional[str]:
    """
    优先读取 step['action_name']。
    若无，则尝试根据整数 action 查表。
    """
    action_name = step.get("action_name")
    if isinstance(action_name, str) and action_name.strip():
        return action_name.strip()

    action_value = step.get("action")
    if action_value is None:
        return None

    # 允许字符串形式的 action
    if isinstance(action_value, str) and action_value.strip():
        default_map = template.get("default_action_names", {})
        mapped = default_map.get(action_value)
        if mapped:
            return mapped
        return action_value.strip()

    # 整数映射
    if isinstance(action_value, (int, float)):
        key = str(int(action_value))
        action_map = template.get("action_map") or {}
        if key in action_map:
            return action_map[key]

    return None


def build_assistant_content(
    action_names: List[str],
    *,
    template: Dict[str, Any],
    include_think: bool,
    meta: Dict[str, Any],
) -> str:
    assistant_cfg = template.get("assistant", {})
    step_format = assistant_cfg.get("step_format", "{idx}. {action}")
    joiner = assistant_cfg.get("joiner", "\n")
    answer_prefix = assistant_cfg.get("answer_prefix", "")
    answer_suffix = assistant_cfg.get("answer_suffix", "")

    lines = [
        step_format.format(idx=i, action=action_names[i - 1])
        for i in range(1, len(action_names) + 1)
    ]
    answer_block = joiner.join(lines)

    parts: List[str] = []

    if include_think:
        think_prefix = assistant_cfg.get("think_prefix", "<think>")
        think_suffix = assistant_cfg.get("think_suffix", "</think>")
        think_lines = [
            "分析当前关卡：箱子数量 = {boxes}, 步数上限 = {max_steps}。".format(
                boxes=meta.get("boxes", "未知"),
                max_steps=meta.get("max_steps_hint", "未限制"),
            ),
            "准备执行 {length} 步解决该任务。".format(length=meta.get("length", "未知")),
        ]
        think_block = "\n".join(think_lines)
        parts.append(f"{think_prefix}\n{think_block}\n{think_suffix}")

    final_block = f"{answer_prefix}{answer_block}{answer_suffix}"
    parts.append(final_block.strip())

    return "\n".join(part for part in parts if part.strip())


def episode_to_sample(
    episode: Dict[str, Any],
    *,
    source_file: str,
    template: Dict[str, Any],
    include_think: bool,
    max_steps: Optional[int],
) -> Optional[Dict[str, Any]]:
    steps = episode.get("steps") or []
    if not steps:
        LOGGER.debug("样本无 steps，跳过: %s", source_file)
        return None

    if max_steps is not None and max_steps > 0:
        steps = steps[:max_steps]

    action_names: List[str] = []
    for idx, step in enumerate(steps, start=1):
        action_name = resolve_action_name(step, idx, template)
        if action_name is None:
            LOGGER.debug("无法解析动作名称 %s step #%d", source_file, idx)
            return None
        action_names.append(action_name)

    board_text = extract_initial_board(episode)
    if not board_text:
        LOGGER.debug("缺少初始棋盘信息，跳过: %s", source_file)
        return None

    system_prompt = template.get("system_prompt", "").strip()
    user_prompt_template = template.get("user_prompt", "").strip()
    if not system_prompt or not user_prompt_template:
        raise ValueError("模板中 system_prompt/user_prompt 缺失或为空")

    user_content = user_prompt_template.format(board=board_text)
    assistant_content = build_assistant_content(
        action_names,
        template=template,
        include_think=include_think,
        meta={
            "boxes": episode.get("num_boxes") or episode.get("boxes"),
            "max_steps_hint": max_steps,
            "length": len(action_names),
        },
    )

    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]

    total_reward = episode.get("total_reward")
    if total_reward is None:
        total_reward = sum(
            step.get("reward", 0.0)
            for step in steps
            if isinstance(step.get("reward"), (int, float))
        )

    sample_id = episode.get("id") or episode.get("episode_id")
    if not sample_id:
        stem = Path(source_file).stem
        sample_id = stem

    meta = {
        "source_file": source_file,
        "length": len(action_names),
        "success": bool(episode.get("success", False)),
        "total_reward": total_reward,
        "reward_per_step": (
            total_reward / len(action_names)
            if action_names
            else None
        ),
        "seed": episode.get("seed"),
        "actions": action_names,
    }

    return {
        "id": sample_id,
        "conversation": conversation,
        "meta": meta,
    }


def write_jsonl(path: Path, samples: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False))
            f.write("\n")


def summarize_dataset(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not samples:
        return {"num_samples": 0}

    lengths = [sample["meta"]["length"] for sample in samples]
    rewards = [
        sample["meta"]["total_reward"]
        for sample in samples
        if isinstance(sample["meta"]["total_reward"], (int, float))
    ]
    success_rate = sum(
        1 for sample in samples if sample["meta"].get("success")
    ) / len(samples)

    action_counter: Counter[str] = Counter()
    for sample in samples:
        action_counter.update(sample["meta"].get("actions") or [])

    stats = {
        "num_samples": len(samples),
        "avg_length": sum(lengths) / len(lengths),
        "max_length": max(lengths),
        "min_length": min(lengths),
        "median_length": (
            sorted(lengths)[len(lengths) // 2]
            if lengths
            else None
        ),
        "success_rate": success_rate,
        "actions_top10": action_counter.most_common(10),
    }

    if rewards:
        stats.update(
            {
                "avg_reward": sum(rewards) / len(rewards),
                "max_reward": max(rewards),
                "min_reward": min(rewards),
            }
        )

    return stats


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    input_dir = Path(args.input_dir)
    output_jsonl = Path(args.output_jsonl)
    val_output_jsonl = (
        Path(args.val_output_jsonl)
        if args.val_output_jsonl
        else None
    )
    stats_output_path = (
        Path(args.stats_output) if args.stats_output else None
    )

    if args.val_ratio > 0 and val_output_jsonl is None:
        raise ValueError(
            "--val-ratio > 0 时必须指定 --val-output-jsonl"
        )

    template = load_template(Path(args.template))

    episode_paths = sorted(input_dir.glob("episode_*.json"))
    if not episode_paths:
        LOGGER.error("目录中未找到 episode_*.json: %s", input_dir)
        return

    LOGGER.info("检测到 %d 个过滤后样本，开始转换…", len(episode_paths))

    samples: List[Dict[str, Any]] = []
    skipped = 0

    for path in episode_paths:
        episode = safe_load_json(path)
        if episode is None:
            skipped += 1
            continue

        sample = episode_to_sample(
            episode,
            source_file=path.name,
            template=template,
            include_think=args.include_think,
            max_steps=args.max_steps,
        )
        if sample is None:
            skipped += 1
            continue
        samples.append(sample)

    if not samples:
        LOGGER.error("无有效样本生成，请检查数据或模板配置")
        return

    LOGGER.info(
        "成功转换 %d 条样本，跳过 %d 条（缺失必要字段或解析失败）",
        len(samples),
        skipped,
    )

    rng = random.Random(args.seed)
    if args.val_ratio > 0:
        rng.shuffle(samples)
        val_size = max(1, int(math.floor(len(samples) * args.val_ratio)))
        val_samples = samples[:val_size]
        train_samples = samples[val_size:]
        LOGGER.info(
            "划分训练/验证：训练 %d 条，验证 %d 条 (%.1f%%)",
            len(train_samples),
            len(val_samples),
            args.val_ratio * 100,
        )
        write_jsonl(output_jsonl, train_samples)
        write_jsonl(val_output_jsonl, val_samples)

        train_stats = summarize_dataset(train_samples)
        val_stats = summarize_dataset(val_samples)
        LOGGER.info("训练集统计: %s", train_stats)
        LOGGER.info("验证集统计: %s", val_stats)

        if stats_output_path:
            stats_payload = {
                "train": train_stats,
                "val": val_stats,
                "skipped": skipped,
                "template": args.template,
                "include_think": args.include_think,
                "max_steps": args.max_steps,
            }
            stats_output_path.parent.mkdir(parents=True, exist_ok=True)
            with stats_output_path.open("w", encoding="utf-8") as f:
                json.dump(stats_payload, f, ensure_ascii=False, indent=2)
    else:
        write_jsonl(output_jsonl, samples)
        dataset_stats = summarize_dataset(samples)
        LOGGER.info("数据集统计: %s", dataset_stats)

        if stats_output_path:
            stats_payload = {
                "dataset": dataset_stats,
                "skipped": skipped,
                "template": args.template,
                "include_think": args.include_think,
                "max_steps": args.max_steps,
            }
            stats_output_path.parent.mkdir(parents=True, exist_ok=True)
            with stats_output_path.open("w", encoding="utf-8") as f:
                json.dump(stats_payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()