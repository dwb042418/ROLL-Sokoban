#!/usr/bin/env python3
"""
简单校验 SFT JSONL 数据的结构合法性。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

REQUIRED_ROLES = ["system", "user", "assistant"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="校验 SFT 对话 JSONL 文件"
    )
    parser.add_argument(
        "--input-jsonl",
        type=str,
        required=True,
        help="待校验的 JSONL 文件",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="仅检查前 N 行（默认检查全部）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.input_jsonl)
    if not path.exists():
        raise FileNotFoundError(path)

    total = 0
    for line_idx, raw in enumerate(path.open("r", encoding="utf-8"), start=1):
        if args.max_lines and line_idx > args.max_lines:
            break

        line = raw.strip()
        if not line:
            continue

        try:
            data: Dict[str, Any] = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"第 {line_idx} 行 JSON 解析失败: {exc}") from exc

        if "conversation" not in data or not isinstance(
            data["conversation"], list
        ):
            raise ValueError(f"第 {line_idx} 行缺少 conversation 列表")

        conv: List[Dict[str, Any]] = data["conversation"]
        if len(conv) < 3:
            raise ValueError(f"第 {line_idx} 行会话长度 < 3")

        roles = [turn.get("role") for turn in conv[:3]]
        if roles != REQUIRED_ROLES:
            raise ValueError(
                f"第 {line_idx} 行前三个角色应为 {REQUIRED_ROLES}，实际为 {roles}"
            )

        if "meta" not in data or not isinstance(data["meta"], dict):
            raise ValueError(f"第 {line_idx} 行缺少 meta 字段")

        meta = data["meta"]
        if "length" not in meta or not isinstance(meta["length"], int):
            raise ValueError(f"第 {line_idx} 行 meta.length 无效")
        if "actions" not in meta or not isinstance(meta["actions"], list):
            raise ValueError(f"第 {line_idx} 行 meta.actions 无效")
        if meta["length"] != len(meta["actions"]):
            raise ValueError(
                f"第 {line_idx} 行长度不匹配: meta.length={meta['length']} vs actions={len(meta['actions'])}"
            )

        total += 1

    print(f"校验通过，共检查 {total} 条样本。")


if __name__ == "__main__":
    main()