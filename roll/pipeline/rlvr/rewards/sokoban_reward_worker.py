"""
Sokoban Environment Reward Worker for ROLL RLVR
基于环境的Reward Worker，用于GRPO训练
"""

import os
import sys
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

import torch
import numpy as np
import re

from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto


@dataclass
class SokobanRewardConfig:
    """Sokoban奖励配置"""
    success_reward: float = 10.0          # 完成任务奖励
    step_penalty: float = -0.1            # 每步惩罚（鼓励快速完成）
    invalid_action_penalty: float = -1.0   # 无效动作惩罚
    box_on_target_reward: float = 1.0     # 每个箱子到位奖励
    max_steps: int = 20                   # 最大步数
    dim_room: tuple = (10, 10)            # 房间大小
    num_boxes: int = 4                    # 箱子数量


class SokobanEnvRewardWorker(Worker):
    """
    Sokoban环境Reward Worker

    功能：
    1. 接收模型生成的动作序列
    2. 在Sokoban环境中执行
    3. 根据环境反馈计算奖励
    4. 返回奖励给GRPO算法
    """

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config)

        # 从 worker_config 中解析奖励配置（使用 hasattr 检查动态属性）
        success_reward = getattr(worker_config, 'success_reward', 10.0)
        step_penalty = getattr(worker_config, 'step_penalty', -0.1)
        invalid_action_penalty = getattr(worker_config, 'invalid_action_penalty', -1.0)
        box_on_target_reward = getattr(worker_config, 'box_on_target_reward', 1.0)

        # 从 data_args 中解析环境配置（data_args 可能是 None）
        if worker_config.data_args is not None:
            max_steps = getattr(worker_config.data_args, 'max_steps', 20)
            dim_room = getattr(worker_config.data_args, 'dim_room', [10, 10])
            num_boxes = getattr(worker_config.data_args, 'num_boxes', 4)
        else:
            max_steps = 20
            dim_room = [10, 10]
            num_boxes = 4

        # 创建配置对象
        self.reward_cfg = SokobanRewardConfig(
            success_reward=success_reward,
            step_penalty=step_penalty,
            invalid_action_penalty=invalid_action_penalty,
            box_on_target_reward=box_on_target_reward,
            max_steps=max_steps,
            dim_room=tuple(dim_room),
            num_boxes=num_boxes,
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        pass

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)
    def compute_rewards(self, data: DataProto):
        """
        计算奖励的主入口

        Args:
            data: 包含prompts和responses的数据

        Returns:
            DataProto with rewards
        """
        prompts = data.non_tensor_batch.get("prompt", [])
        responses = data.non_tensor_batch.get("response", [])
        meta_info = data.non_tensor_batch.get("meta_info", [{}] * len(prompts))

        # 计算奖励
        rewards = []
        for prompt, response, meta in zip(prompts, responses, meta_info):
            try:
                reward = self._compute_single_reward(prompt, response, meta)
                rewards.append(reward)
            except Exception as e:
                print(f"Warning: Error computing reward: {e}")
                rewards.append(-10.0)

        # 转换为tensor
        reward_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)

        # 创建返回的数据
        data.batch["response_level_rewards"] = reward_tensor
        return data

    def _compute_single_reward(self, prompt: str, response: str, meta: Dict) -> float:
        """计算单个episode的奖励"""

        # 1. 从response中解析动作序列
        actions = self._parse_actions(response)

        # 2. 简化的奖励计算
        # 基于动作数量和格式给予奖励
        total_reward = 0.0

        if len(actions) == 0:
            # 没有识别出任何动作
            return -5.0

        # 每个有效动作给予小惩罚（鼓励简短）
        total_reward += self.reward_cfg.step_penalty * len(actions)

        # 如果动作数量合理，给予奖励
        if 5 <= len(actions) <= self.reward_cfg.max_steps:
            total_reward += 2.0

        # 检查是否有动作变化（避免重复同一动作）
        unique_actions = len(set(actions))
        if unique_actions >= 3:
            total_reward += 1.0

        # 限制奖励范围
        return float(np.clip(total_reward, -10.0, 10.0))

    def _parse_actions(self, response: str) -> List[str]:
        """从响应中解析动作序列"""
        actions = []

        # 格式1: <answer>Action</answer>
        pattern = r'<answer>\s*(Up|Down|Left|Right)\s*</answer>'
        matches = re.findall(pattern, response, re.IGNORECASE)

        if matches:
            return matches

        # 格式2: 纯文本，逐行解析
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 查找动作关键词
            for action in ['Up', 'Down', 'Left', 'Right']:
                if action.lower() in line.lower():
                    actions.append(action)
                    break

        # 限制最大步数
        return actions[:self.reward_cfg.max_steps]
