#!/usr/bin/env python3
"""
RL训练曲线可视化脚本
用于可视化GRPO训练过程中的各项指标
"""

import os
import argparse
from pathlib import Path
import numpy as np
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib or seaborn not available. Please install: pip install matplotlib seaborn")

try:
    from tensorboard.backend.event_processing import event_accumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: tensorboard not available. Please install: pip install tensorboard")


def parse_tensorboard_logs(log_dir):
    """解析TensorBoard日志"""
    if not TENSORBOARD_AVAILABLE:
        raise ImportError("TensorBoard is not installed. Run: pip install tensorboard")

    if not os.path.exists(log_dir):
        raise ValueError(f"Log directory does not exist: {log_dir}")

    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    # 获取所有标量数据
    scalar_data = {}
    for tag in ea.Tags()['scalars']:
        try:
            scalar_data[tag] = ea.Scalars(tag)
        except Exception as e:
            print(f"Warning: Failed to read tag {tag}: {e}")

    return scalar_data


def plot_training_curves(scalar_data, output_dir, exp_name="GRPO Training"):
    """绘制训练曲线"""
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping plot generation: matplotlib not available")
        return

    # 设置样式
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (18, 12)
    plt.rcParams['font.size'] = 11
    plt.rcParams['font.family'] = 'DejaVu Sans'

    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{exp_name} - Training Metrics', fontsize=18, fontweight='bold', y=0.995)

    # 定义颜色方案
    colors = {
        'policy_loss': '#2E86AB',
        'total_loss': '#A23B72',
        'kl_loss': '#F18F01',
        'reward': '#C73E1D',
        'success_rate': '#3B1F2B',
        'grad_norm': '#6B705C'
    }

    # 1. Policy Loss
    if 'actor_train/policy_loss' in scalar_data:
        data = scalar_data['actor_train/policy_loss']
        steps = [d.step for d in data]
        values = [d.value for d in data]
        axes[0, 0].plot(steps, values, linewidth=2.5, color=colors['policy_loss'], alpha=0.8)
        axes[0, 0].fill_between(steps, values, alpha=0.3, color=colors['policy_loss'])
        axes[0, 0].set_title('Policy Loss', fontweight='bold', fontsize=13)
        axes[0, 0].set_xlabel('Training Steps')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3, linestyle='--')
    else:
        axes[0, 0].text(0.5, 0.5, 'Policy Loss data not available',
                        ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Policy Loss', fontweight='bold', fontsize=13)

    # 2. Total Loss
    if 'actor_train/loss' in scalar_data:
        data = scalar_data['actor_train/loss']
        steps = [d.step for d in data]
        values = [d.value for d in data]
        axes[0, 1].plot(steps, values, linewidth=2.5, color=colors['total_loss'], alpha=0.8)
        axes[0, 1].fill_between(steps, values, alpha=0.3, color=colors['total_loss'])
        axes[0, 1].set_title('Total Loss', fontweight='bold', fontsize=13)
        axes[0, 1].set_xlabel('Training Steps')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3, linestyle='--')
    else:
        axes[0, 1].text(0.5, 0.5, 'Total Loss data not available',
                        ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Total Loss', fontweight='bold', fontsize=13)

    # 3. KL Divergence Loss
    if 'actor_train/kl_loss' in scalar_data:
        data = scalar_data['actor_train/kl_loss']
        steps = [d.step for d in data]
        values = [d.value for d in data]
        axes[0, 2].plot(steps, values, linewidth=2.5, color=colors['kl_loss'], alpha=0.8)
        axes[0, 2].fill_between(steps, values, alpha=0.3, color=colors['kl_loss'])
        axes[0, 2].set_title('KL Divergence Loss', fontweight='bold', fontsize=13)
        axes[0, 2].set_xlabel('Training Steps')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].grid(True, alpha=0.3, linestyle='--')
    else:
        axes[0, 2].text(0.5, 0.5, 'KL Loss data not available',
                        ha='center', va='center', transform=axes[0, 2].transAxes)
        axes[0, 2].set_title('KL Divergence Loss', fontweight='bold', fontsize=13)

    # 4. Mean Reward
    if 'rollout/mean_reward' in scalar_data:
        data = scalar_data['rollout/mean_reward']
        steps = [d.step for d in data]
        values = [d.value for d in data]
        axes[1, 0].plot(steps, values, linewidth=2.5, color=colors['reward'], alpha=0.8, marker='o', markersize=3)
        axes[1, 0].fill_between(steps, values, alpha=0.3, color=colors['reward'])
        axes[1, 0].set_title('Mean Reward', fontweight='bold', fontsize=13)
        axes[1, 0].set_xlabel('Training Steps')
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].grid(True, alpha=0.3, linestyle='--')
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    else:
        axes[1, 0].text(0.5, 0.5, 'Mean Reward data not available',
                        ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Mean Reward', fontweight='bold', fontsize=13)

    # 5. Success Rate
    if 'rollout/success_rate' in scalar_data:
        data = scalar_data['rollout/success_rate']
        steps = [d.step for d in data]
        values = [d.value for d in data]
        axes[1, 1].plot(steps, values, linewidth=2.5, color=colors['success_rate'], alpha=0.8, marker='s', markersize=3)
        axes[1, 1].fill_between(steps, values, alpha=0.3, color=colors['success_rate'])
        axes[1, 1].set_title('Success Rate', fontweight='bold', fontsize=13)
        axes[1, 1].set_xlabel('Training Steps')
        axes[1, 1].set_ylabel('Success Rate (%)')
        axes[1, 1].grid(True, alpha=0.3, linestyle='--')

        # 添加趋势线
        if len(values) > 3:
            try:
                z = np.polyfit(steps, values, 3)
                p = np.poly1d(z)
                axes[1, 1].plot(steps, p(steps), "--", alpha=0.6, color='red', linewidth=2, label='Trend')
                axes[1, 1].legend(loc='best', fontsize=9)
            except:
                pass
    else:
        axes[1, 1].text(0.5, 0.5, 'Success Rate data not available',
                        ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Success Rate', fontweight='bold', fontsize=13)

    # 6. Gradient Norm
    if 'actor_train/grad_norm' in scalar_data:
        data = scalar_data['actor_train/grad_norm']
        steps = [d.step for d in data]
        values = [d.value for d in data]
        axes[1, 2].plot(steps, values, linewidth=2.5, color=colors['grad_norm'], alpha=0.8)
        axes[1, 2].fill_between(steps, values, alpha=0.3, color=colors['grad_norm'])
        axes[1, 2].set_title('Gradient Norm', fontweight='bold', fontsize=13)
        axes[1, 2].set_xlabel('Training Steps')
        axes[1, 2].set_ylabel('Gradient Norm')
        axes[1, 2].grid(True, alpha=0.3, linestyle='--')
    else:
        axes[1, 2].text(0.5, 0.5, 'Gradient Norm data not available',
                        ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Gradient Norm', fontweight='bold', fontsize=13)

    plt.tight_layout()

    # 保存图片
    output_path = os.path.join(output_dir, 'rl_training_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 训练曲线已保存到: {output_path}")

    # 保存PDF版本
    output_path_pdf = os.path.join(output_dir, 'rl_training_curves.pdf')
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"✓ 训练曲线已保存到: {output_path_pdf}")

    plt.close()


def plot_comparison_curves(scalar_data, output_dir, exp_name="GRPO Training"):
    """绘制对比曲线"""
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping comparison plot: matplotlib not available")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{exp_name} - Analysis', fontsize=18, fontweight='bold')

    # 1. Loss Components Comparison
    if 'actor_train/policy_loss' in scalar_data and 'actor_train/kl_loss' in scalar_data:
        policy_data = scalar_data['actor_train/policy_loss']
        kl_data = scalar_data['actor_train/kl_loss']

        steps_policy = [d.step for d in policy_data]
        values_policy = [d.value for d in policy_data]

        steps_kl = [d.step for d in kl_data]
        values_kl = [d.value for d in kl_data]

        axes[0].plot(steps_policy, values_policy, label='Policy Loss',
                    linewidth=2.5, color='#2E86AB', alpha=0.8)
        axes[0].plot(steps_kl, values_kl, label='KL Loss',
                    linewidth=2.5, color='#F18F01', alpha=0.8)
        axes[0].set_title('Loss Components Comparison', fontweight='bold', fontsize=14)
        axes[0].set_xlabel('Training Steps')
        axes[0].set_ylabel('Loss')
        axes[0].legend(fontsize=11, loc='best')
        axes[0].grid(True, alpha=0.3, linestyle='--')
    else:
        axes[0].text(0.5, 0.5, 'Loss data not available',
                    ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title('Loss Components Comparison', fontweight='bold', fontsize=14)

    # 2. Reward vs Success Rate
    if 'rollout/mean_reward' in scalar_data and 'rollout/success_rate' in scalar_data:
        reward_data = scalar_data['rollout/mean_reward']
        success_data = scalar_data['rollout/success_rate']

        reward_steps = [d.step for d in reward_data]
        reward_values = [d.value for d in reward_data]

        success_steps = [d.step for d in success_data]
        success_values = [d.value for d in success_data]

        # 双y轴
        color1 = 'tab:blue'
        axes[1].set_xlabel('Training Steps', fontsize=12)
        axes[1].set_ylabel('Mean Reward', color=color1, fontsize=12)
        line1 = axes[1].plot(reward_steps, reward_values, color=color1,
                            label='Mean Reward', linewidth=2.5, alpha=0.8, marker='o', markersize=3)
        axes[1].tick_params(axis='y', labelcolor=color1)
        axes[1].grid(True, alpha=0.3, linestyle='--')

        ax2 = axes[1].twinx()
        color2 = 'tab:orange'
        ax2.set_ylabel('Success Rate (%)', color=color2, fontsize=12)
        line2 = ax2.plot(success_steps, success_values, color=color2,
                        label='Success Rate', linewidth=2.5, alpha=0.8, marker='s', markersize=3)
        ax2.tick_params(axis='y', labelcolor=color2)

        axes[1].set_title('Reward vs Success Rate', fontweight='bold', fontsize=14)

        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        axes[1].legend(lines, labels, fontsize=11, loc='best')
    else:
        axes[1].text(0.5, 0.5, 'Reward/Success Rate data not available',
                    ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Reward vs Success Rate', fontweight='bold', fontsize=14)

    plt.tight_layout()

    # 保存图片
    output_path = os.path.join(output_dir, 'rl_analysis_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 分析曲线已保存到: {output_path}")

    # 保存PDF版本
    output_path_pdf = os.path.join(output_dir, 'rl_analysis_curves.pdf')
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"✓ 分析曲线已保存到: {output_path_pdf}")

    plt.close()


def print_summary_statistics(scalar_data):
    """打印统计摘要"""
    print("\n" + "="*60)
    print("训练统计摘要")
    print("="*60)

    # 打印可用指标
    print("\n可用指标:")
    for tag in sorted(scalar_data.keys()):
        data = scalar_data[tag]
        if len(data) > 0:
            first_val = data[0].value
            last_val = data[-1].value
            min_val = min(d.value for d in data)
            max_val = max(d.value for d in data)
            print(f"  {tag}:")
            print(f"    - 数据点: {len(data)}")
            print(f"    - 初始值: {first_val:.4f}")
            print(f"    - 最终值: {last_val:.4f}")
            print(f"    - 最小值: {min_val:.4f}")
            print(f"    - 最大值: {max_val:.4f}")


def main():
    parser = argparse.ArgumentParser(description='可视化RL训练曲线')
    parser.add_argument('--tensorboard-dir', type=str, required=True,
                        help='TensorBoard日志目录')
    parser.add_argument('--output-dir', type=str, default='reports/figures',
                        help='输出目录')
    parser.add_argument('--exp-name', type=str, default='GRPO Training - Sokoban Llama-3.2-3B',
                        help='实验名称（用于图表标题）')
    parser.add_argument('--no-plots', action='store_true',
                        help='只显示统计信息，不生成图表')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*60)
    print("RL训练曲线可视化工具")
    print("="*60)
    print(f"TensorBoard目录: {args.tensorboard_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"实验名称: {args.exp_name}")
    print("="*60 + "\n")

    # 解析日志
    print("正在解析TensorBoard日志...")
    try:
        scalar_data = parse_tensorboard_logs(args.tensorboard_dir)
        print(f"✓ 成功解析，找到 {len(scalar_data)} 个指标\n")
    except Exception as e:
        print(f"✗ 解析失败: {e}")
        return

    # 打印统计摘要
    print_summary_statistics(scalar_data)

    # 绘制曲线
    if not args.no_plots:
        if not MATPLOTLIB_AVAILABLE:
            print("\n✗ 无法生成图表：matplotlib 或 seaborn 未安装")
            print("  请运行: pip install matplotlib seaborn")
            return

        print("\n正在生成图表...")
        plot_training_curves(scalar_data, args.output_dir, args.exp_name)
        plot_comparison_curves(scalar_data, args.output_dir, args.exp_name)

        print("\n" + "="*60)
        print("✓ 可视化完成！")
        print("="*60)
        print(f"图表已保存到: {args.output_dir}")
        print(f"  - rl_training_curves.png (训练指标)")
        print(f"  - rl_training_curves.pdf")
        print(f"  - rl_analysis_curves.png (对比分析)")
        print(f"  - rl_analysis_curves.pdf")
        print("="*60)


if __name__ == '__main__':
    main()
