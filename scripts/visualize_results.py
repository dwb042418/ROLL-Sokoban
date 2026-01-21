#!/usr/bin/env python3
"""
训练结果可视化脚本
生成训练曲线图和评测结果对比图
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def load_tensorboard_logs(log_dir: str) -> Dict[str, List[Tuple[int, float]]]:
    """加载TensorBoard日志

    Args:
        log_dir: TensorBoard日志目录

    Returns:
        字典，key为指标名称，value为(step, value)列表
    """
    logs = {}

    # 查找所有event文件所在的目录
    event_dirs = []
    if os.path.isfile(log_dir):
        # 单个event文件
        event_dirs = [os.path.dirname(log_dir)]
    else:
        # 目录，查找所有包含event文件的子目录
        for root, dirs, files in os.walk(log_dir):
            if any(f.startswith('events.out.tfevents') for f in files):
                event_dirs.append(root)

    # 按名称排序（通常是时间戳），使用最新的
    event_dirs.sort()

    print(f"   找到 {len(event_dirs)} 个event目录")
    if len(event_dirs) == 0:
        print(f"   警告: 在 {log_dir} 中未找到TensorBoard event文件")
        return logs

    # 只使用最新的几个目录
    for event_dir in event_dirs[-3:]:
        try:
            print(f"   正在加载: {event_dir}")
            ea = event_accumulator.EventAccumulator(event_dir)
            ea.Reload()

            # 读取所有scalar数据
            tags = ea.Tags()['scalars']
            print(f"   找到 {len(tags)} 个指标")

            for tag in tags:
                if tag not in logs:
                    logs[tag] = []

                events = ea.Scalars(tag)
                for event in events:
                    logs[tag].append((event.step, event.value))

            # 如果已经有数据了，就停止加载更早的目录
            if len(logs) > 0:
                break

        except Exception as e:
            print(f"   Warning: Failed to load {event_dir}: {e}")
            continue

    return logs


def load_evaluation_results(eval_dir: str) -> List[Dict]:
    """加载评测结果JSON文件

    Args:
        eval_dir: 评测结果目录

    Returns:
        评测结果列表
    """
    results = []

    eval_path = Path(eval_dir)
    if not eval_path.exists():
        return results

    for json_file in eval_path.glob('*.json'):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                data['source_file'] = json_file.name
                results.append(data)
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")

    return results


def plot_training_curves(logs: Dict[str, List[Tuple[int, float]]], output_dir: str):
    """绘制训练曲线

    Args:
        logs: TensorBoard日志数据
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Loss曲线
    if 'sft_train/loss' in logs:
        plt.figure(figsize=(10, 6))
        steps, values = zip(*logs['sft_train/loss'])

        plt.plot(steps, values, 'b-', linewidth=2, label='Training Loss')
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Sokoban SFT Training Loss', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # 添加数值标注
        if len(values) > 0:
            plt.text(steps[0], values[0], f'{values[0]:.4f}',
                    fontsize=10, verticalalignment='bottom')
            plt.text(steps[-1], values[-1], f'{values[-1]:.4f}',
                    fontsize=10, verticalalignment='top')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_loss.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_dir}/training_loss.png")
        plt.close()

    # 2. Gradient Norm曲线
    if 'sft_train/grad_norm' in logs:
        plt.figure(figsize=(10, 6))
        steps, values = zip(*logs['sft_train/grad_norm'])

        plt.plot(steps, values, 'r-', linewidth=2, label='Gradient Norm')
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Gradient Norm', fontsize=12)
        plt.title('Gradient Norm During Training', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gradient_norm.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_dir}/gradient_norm.png")
        plt.close()

    # 3. Training Speed曲线
    if 'time/step_train' in logs:
        plt.figure(figsize=(10, 6))
        steps, values = zip(*logs['time/step_train'])

        plt.plot(steps, values, 'g-', linewidth=2, label='Time per Step')
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Time (seconds)', fontsize=12)
        plt.title('Training Speed (Time per Step)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # 添加平均线
        avg_time = np.mean(values)
        plt.axhline(y=avg_time, color='orange', linestyle='--',
                   label=f'Average: {avg_time:.2f}s')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_speed.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_dir}/training_speed.png")
        plt.close()

    # 4. 综合训练曲线（Loss + Grad Norm）
    if 'sft_train/loss' in logs and 'sft_train/grad_norm' in logs:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Loss曲线
        steps, values = zip(*logs['sft_train/loss'])
        ax1.plot(steps, values, 'b-', linewidth=2)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training Loss', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Gradient Norm曲线
        steps, values = zip(*logs['sft_train/grad_norm'])
        ax2.plot(steps, values, 'r-', linewidth=2)
        ax2.set_xlabel('Training Steps', fontsize=12)
        ax2.set_ylabel('Gradient Norm', fontsize=12)
        ax2.set_title('Gradient Norm', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.suptitle('Sokoban SFT Training Metrics', fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_overview.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_dir}/training_overview.png")
        plt.close()


def plot_evaluation_comparison(results: List[Dict], output_dir: str):
    """绘制评测结果对比图

    Args:
        results: 评测结果列表
        output_dir: 输出目录
    """
    if len(results) == 0:
        print("No evaluation results found, skipping comparison plots.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 提取关键指标
    models = []
    success_rates = []
    avg_steps = []
    avg_rewards = []

    for result in results:
        if 'evaluation' in result:
            eval_data = result['evaluation']

            # 模型名称
            model_name = result.get('source_file', 'Unknown').replace('.json', '')
            models.append(model_name)

            # 指标
            success_rates.append(eval_data.get('success_rate', 0))
            avg_steps.append(eval_data.get('avg_steps', 0))
            avg_rewards.append(eval_data.get('avg_reward', 0))

    if len(models) == 0:
        print("No valid evaluation data found.")
        return

    # 1. Success Rate对比
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(models)), success_rates, color='steelblue', alpha=0.7)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Success Rate (%)', fontsize=12)
    plt.title('Success Rate Comparison', fontsize=14, fontweight='bold')
    plt.xticks(range(len(models)), models, rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)

    # 添加数值标注
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eval_success_rate.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/eval_success_rate.png")
    plt.close()

    # 2. Avg Steps对比
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(models)), avg_steps, color='coral', alpha=0.7)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Average Steps', fontsize=12)
    plt.title('Average Steps Comparison', fontsize=14, fontweight='bold')
    plt.xticks(range(len(models)), models, rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)

    # 添加数值标注
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eval_avg_steps.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/eval_avg_steps.png")
    plt.close()

    # 3. Avg Reward对比
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(models)), avg_rewards, color='seagreen', alpha=0.7)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.title('Average Reward Comparison', fontsize=14, fontweight='bold')
    plt.xticks(range(len(models)), models, rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)

    # 添加数值标注
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eval_avg_reward.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/eval_avg_reward.png")
    plt.close()

    # 4. 综合对比图
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    # Success Rate
    bars1 = ax1.bar(range(len(models)), success_rates, color='steelblue', alpha=0.7)
    ax1.set_ylabel('Success Rate (%)', fontsize=11)
    ax1.set_title('Success Rate', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax1.grid(True, axis='y', alpha=0.3)
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    # Avg Steps
    bars2 = ax2.bar(range(len(models)), avg_steps, color='coral', alpha=0.7)
    ax2.set_ylabel('Average Steps', fontsize=11)
    ax2.set_title('Average Steps', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax2.grid(True, axis='y', alpha=0.3)
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    # Avg Reward
    bars3 = ax3.bar(range(len(models)), avg_rewards, color='seagreen', alpha=0.7)
    ax3.set_ylabel('Average Reward', fontsize=11)
    ax3.set_title('Average Reward', fontsize=12, fontweight='bold')
    ax3.set_xticks(range(len(models)))
    ax3.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax3.grid(True, axis='y', alpha=0.3)
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('Evaluation Metrics Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eval_comparison_overview.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/eval_comparison_overview.png")
    plt.close()


def plot_training_summary_table(logs: Dict[str, List[Tuple[int, float]]], output_dir: str):
    """生成训练统计表格图

    Args:
        logs: TensorBoard日志数据
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 提取统计信息
    stats = []

    for tag, data in logs.items():
        if len(data) > 0:
            steps, values = zip(*data)
            stats.append({
                'Metric': tag.split('/')[-1],  # 去掉前缀
                'Initial': values[0],
                'Final': values[-1],
                'Min': min(values),
                'Max': max(values),
                'Mean': np.mean(values),
                'Std': np.std(values),
            })

    if len(stats) == 0:
        return

    # 绘制表格
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')

    # 准备表格数据
    columns = ['Metric', 'Initial', 'Final', 'Min', 'Max', 'Mean', 'Std']
    rows = []
    for stat in stats:
        row = [
            stat['Metric'],
            f"{stat['Initial']:.4f}",
            f"{stat['Final']:.4f}",
            f"{stat['Min']:.4f}",
            f"{stat['Max']:.4f}",
            f"{stat['Mean']:.4f}",
            f"{stat['Std']:.4f}",
        ]
        rows.append(row)

    # 创建表格
    table = ax.table(cellText=rows, colLabels=columns,
                    cellLoc='center', loc='center',
                    colWidths=[0.2, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # 设置表头样式
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # 设置交替行颜色
    for i in range(1, len(rows) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')

    plt.title('Training Statistics Summary', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(os.path.join(output_dir, 'training_stats_table.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/training_stats_table.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='可视化训练和评测结果')
    parser.add_argument('--tensorboard-dir', type=str,
                       default='output/tensorboard/sokoban_sft_baseline',
                       help='TensorBoard日志目录')
    parser.add_argument('--eval-dir', type=str,
                       default='output/evals',
                       help='评测结果目录')
    parser.add_argument('--output-dir', type=str,
                       default='reports/figures',
                       help='输出目录')

    args = parser.parse_args()

    print("="*60)
    print("训练结果可视化工具")
    print("="*60)
    print(f"TensorBoard目录: {args.tensorboard_dir}")
    print(f"评测结果目录: {args.eval_dir}")
    print(f"输出目录: {args.output_dir}")
    print("="*60)
    print()

    # 1. 加载TensorBoard日志
    print("1. 加载TensorBoard日志...")
    logs = load_tensorboard_logs(args.tensorboard_dir)
    print(f"   找到 {len(logs)} 个指标")
    for tag in logs.keys():
        print(f"   - {tag}: {len(logs[tag])} 个数据点")
    print()

    # 2. 绘制训练曲线
    if len(logs) > 0:
        print("2. 生成训练曲线图...")
        plot_training_curves(logs, args.output_dir)
        plot_training_summary_table(logs, args.output_dir)
        print()

    # 3. 加载评测结果
    print("3. 加载评测结果...")
    results = load_evaluation_results(args.eval_dir)
    print(f"   找到 {len(results)} 个评测结果")
    for result in results:
        model_name = result.get('source_file', 'Unknown')
        if 'evaluation' in result:
            print(f"   - {model_name}: "
                  f"Success Rate={result['evaluation'].get('success_rate', 0):.1f}%, "
                  f"Avg Steps={result['evaluation'].get('avg_steps', 0):.1f}")
    print()

    # 4. 绘制评测对比图
    if len(results) > 0:
        print("4. 生成评测对比图...")
        plot_evaluation_comparison(results, args.output_dir)
        print()

    print("="*60)
    print(f"✓ 所有图表已保存到: {args.output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
