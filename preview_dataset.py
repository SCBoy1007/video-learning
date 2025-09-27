#!/usr/bin/env python3
"""
快速预览 VisionReasoner 数据集的脚本
"""
import os
import json
from datasets import load_from_disk
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

def preview_dataset(num_samples=3):
    """预览数据集中的样本"""

    # 数据集路径 - 相对于脚本位置
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "data", "VisionReasoner_multi_object_1k_840")

    if not os.path.exists(dataset_path):
        print(f"❌ 数据集路径不存在: {dataset_path}")
        return

    print("🔍 加载VisionReasoner数据集...")
    try:
        dataset = load_from_disk(dataset_path)
        train_data = dataset['train']
        print(f"✅ 数据集加载成功! 共有 {len(train_data)} 个训练样本")
    except Exception as e:
        print(f"❌ 加载数据集失败: {e}")
        return

    print(f"\n📊 数据集信息:")
    print(f"   - 样本数量: {len(train_data)}")
    print(f"   - 数据字段: {list(train_data.features.keys())}")

    # 预览前几个样本
    print(f"\n🔍 预览前 {num_samples} 个样本:")
    print("=" * 80)

    for i in range(min(num_samples, len(train_data))):
        sample = train_data[i]

        print(f"\n📋 样本 {i+1}:")
        print(f"   ID: {sample['id']}")
        print(f"   图像尺寸: {sample['img_width']} x {sample['img_height']}")

        # 显示问题
        problem = sample['problem']
        print(f"   问题: {problem}")

        # 显示答案并解析
        solution = sample['solution']
        print(f"   答案: {solution}")

        # 尝试解析答案中的坐标信息
        try:
            if solution.strip():
                solution_data = json.loads(solution)
                if isinstance(solution_data, list) and len(solution_data) > 0:
                    print(f"   📍 检测到 {len(solution_data)} 个目标:")
                    for j, obj in enumerate(solution_data):
                        if 'bbox_2d' in obj:
                            bbox = obj['bbox_2d']
                            print(f"      目标{j+1}: bbox={bbox}")
                        if 'point_2d' in obj:
                            point = obj['point_2d']
                            print(f"      目标{j+1}: point={point}")
        except json.JSONDecodeError:
            print(f"   ⚠️  答案格式解析失败")

        print("-" * 60)

def visualize_sample(sample_idx=0, save_path="preview_sample.png"):
    """可视化指定样本，显示图像和标注"""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "data", "VisionReasoner_multi_object_1k_840")

    print(f"🎨 可视化样本 {sample_idx}...")

    try:
        dataset = load_from_disk(dataset_path)
        train_data = dataset['train']

        if sample_idx >= len(train_data):
            print(f"❌ 样本索引超出范围: {sample_idx} >= {len(train_data)}")
            return

        sample = train_data[sample_idx]

        # 获取图像
        image = sample['image']
        width, height = sample['img_width'], sample['img_height']

        # 创建图形
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        ax.set_title(f"Sample {sample_idx}: {sample['problem'][:50]}...", fontsize=12, pad=20)

        # 解析并绘制标注
        try:
            solution = sample['solution']
            if solution.strip():
                solution_data = json.loads(solution)

                colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']

                for i, obj in enumerate(solution_data):
                    color = colors[i % len(colors)]

                    # 绘制bbox
                    if 'bbox_2d' in obj:
                        bbox = obj['bbox_2d']
                        x1, y1, x2, y2 = bbox

                        # 创建矩形
                        rect = patches.Rectangle(
                            (x1, y1), x2-x1, y2-y1,
                            linewidth=2, edgecolor=color, facecolor='none'
                        )
                        ax.add_patch(rect)

                        # 添加标签
                        ax.text(x1, y1-5, f'Object {i+1}',
                               color=color, fontweight='bold', fontsize=10,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

                    # 绘制点
                    if 'point_2d' in obj:
                        point = obj['point_2d']
                        px, py = point
                        ax.plot(px, py, 'o', color=color, markersize=8, markeredgecolor='white', markeredgewidth=2)

        except json.JSONDecodeError:
            print("⚠️  无法解析标注信息")

        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)  # 翻转Y轴
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 可视化结果已保存到: {save_path}")
        plt.show()

    except Exception as e:
        print(f"❌ 可视化失败: {e}")

def show_dataset_stats():
    """显示数据集统计信息"""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "data", "VisionReasoner_multi_object_1k_840")

    print("📊 数据集统计信息:")

    try:
        dataset = load_from_disk(dataset_path)
        train_data = dataset['train']

        # 基本统计
        total_samples = len(train_data)
        print(f"   总样本数: {total_samples}")

        # 统计图像尺寸分布
        widths = [sample['img_width'] for sample in train_data]
        heights = [sample['img_height'] for sample in train_data]

        print(f"   图像宽度范围: {min(widths)} - {max(widths)}")
        print(f"   图像高度范围: {min(heights)} - {max(heights)}")
        print(f"   最常见尺寸: {max(set(zip(widths, heights)), key=lambda x: list(zip(widths, heights)).count(x))}")

        # 统计问题长度
        problem_lengths = [len(sample['problem']) for sample in train_data]
        print(f"   问题长度范围: {min(problem_lengths)} - {max(problem_lengths)} 字符")
        print(f"   平均问题长度: {sum(problem_lengths)/len(problem_lengths):.1f} 字符")

        # 统计目标数量
        object_counts = []
        for sample in train_data:
            try:
                solution = sample['solution']
                if solution.strip():
                    solution_data = json.loads(solution)
                    object_counts.append(len(solution_data))
                else:
                    object_counts.append(0)
            except:
                object_counts.append(0)

        print(f"   每个样本目标数量范围: {min(object_counts)} - {max(object_counts)}")
        print(f"   平均每个样本目标数量: {sum(object_counts)/len(object_counts):.1f}")

    except Exception as e:
        print(f"❌ 统计失败: {e}")

if __name__ == "__main__":
    print("🚀 VisionReasoner 数据集预览工具")
    print("=" * 50)

    # 显示数据集统计
    show_dataset_stats()

    # 预览样本
    preview_dataset(num_samples=5)

    # 可视化第一个样本
    print(f"\n🎨 生成第一个样本的可视化图像...")
    visualize_sample(sample_idx=0)

    print(f"\n✅ 预览完成!")
    print(f"💡 你可以修改参数来查看更多样本:")
    print(f"   - preview_dataset(num_samples=10)  # 查看更多样本")
    print(f"   - visualize_sample(sample_idx=5)   # 可视化第6个样本")