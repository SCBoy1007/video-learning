#!/usr/bin/env python3
"""
修复数据集坐标系统不一致问题

将原始体素坐标转换为视频像素坐标，解决训练时坐标系不匹配的问题
- MSD: 240×240×155 → 252×252×155 (video: 1008×252)
- BraTS: 182×218×182 → 196×224×182 (video: 784×224)
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from datasets import load_from_disk
import numpy as np


class DatasetCoordinateFixer:
    def __init__(self, data_dir: str, backup: bool = True, validate: bool = True):
        self.data_dir = Path(data_dir)
        self.backup = backup
        self.validate = validate

        # 数据集配置
        self.dataset_configs = {
            'MSD_T1_MRI_video': {
                'original_shape': [240, 240, 155],
                'video_shape': [252, 252, 155],
                'video_size': [1008, 252],  # 4模态拼接
                'scale_factors': [252/240, 252/240, 1.0]
            },
            'BraTS_GLI_TrainingData_video': {
                'original_shape': [182, 218, 182],
                'video_shape': [196, 224, 182],
                'video_size': [784, 224],  # 4模态拼接
                'scale_factors': [196/182, 224/218, 1.0]
            },
            'BraTS_GLI_TrainingData_Additional_video': {
                'original_shape': [182, 218, 182],
                'video_shape': [196, 224, 182],
                'video_size': [784, 224],  # 4模态拼接
                'scale_factors': [196/182, 224/218, 1.0]
            }
        }

        self.conversion_stats = {}

    def backup_dataset(self, dataset_name: str) -> bool:
        """备份原始数据集"""
        original_path = self.data_dir / dataset_name
        backup_path = self.data_dir / f"{dataset_name}_backup"

        if not original_path.exists():
            print(f"❌ 数据集不存在: {original_path}")
            return False

        if backup_path.exists():
            print(f"⚠️  备份已存在: {backup_path}")
            return True

        try:
            print(f"📂 备份数据集: {original_path} → {backup_path}")
            shutil.copytree(original_path, backup_path)
            return True
        except Exception as e:
            print(f"❌ 备份失败: {e}")
            return False

    def convert_bbox_coordinates(self, bbox: List[float], scale_factors: List[float]) -> List[int]:
        """转换3D边界框坐标"""
        if len(bbox) != 6:
            raise ValueError(f"Invalid bbox format: {bbox}")

        # [x1, y1, z1, x2, y2, z2] 格式
        converted = [
            int(bbox[0] * scale_factors[0]),  # x1
            int(bbox[1] * scale_factors[1]),  # y1
            int(bbox[2] * scale_factors[2]),  # z1
            int(bbox[3] * scale_factors[0]),  # x2
            int(bbox[4] * scale_factors[1]),  # y2
            int(bbox[5] * scale_factors[2])   # z2
        ]

        return converted

    def convert_dataset_coordinates(self, dataset_name: str) -> bool:
        """转换单个数据集的坐标"""
        print(f"\n🔄 处理数据集: {dataset_name}")

        original_path = self.data_dir / dataset_name
        fixed_path = self.data_dir / f"{dataset_name}_fixed"

        if not original_path.exists():
            print(f"❌ 数据集不存在: {original_path}")
            return False

        # 获取配置
        config = self.dataset_configs[dataset_name]
        scale_factors = config['scale_factors']

        try:
            # 加载数据集
            dataset = load_from_disk(str(original_path / 'train'))
            print(f"  📊 加载样本数: {len(dataset)}")

            # 统计信息
            stats = {
                'total_samples': len(dataset),
                'converted_samples': 0,
                'errors': 0,
                'bbox_changes': [],
                'coordinate_ranges': {'before': {}, 'after': {}}
            }

            def convert_sample_coordinates(example):
                """转换单个样本的坐标"""
                try:
                    # 解析solution
                    solution = json.loads(example['solution'])
                    if not isinstance(solution, list) or len(solution) == 0:
                        raise ValueError("Invalid solution format")

                    original_bbox = solution[0]['bbox_3d'].copy()

                    # 转换bbox坐标
                    converted_bbox = self.convert_bbox_coordinates(original_bbox, scale_factors)
                    solution[0]['bbox_3d'] = converted_bbox

                    # 更新solution
                    example['solution'] = json.dumps(solution)

                    # 添加元数据
                    example['coordinate_system'] = 'video_pixels'
                    example['original_shape'] = str(config['original_shape'])
                    example['video_shape'] = str(config['video_shape'])
                    example['scale_factors'] = str(scale_factors)

                    # 记录统计信息
                    original_volume = (
                        (original_bbox[3] - original_bbox[0]) *
                        (original_bbox[4] - original_bbox[1]) *
                        (original_bbox[5] - original_bbox[2])
                    )
                    converted_volume = (
                        (converted_bbox[3] - converted_bbox[0]) *
                        (converted_bbox[4] - converted_bbox[1]) *
                        (converted_bbox[5] - converted_bbox[2])
                    )

                    volume_change = converted_volume / original_volume if original_volume > 0 else 1.0

                    bbox_change = {
                        'original': original_bbox,
                        'converted': converted_bbox,
                        'volume_change': volume_change
                    }
                    stats['bbox_changes'].append(bbox_change)
                    stats['converted_samples'] += 1

                    return example

                except Exception as e:
                    print(f"    ❌ 转换样本失败: {e}")
                    stats['errors'] += 1
                    return example

            # 应用坐标转换
            print(f"  🔄 转换坐标系统...")
            dataset_converted = dataset.map(convert_sample_coordinates)

            # 保存转换后的数据集
            print(f"  💾 保存到: {fixed_path}")
            dataset_converted.save_to_disk(str(fixed_path))

            # 保存统计信息
            self.conversion_stats[dataset_name] = stats

            print(f"  ✅ 转换完成: {stats['converted_samples']}/{stats['total_samples']} 成功")
            if stats['errors'] > 0:
                print(f"    ⚠️  错误数: {stats['errors']}")

            return True

        except Exception as e:
            print(f"❌ 处理数据集失败: {e}")
            return False

    def validate_conversion(self, dataset_name: str) -> bool:
        """验证转换结果"""
        print(f"\n🔍 验证数据集: {dataset_name}")

        fixed_path = self.data_dir / f"{dataset_name}_fixed"
        if not fixed_path.exists():
            print(f"❌ 转换后数据集不存在: {fixed_path}")
            return False

        try:
            dataset = load_from_disk(str(fixed_path / 'train'))
            config = self.dataset_configs[dataset_name]

            # 验证样本
            sample = dataset[0]
            solution = json.loads(sample['solution'])
            bbox = solution[0]['bbox_3d']

            # 检查坐标范围
            video_w, video_h = config['video_size']
            single_modal_w = video_w // 4  # 单模态宽度
            single_modal_h = video_h

            x_valid = 0 <= bbox[0] < single_modal_w and 0 <= bbox[3] < single_modal_w
            y_valid = 0 <= bbox[1] < single_modal_h and 0 <= bbox[4] < single_modal_h
            z_valid = 0 <= bbox[2] < config['original_shape'][2] and 0 <= bbox[5] < config['original_shape'][2]

            print(f"  📏 坐标范围验证:")
            print(f"    X坐标: {'✅' if x_valid else '❌'} (应在0-{single_modal_w})")
            print(f"    Y坐标: {'✅' if y_valid else '❌'} (应在0-{single_modal_h})")
            print(f"    Z坐标: {'✅' if z_valid else '❌'} (应在0-{config['original_shape'][2]})")

            # 验证元数据
            has_metadata = all(key in sample for key in ['coordinate_system', 'original_shape', 'video_shape'])
            print(f"  📋 元数据: {'✅' if has_metadata else '❌'}")

            if has_metadata:
                print(f"    坐标系统: {sample['coordinate_system']}")
                print(f"    原始形状: {sample['original_shape']}")
                print(f"    视频形状: {sample['video_shape']}")

            return x_valid and y_valid and z_valid and has_metadata

        except Exception as e:
            print(f"❌ 验证失败: {e}")
            return False

    def print_conversion_statistics(self):
        """打印转换统计信息"""
        print(f"\n📊 转换统计报告")
        print("=" * 60)

        for dataset_name, stats in self.conversion_stats.items():
            print(f"\n📁 {dataset_name}:")
            print(f"  总样本数: {stats['total_samples']}")
            print(f"  成功转换: {stats['converted_samples']}")
            print(f"  转换错误: {stats['errors']}")

            if len(stats['bbox_changes']) > 0:
                volume_changes = [change['volume_change'] for change in stats['bbox_changes'][:5]]
                avg_volume_change = np.mean(volume_changes)
                print(f"  平均体积变化: {avg_volume_change:.3f}x")

                # 显示几个转换示例
                print(f"  转换示例:")
                for i, change in enumerate(stats['bbox_changes'][:3]):
                    orig = change['original']
                    conv = change['converted']
                    print(f"    [{i+1}] {orig} → {conv}")

    def replace_original_datasets(self):
        """用修复后的数据集替换原始数据集"""
        print(f"\n🔄 替换原始数据集...")

        for dataset_name in self.dataset_configs.keys():
            original_path = self.data_dir / dataset_name
            fixed_path = self.data_dir / f"{dataset_name}_fixed"

            if not fixed_path.exists():
                print(f"⚠️  跳过 {dataset_name}: 修复版本不存在")
                continue

            try:
                # 删除原始版本
                if original_path.exists():
                    shutil.rmtree(original_path)

                # 重命名修复版本
                shutil.move(str(fixed_path), str(original_path))
                print(f"✅ {dataset_name}: 替换成功")

            except Exception as e:
                print(f"❌ {dataset_name}: 替换失败 - {e}")

    def run(self):
        """执行完整的坐标修复流程"""
        print("🚀 开始修复数据集坐标系统")
        print("=" * 60)

        success_count = 0
        total_datasets = len(self.dataset_configs)

        for dataset_name in self.dataset_configs.keys():
            print(f"\n📂 处理数据集 {dataset_name}")

            # 1. 备份原始数据
            if self.backup:
                if not self.backup_dataset(dataset_name):
                    continue

            # 2. 转换坐标
            if not self.convert_dataset_coordinates(dataset_name):
                continue

            # 3. 验证转换结果
            if self.validate:
                if self.validate_conversion(dataset_name):
                    success_count += 1
                    print(f"✅ {dataset_name}: 处理成功")
                else:
                    print(f"❌ {dataset_name}: 验证失败")
            else:
                success_count += 1

        # 4. 打印统计信息
        self.print_conversion_statistics()

        # 5. 总结
        print(f"\n🎉 处理完成: {success_count}/{total_datasets} 数据集成功")

        if success_count == total_datasets:
            try:
                replace = input("\n是否用修复后的数据集替换原始数据集? (y/N): ").strip().lower()
                if replace == 'y':
                    self.replace_original_datasets()
                    print("✅ 所有数据集已更新为修复版本")
                else:
                    print("ℹ️  修复后的数据集保存为 *_fixed 版本")
            except EOFError:
                print("\nℹ️  修复后的数据集保存为 *_fixed 版本")


def main():
    parser = argparse.ArgumentParser(description="修复数据集坐标系统")
    parser.add_argument("--data-dir", default=".", help="数据集目录路径")
    parser.add_argument("--no-backup", action="store_true", help="跳过备份步骤")
    parser.add_argument("--no-validate", action="store_true", help="跳过验证步骤")
    parser.add_argument("--auto-replace", action="store_true", help="自动替换原始数据集")

    args = parser.parse_args()

    # 创建修复器
    fixer = DatasetCoordinateFixer(
        data_dir=args.data_dir,
        backup=not args.no_backup,
        validate=not args.no_validate
    )

    # 执行修复
    fixer.run()


if __name__ == "__main__":
    main()