# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import json
import numpy as np


def brain_tumor_3d_format_reward(predict_str: str) -> float:
    """
    检查输出格式是否符合要求 (1.0分)
    要求包含有效的JSON格式，包含bbox_3d, peak_slice, tumor_ratio字段
    """
    try:
        # 从文本中提取JSON
        json_data = extract_json_from_text(predict_str)
        if not json_data:
            return 0.0

        # 检查必需字段
        required_fields = ['bbox_3d', 'peak_slice', 'tumor_ratio']
        for item in json_data:
            if not all(field in item for field in required_fields):
                return 0.0

            # 检查bbox_3d格式
            bbox = item['bbox_3d']
            if not isinstance(bbox, list) or len(bbox) != 6:
                return 0.0

            # 检查数值类型
            if not isinstance(item['peak_slice'], (int, float)):
                return 0.0
            if not isinstance(item['tumor_ratio'], (int, float)):
                return 0.0

        return 1.0

    except Exception:
        return 0.0


def brain_tumor_3d_iou_reward(predict_str: str, ground_truth: str) -> float:
    """
    3D边界框IoU奖励 (最高1.5分)
    IoU > 0.7: 1.5分, IoU > 0.5: 1.0分, IoU > 0.3: 0.5分, 其他: 0.0分
    """
    try:
        # 解析ground truth
        gt_data = json.loads(ground_truth)
        gt_bbox = gt_data[0]['bbox_3d']

        # 解析预测结果
        pred_data = extract_json_from_text(predict_str)
        if not pred_data:
            return 0.0

        pred_bbox = pred_data[0]['bbox_3d']

        # 计算3D IoU
        iou = compute_3d_iou(pred_bbox, gt_bbox)

        # 分级奖励
        if iou > 0.7:
            return 1.5
        elif iou > 0.5:
            return 1.0
        elif iou > 0.3:
            return 0.5
        else:
            return 0.0

    except Exception:
        return 0.0


def brain_tumor_3d_peak_slice_reward(predict_str: str, ground_truth: str) -> float:
    """
    峰值切片准确性奖励 (最高1.0分)
    误差 ≤3: 1.0分, ≤5: 0.7分, ≤10: 0.3分, >10: 0.0分
    """
    try:
        # 解析ground truth
        gt_data = json.loads(ground_truth)
        gt_slice = gt_data[0]['peak_slice']

        # 解析预测结果
        pred_data = extract_json_from_text(predict_str)
        if not pred_data:
            return 0.0

        pred_slice = pred_data[0]['peak_slice']

        # 计算误差
        error = abs(pred_slice - gt_slice)

        # 分级奖励
        if error <= 3:
            return 1.0
        elif error <= 5:
            return 0.7
        elif error <= 10:
            return 0.3
        else:
            return 0.0

    except Exception:
        return 0.0


def brain_tumor_3d_ratio_reward(predict_str: str, ground_truth: str) -> float:
    """
    肿瘤比例准确性奖励 (最高1.0分)
    相对误差 ≤10%: 1.0分, ≤20%: 0.7分, ≤30%: 0.3分, >30%: 0.0分
    """
    try:
        # 解析ground truth
        gt_data = json.loads(ground_truth)
        gt_ratio = gt_data[0]['tumor_ratio']

        # 解析预测结果
        pred_data = extract_json_from_text(predict_str)
        if not pred_data:
            return 0.0

        pred_ratio = pred_data[0]['tumor_ratio']

        # 处理特殊情况：ground truth为0
        if gt_ratio == 0:
            return 1.0 if pred_ratio == 0 else 0.0

        # 计算相对误差
        relative_error = abs(pred_ratio - gt_ratio) / gt_ratio

        # 分级奖励
        if relative_error <= 0.1:  # 10%
            return 1.0
        elif relative_error <= 0.2:  # 20%
            return 0.7
        elif relative_error <= 0.3:  # 30%
            return 0.3
        else:
            return 0.0

    except Exception:
        return 0.0


def brain_tumor_3d_completeness_reward(predict_str: str) -> float:
    """
    完整性奖励 (0.5分)
    检查是否包含所有必需字段
    """
    try:
        json_data = extract_json_from_text(predict_str)
        if not json_data:
            return 0.0

        required_fields = ['bbox_3d', 'peak_slice', 'tumor_ratio']

        for item in json_data:
            # 检查所有必需字段是否存在且非空
            for field in required_fields:
                if field not in item or item[field] is None:
                    return 0.0

            # 检查bbox_3d的长度
            if len(item['bbox_3d']) != 6:
                return 0.0

        return 0.5

    except Exception:
        return 0.0


def brain_tumor_3d_non_repeat_reward(predict_str: str) -> float:
    """
    防重复奖励 (0.5分)
    检查输出是否有重复句子，提高输出质量
    """
    non_repeat_reward = 0.5  # 初始满分
    try:
        sentences = predict_str.split('.')

        # 移除空句子
        sentences = [s.strip() for s in sentences if s.strip()]

        # 检查重复
        seen = set()
        repeats = 0

        for sentence in sentences:
            if sentence in seen:
                repeats += 1
            if repeats >= 2:
                non_repeat_reward = 0.0
                break
            seen.add(sentence)

    except Exception:
        pass

    return non_repeat_reward


def brain_tumor_3d_compute_score(predict_str: str, ground_truth: str) -> float:
    """
    3D脑肿瘤检测总奖励函数 (最高5.5分)

    组成：
    - 格式奖励: 1.0分
    - 3D IoU奖励: 1.5分
    - 峰值切片奖励: 1.0分
    - 肿瘤比例奖励: 1.0分
    - 完整性奖励: 0.5分
    - 防重复奖励: 0.5分
    """
    format_reward = brain_tumor_3d_format_reward(predict_str)
    iou_reward = brain_tumor_3d_iou_reward(predict_str, ground_truth)
    peak_slice_reward = brain_tumor_3d_peak_slice_reward(predict_str, ground_truth)
    ratio_reward = brain_tumor_3d_ratio_reward(predict_str, ground_truth)
    completeness_reward = brain_tumor_3d_completeness_reward(predict_str)
    non_repeat_reward = brain_tumor_3d_non_repeat_reward(predict_str)

    total_reward = format_reward + iou_reward + peak_slice_reward + ratio_reward + completeness_reward + non_repeat_reward

    return total_reward


def extract_json_from_text(text: str):
    """
    从文本中提取JSON数据
    支持多种格式：列表格式、单对象格式、包含其他文字的格式
    """
    # 尝试多种JSON提取模式
    patterns = [
        r'\[\s*\{.*?\}\s*\]',  # 列表格式 [{"key": "value"}]
        r'\{[^{}]*\}',         # 单个对象格式 {"key": "value"}
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                # 处理单引号问题
                json_str = match.replace("'", '"')
                data = json.loads(json_str)

                # 确保返回列表格式
                if isinstance(data, dict):
                    data = [data]
                elif not isinstance(data, list):
                    continue

                # 验证数据结构
                if len(data) > 0 and isinstance(data[0], dict):
                    return data

            except (json.JSONDecodeError, ValueError):
                continue

    return None


def compute_3d_iou(box1, box2):
    """
    计算两个3D边界框的IoU
    box格式: [x1, y1, z1, x2, y2, z2]
    """
    try:
        # 确保输入是数值列表
        if len(box1) != 6 or len(box2) != 6:
            return 0.0

        box1 = [float(x) for x in box1]
        box2 = [float(x) for x in box2]

        # 确保bbox坐标顺序正确 (x1 < x2, y1 < y2, z1 < z2)
        box1 = [min(box1[0], box1[3]), min(box1[1], box1[4]), min(box1[2], box1[5]),
                max(box1[0], box1[3]), max(box1[1], box1[4]), max(box1[2], box1[5])]
        box2 = [min(box2[0], box2[3]), min(box2[1], box2[4]), min(box2[2], box2[5]),
                max(box2[0], box2[3]), max(box2[1], box2[4]), max(box2[2], box2[5])]

        # 计算交集
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        z1_inter = max(box1[2], box2[2])
        x2_inter = min(box1[3], box2[3])
        y2_inter = min(box1[4], box2[4])
        z2_inter = min(box1[5], box2[5])

        # 检查是否有交集
        if x1_inter < x2_inter and y1_inter < y2_inter and z1_inter < z2_inter:
            inter_volume = (x2_inter - x1_inter) * (y2_inter - y1_inter) * (z2_inter - z1_inter)
        else:
            inter_volume = 0.0

        # 计算两个box的体积
        volume1 = (box1[3] - box1[0]) * (box1[4] - box1[1]) * (box1[5] - box1[2])
        volume2 = (box2[3] - box2[0]) * (box2[4] - box2[1]) * (box2[5] - box2[2])

        # 避免除零错误
        union_volume = volume1 + volume2 - inter_volume
        if union_volume <= 0:
            return 0.0

        iou = inter_volume / union_volume
        return max(0.0, min(1.0, iou))  # 确保IoU在[0,1]范围内

    except Exception:
        return 0.0


if __name__ == "__main__":
    # 测试代码
    predict_str = """[{"bbox_3d": [91, 33, 102, 131, 84, 150], "peak_slice": 124, "tumor_ratio": 0.022}]"""
    ground_truth = """[{"bbox_3d": [91, 33, 102, 131, 84, 150], "peak_slice": 124, "tumor_ratio": 0.021957}]"""

    score = brain_tumor_3d_compute_score(predict_str, ground_truth)
    print(f"Total score: {score}/5.5")

    # 测试各个组件
    print(f"Format reward: {brain_tumor_3d_format_reward(predict_str)}")
    print(f"IoU reward: {brain_tumor_3d_iou_reward(predict_str, ground_truth)}")
    print(f"Peak slice reward: {brain_tumor_3d_peak_slice_reward(predict_str, ground_truth)}")
    print(f"Ratio reward: {brain_tumor_3d_ratio_reward(predict_str, ground_truth)}")
    print(f"Completeness reward: {brain_tumor_3d_completeness_reward(predict_str)}")
    print(f"Non-repeat reward: {brain_tumor_3d_non_repeat_reward(predict_str)}")