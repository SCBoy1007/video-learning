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
import os

# Global step counter for two-stage training
# Set via environment variable from training script
_GLOBAL_TRAINING_STEP = 0

def set_global_training_step(step: int):
    """Set current training step for curriculum learning"""
    global _GLOBAL_TRAINING_STEP
    _GLOBAL_TRAINING_STEP = step

def get_global_training_step() -> int:
    """Get current training step"""
    # Try environment variable first, fallback to global var
    env_step = os.environ.get('VERL_TRAINING_STEP', None)
    if env_step is not None:
        return int(env_step)
    return _GLOBAL_TRAINING_STEP


def brain_tumor_3d_thinking_format_reward(predict_str: str) -> float:
    """
    思维格式奖励 (1.0分) - 渐进式长度奖励
    检查是否包含<think>和<answer>标签，且<think>内容质量合格

    评分标准（基于长度）：
    - < 50 字符: 0.0 分（不合格，直接拒绝）
    - 50-100 字符: 0.3 分（基础，太简短）
    - 100-150 字符: 0.6 分（合格）
    - 150-200 字符: 0.8 分（良好）
    - 200+ 字符: 1.0 分（优秀，鼓励详细推理）

    额外检查：
    - <think> 内容不能以 JSON 格式开头（[ 或 {）
    """
    try:
        pattern = r"<think>(.*?)</think>\s*<answer>.*?</answer>"
        match = re.search(pattern, predict_str.strip(), re.DOTALL | re.IGNORECASE)

        if not match:
            return 0.0

        think_content = match.group(1).strip()
        content_length = len(think_content)

        # 检查1：不能以JSON格式开头
        if think_content.startswith('[') or think_content.startswith('{'):
            return 0.0

        # 检查2：基于长度的渐进式奖励
        if content_length < 50:
            return 0.0
        elif content_length < 100:
            return 0.3
        elif content_length < 150:
            return 0.6
        elif content_length < 200:
            return 0.8
        else:  # >= 200
            return 1.0

    except Exception:
        return 0.0


def brain_tumor_3d_video_keyword_reward(predict_str: str) -> float:
    """
    视频关键词奖励 (1.0分)
    强制要求<think>标签以"This video shows"开头，确保模型意识到输入是视频
    """
    try:
        # 提取<think>标签内容
        think_pattern = r'<think>(.*?)</think>'
        think_match = re.search(think_pattern, predict_str, re.DOTALL | re.IGNORECASE)

        if not think_match:
            return 0.0

        think_content = think_match.group(1).strip()

        # 检查是否以"This video shows"开头（不区分大小写）
        if think_content.lower().startswith('this video shows'):
            return 1.0

        return 0.0
    except Exception:
        return 0.0


def brain_tumor_3d_format_reward(predict_str: str) -> float:
    """
    检查输出格式是否符合要求 (1.0分)
    要求包含有效的JSON格式，包含bbox_2d, peak_slice, start_slice, end_slice, tumor_ratio字段
    优先从<answer>标签内提取JSON，如果没有标签则降级到全文搜索
    """
    try:
        # 优先从<answer>标签内提取JSON（模仿Seg-Zero）
        answer_pattern = r'<answer>(.*?)</answer>'
        answer_match = re.search(answer_pattern, predict_str, re.DOTALL | re.IGNORECASE)

        if answer_match:
            json_text = answer_match.group(1)
        else:
            # 如果没有<answer>标签，降级到全文搜索（但会损失thinking_format_reward）
            json_text = predict_str

        # 从文本中提取JSON
        json_data = extract_json_from_text(json_text)
        if not json_data:
            return 0.0

        # 检查必需字段
        required_fields = ['bbox_2d', 'peak_slice', 'start_slice', 'end_slice', 'tumor_ratio']
        for item in json_data:
            if not all(field in item for field in required_fields):
                return 0.0

            # 检查bbox_2d格式 (4个值: x1, y1, x2, y2)
            bbox = item['bbox_2d']
            if not isinstance(bbox, list) or len(bbox) != 4:
                return 0.0

            # 检查数值类型
            if not isinstance(item['peak_slice'], (int, float)):
                return 0.0
            if not isinstance(item['start_slice'], (int, float)):
                return 0.0
            if not isinstance(item['end_slice'], (int, float)):
                return 0.0
            if not isinstance(item['tumor_ratio'], (int, float)):
                return 0.0

        return 1.0

    except Exception:
        return 0.0


def brain_tumor_2d_bbox_iou_reward(predict_str: str, ground_truth: str) -> float:
    """
    2D边界框IoU奖励 (最高1.0分)

    在peak slice上计算2D bbox的IoU，相比3D bbox更容易学习

    平滑给分策略：
    - 有效2D框（面积>0）: 基础分 0.1
    - 与GT有任何重叠（IoU>0）: 额外 0.2
    - IoU线性映射: IoU × 0.7 (IoU=1.0时满分0.7)
    - 总分 = 0.1 + 0.2(if overlap) + IoU×0.7

    示例：
    - 无效框: 0.0
    - 有效但不重叠: 0.1
    - IoU=0.3: 0.51
    - IoU=0.7: 0.79
    - IoU=1.0: 1.0
    """
    try:
        # 解析ground truth，从bbox_3d提取2D坐标
        gt_data = json.loads(ground_truth)
        gt_bbox_3d = gt_data[0]['bbox_3d']
        # 提取xy坐标作为2D bbox: [x1, y1, x2, y2]
        gt_bbox_2d = [gt_bbox_3d[0], gt_bbox_3d[1], gt_bbox_3d[3], gt_bbox_3d[4]]

        # 优先从<answer>标签内提取JSON
        answer_pattern = r'<answer>(.*?)</answer>'
        answer_match = re.search(answer_pattern, predict_str, re.DOTALL | re.IGNORECASE)
        json_text = answer_match.group(1) if answer_match else predict_str

        # 解析预测结果
        pred_data = extract_json_from_text(json_text)
        if not pred_data:
            return 0.0

        pred_bbox_2d = pred_data[0]['bbox_2d']

        # 验证是否为有效2D框（4个数值，且面积>0）
        if not isinstance(pred_bbox_2d, list) or len(pred_bbox_2d) != 4:
            return 0.0

        try:
            pred_bbox_2d = [float(x) for x in pred_bbox_2d]
        except (ValueError, TypeError):
            return 0.0

        # 计算预测框面积（规范化坐标顺序）
        x1 = min(pred_bbox_2d[0], pred_bbox_2d[2])
        y1 = min(pred_bbox_2d[1], pred_bbox_2d[3])
        x2 = max(pred_bbox_2d[0], pred_bbox_2d[2])
        y2 = max(pred_bbox_2d[1], pred_bbox_2d[3])
        pred_area = (x2 - x1) * (y2 - y1)

        # 如果预测框无效（面积≤0），返回0
        if pred_area <= 0:
            return 0.0

        # 基础分：有效框就给0.1
        base_score = 0.1

        # 计算2D IoU
        iou = compute_2d_iou(pred_bbox_2d, gt_bbox_2d)

        # 重叠奖励：只要有任何重叠就额外给0.2
        overlap_bonus = 0.2 if iou > 0 else 0.0

        # IoU线性奖励：IoU越高分越高
        iou_score = iou * 0.7

        # 总分 = 基础分 + 重叠奖励 + IoU线性分
        total = base_score + overlap_bonus + iou_score

        return min(1.0, total)  # 确保不超过1.0

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

        # 优先从<answer>标签内提取JSON
        answer_pattern = r'<answer>(.*?)</answer>'
        answer_match = re.search(answer_pattern, predict_str, re.DOTALL | re.IGNORECASE)
        json_text = answer_match.group(1) if answer_match else predict_str

        # 解析预测结果
        pred_data = extract_json_from_text(json_text)
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


def brain_tumor_start_slice_reward(predict_str: str, ground_truth: str) -> float:
    """
    起始切片准确性奖励 (最高1.0分)
    从bbox_3d的z坐标提取start_slice (min(z1, z2))
    误差 ≤3: 1.0分, ≤5: 0.7分, ≤10: 0.3分, >10: 0.0分
    """
    try:
        # 解析ground truth，从bbox_3d提取start_slice
        gt_data = json.loads(ground_truth)
        gt_bbox_3d = gt_data[0]['bbox_3d']
        gt_start_slice = min(gt_bbox_3d[2], gt_bbox_3d[5])  # min(z1, z2)

        # 优先从<answer>标签内提取JSON
        answer_pattern = r'<answer>(.*?)</answer>'
        answer_match = re.search(answer_pattern, predict_str, re.DOTALL | re.IGNORECASE)
        json_text = answer_match.group(1) if answer_match else predict_str

        # 解析预测结果
        pred_data = extract_json_from_text(json_text)
        if not pred_data:
            return 0.0

        pred_start_slice = pred_data[0]['start_slice']

        # 计算误差
        error = abs(pred_start_slice - gt_start_slice)

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


def brain_tumor_end_slice_reward(predict_str: str, ground_truth: str) -> float:
    """
    结束切片准确性奖励 (最高1.0分)
    从bbox_3d的z坐标提取end_slice (max(z1, z2))
    误差 ≤3: 1.0分, ≤5: 0.7分, ≤10: 0.3分, >10: 0.0分
    """
    try:
        # 解析ground truth，从bbox_3d提取end_slice
        gt_data = json.loads(ground_truth)
        gt_bbox_3d = gt_data[0]['bbox_3d']
        gt_end_slice = max(gt_bbox_3d[2], gt_bbox_3d[5])  # max(z1, z2)

        # 优先从<answer>标签内提取JSON
        answer_pattern = r'<answer>(.*?)</answer>'
        answer_match = re.search(answer_pattern, predict_str, re.DOTALL | re.IGNORECASE)
        json_text = answer_match.group(1) if answer_match else predict_str

        # 解析预测结果
        pred_data = extract_json_from_text(json_text)
        if not pred_data:
            return 0.0

        pred_end_slice = pred_data[0]['end_slice']

        # 计算误差
        error = abs(pred_end_slice - gt_end_slice)

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

        # 优先从<answer>标签内提取JSON
        answer_pattern = r'<answer>(.*?)</answer>'
        answer_match = re.search(answer_pattern, predict_str, re.DOTALL | re.IGNORECASE)
        json_text = answer_match.group(1) if answer_match else predict_str

        # 解析预测结果
        pred_data = extract_json_from_text(json_text)
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


def brain_tumor_3d_non_repeat_reward(predict_str: str) -> float:
    """
    防重复奖励 (返回0-1标准化分数，外部乘权重)
    检查输出是否有重复句子，提高输出质量
    """
    non_repeat_reward = 1.0  # 初始满分（标准化分数）
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

    return non_repeat_reward  # 返回标准化分数，外部乘权重0.5


# Reward weights configuration (centralized for easy tuning)
REWARD_WEIGHTS = {
    'thinking_format': 0.5,  # 推理格式质量
    'video_keyword': 0.5,    # 视频分析门槛
    'format': 0.5,           # JSON格式验证
    'bbox_2d_iou': 1.0,      # 2D边界框IoU (peak slice上的2D框，更容易学习)
    'peak_slice': 0.5,       # 峰值切片准确性
    'start_slice': 0.5,      # 起始切片准确性
    'end_slice': 0.5,        # 结束切片准确性
    'tumor_ratio': 0.5,      # 肿瘤比例准确性
    # non_repeat removed - should be handled during sampling, not as reward
}


def get_reward_weights():
    """获取奖励权重配置，供其他模块使用"""
    return REWARD_WEIGHTS.copy()


def brain_tumor_3d_compute_score(predict_str: str, ground_truth: str, return_details: bool = False):
    """
    3D脑肿瘤检测总奖励函数 (最高4.5分) - 2D bbox + slice range版本

    **两阶段训练策略**：
    - Stage 1 (Steps 1-50): 只学习格式 (thinking + video_keyword + format)，医学指标全部mask为0
    - Stage 2 (Steps 51+): 全部指标生效

    组成（平衡权重 + 门槛机制）：
    - 思维格式奖励: 0.5分 (必须有50+字符且非JSON的推理内容)
    - 视频关键词奖励: 0.5分 (必须以"This video shows"开头，作为医学测量的门槛)
    - 格式奖励: 0.5分 (JSON格式验证: bbox_2d, peak_slice, start_slice, end_slice, tumor_ratio)
    - 2D bbox IoU奖励: 1.0分 (peak slice上的2D框，更容易学习) [Stage 2]
    - 峰值切片奖励: 0.5分 (肿瘤最大的切片索引) [Stage 2]
    - 起始切片奖励: 0.5分 (肿瘤起始的切片索引) [Stage 2]
    - 结束切片奖励: 0.5分 (肿瘤结束的切片索引) [Stage 2]
    - 肿瘤比例奖励: 0.5分 (体积比例) [Stage 2]

    **输出格式**：
    {
      "bbox_2d": [x1, y1, x2, y2],  // peak slice上的2D框
      "peak_slice": 74,              // 肿瘤最大的切片
      "start_slice": 55,             // 肿瘤起始切片
      "end_slice": 95,               // 肿瘤结束切片
      "tumor_ratio": 0.022           // 体积比例
    }

    **门槛机制**：
    如果 video_keyword = 0（未写"This video shows"），则所有医学测量奖励全部置0。
    逻辑：没有视频分析推理，医学测量结果就是无效的。

    Args:
        predict_str: 模型预测字符串
        ground_truth: Ground truth JSON字符串 (包含bbox_3d, peak_slice, tumor_ratio)
        return_details: 如果为True，返回(total_score, details_dict)，否则只返回total_score

    Returns:
        float or tuple: 总分或(总分, 详细字典)
    """
    # Get current training step for curriculum learning
    current_step = get_global_training_step()

    # Stage 1: Steps 1-50, only learn format (mask medical metrics)
    # Stage 2: Steps 51+, enable all metrics
    stage1_format_only = (current_step <= 50)

    # Use centralized weights
    thinking_format_reward = brain_tumor_3d_thinking_format_reward(predict_str) * REWARD_WEIGHTS['thinking_format']

    # Video keyword as a gate: if not satisfied, medical measurements are invalid
    video_keyword_raw = brain_tumor_3d_video_keyword_reward(predict_str)
    video_keyword_reward = video_keyword_raw * REWARD_WEIGHTS['video_keyword']

    format_reward = brain_tumor_3d_format_reward(predict_str) * REWARD_WEIGHTS['format']

    # Medical metrics: masked in Stage 1 (first 50 steps)
    if stage1_format_only or video_keyword_raw == 0:
        # Stage 1: Force all medical metrics to 0 to focus on format learning
        # OR video_keyword gate not satisfied
        bbox_2d_iou_reward = 0.0
        peak_slice_reward = 0.0
        start_slice_reward = 0.0
        end_slice_reward = 0.0
        ratio_reward = 0.0
    else:
        # Stage 2: Enable medical metrics
        bbox_2d_iou_reward = brain_tumor_2d_bbox_iou_reward(predict_str, ground_truth) * REWARD_WEIGHTS['bbox_2d_iou']
        peak_slice_reward = brain_tumor_3d_peak_slice_reward(predict_str, ground_truth) * REWARD_WEIGHTS['peak_slice']
        start_slice_reward = brain_tumor_start_slice_reward(predict_str, ground_truth) * REWARD_WEIGHTS['start_slice']
        end_slice_reward = brain_tumor_end_slice_reward(predict_str, ground_truth) * REWARD_WEIGHTS['end_slice']
        ratio_reward = brain_tumor_3d_ratio_reward(predict_str, ground_truth) * REWARD_WEIGHTS['tumor_ratio']

    total_reward = thinking_format_reward + video_keyword_reward + format_reward + bbox_2d_iou_reward + peak_slice_reward + start_slice_reward + end_slice_reward + ratio_reward

    if return_details:
        details = {
            'thinking_format': thinking_format_reward,
            'video_keyword': video_keyword_reward,
            'format': format_reward,
            'bbox_2d_iou': bbox_2d_iou_reward,
            'peak_slice': peak_slice_reward,
            'start_slice': start_slice_reward,
            'end_slice': end_slice_reward,
            'tumor_ratio': ratio_reward,
            'total': total_reward
        }
        return total_reward, details

    return total_reward


def extract_json_from_text(text: str):
    """
    从文本中提取JSON数据
    支持多种格式：列表格式、单对象格式、包含其他文字的格式
    """
    # 尝试多种JSON提取模式 (使用贪婪匹配来正确处理嵌套结构)
    patterns = [
        r'\[.*\]',  # 列表格式 [{"key": "value"}] - 贪婪匹配整个数组
        r'\{.*\}',  # 单个对象格式 {"key": "value"} - 贪婪匹配整个对象
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


def compute_2d_iou(box1, box2):
    """
    计算两个2D边界框的IoU
    box格式: [x1, y1, x2, y2]
    """
    try:
        # 确保输入是数值列表
        if len(box1) != 4 or len(box2) != 4:
            return 0.0

        box1 = [float(x) for x in box1]
        box2 = [float(x) for x in box2]

        # 确保bbox坐标顺序正确 (x1 < x2, y1 < y2)
        box1 = [min(box1[0], box1[2]), min(box1[1], box1[3]),
                max(box1[0], box1[2]), max(box1[1], box1[3])]
        box2 = [min(box2[0], box2[2]), min(box2[1], box2[3]),
                max(box2[0], box2[2]), max(box2[1], box2[3])]

        # 计算交集
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])

        # 检查是否有交集
        if x1_inter < x2_inter and y1_inter < y2_inter:
            inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        else:
            inter_area = 0.0

        # 计算两个box的面积
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # 避免除零错误
        union_area = area1 + area2 - inter_area
        if union_area <= 0:
            return 0.0

        iou = inter_area / union_area
        return max(0.0, min(1.0, iou))  # 确保IoU在[0,1]范围内

    except Exception:
        return 0.0


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
    # 测试代码 - 新格式: 2D bbox + slice range
    predict_str = """<think>This video shows an MRI sequence where tumor is located in right hemisphere, spanning slices 102-150</think><answer>[{"bbox_2d": [91, 33, 131, 84], "peak_slice": 124, "start_slice": 102, "end_slice": 150, "tumor_ratio": 0.022}]</answer>"""
    ground_truth = """[{"bbox_3d": [91, 33, 102, 131, 84, 150], "peak_slice": 124, "tumor_ratio": 0.021957}]"""

    score, details = brain_tumor_3d_compute_score(predict_str, ground_truth, return_details=True)
    print(f"Total score: {score}/4.5")

    # 测试各个组件
    print(f"\nDetailed breakdown:")
    print(f"Thinking format reward: {brain_tumor_3d_thinking_format_reward(predict_str)}")
    print(f"Video keyword reward: {brain_tumor_3d_video_keyword_reward(predict_str)}")
    print(f"Format reward: {brain_tumor_3d_format_reward(predict_str)}")
    print(f"2D bbox IoU reward: {brain_tumor_2d_bbox_iou_reward(predict_str, ground_truth)}")
    print(f"Peak slice reward: {brain_tumor_3d_peak_slice_reward(predict_str, ground_truth)}")
    print(f"Start slice reward: {brain_tumor_start_slice_reward(predict_str, ground_truth)}")
    print(f"End slice reward: {brain_tumor_end_slice_reward(predict_str, ground_truth)}")
    print(f"Ratio reward: {brain_tumor_3d_ratio_reward(predict_str, ground_truth)}")