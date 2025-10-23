import re
import json
from scipy.optimize import linear_sum_assignment
import numpy as np

def vision_reasoner_format_reward(predict_str: str) -> float:
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    match = re.fullmatch(pattern, predict_str, re.DOTALL)
    thinking_format_reward = 1.0 if match else 0.0 
    
    def segmentation_format(predict_str: str) -> float:
        segmentation_format_reward = 0.0
        try:
            json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
            if not json_match:
                return segmentation_format_reward
            data = json.loads(json_match.group(1))
            
            data_cnt = len(data)
            
            for item in data:
                cur_reward = 0.0

                if 'bbox_2d' in item:
                    bbox_2d = item['bbox_2d']
                    if isinstance(bbox_2d, list) and len(bbox_2d) == 4:
                        cur_reward += 1.0
                    
                if 'point_2d' in item:
                    point_2d = item['point_2d']
                    if isinstance(point_2d, list) and len(point_2d) == 2:
                        cur_reward += 1.0
                
                segmentation_format_reward += cur_reward / data_cnt
        except Exception:
            pass
        return segmentation_format_reward
        
    segmentation_format_reward = segmentation_format(predict_str)
    
    return thinking_format_reward + segmentation_format_reward

def vision_reasoner_accuracy_reward(predict_str: str, ground_truth: str, image_size: int = 280) -> float:
    """
    Calculate accuracy reward with coordinate normalization handling.

    Qwen3-VL outputs normalized coordinates (0-1000), while ground truth uses pixel coordinates.
    This function converts Qwen3-VL's normalized coords to pixels before comparison.

    Args:
        predict_str: Model prediction with <answer>...</answer> tags
        ground_truth: Ground truth JSON with pixel coordinates
        image_size: Image dimension (assuming square images, default 280x280)
    """
    max_accuracy_reward = 0.0
    MAX_OBJECTS = 120  # 设置上限

    try:
        gt_data = json.loads(ground_truth)

        # ==================== CRITICAL: Multi-image support ====================
        # For multi-image tasks, ground_truth may contain multiple bboxes from different slices
        # We only want to compare with the LARGEST tumor bbox (the target)
        # ========================================================================
        if len(gt_data) > 1:
            # Multiple bboxes: find the one with largest area
            max_area = 0
            largest_idx = 0
            for idx, item in enumerate(gt_data):
                bbox = item['bbox_2d']
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area > max_area:
                    max_area = area
                    largest_idx = idx
            # Keep only the largest bbox
            gt_data = [gt_data[largest_idx]]

        gt_bboxes = [item['bbox_2d'] for item in gt_data]
        gt_points = [item['point_2d'] for item in gt_data]

        #json_match = re.search(r'```json\s*(.*?)\s*```', predict_str, re.DOTALL)
        json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
            pred_bboxes = [item['bbox_2d'] for item in data]
            pred_points = [item['point_2d'] for item in data]

            # 只有当预测或真实值超过上限时才截断
            if len(pred_bboxes) > MAX_OBJECTS:
                pred_bboxes = pred_bboxes[:MAX_OBJECTS]
                pred_points = pred_points[:MAX_OBJECTS]

            if len(gt_bboxes) > MAX_OBJECTS:
                gt_bboxes = gt_bboxes[:MAX_OBJECTS]
                gt_points = gt_points[:MAX_OBJECTS]

            # 预处理数据为numpy数组
            pred_bboxes = np.array(pred_bboxes)  # (M,4)
            pred_points = np.array(pred_points)  # (M,2)
            gt_bboxes = np.array(gt_bboxes)    # (N,4)
            gt_points = np.array(gt_points)     # (N,2)

            # ==================== CRITICAL: Convert Qwen3-VL normalized coords to pixels ====================
            # Qwen3-VL uses normalized coordinates (0-1000), need to convert to pixel coords
            # Formula: pixel_coord = (normalized_coord / 1000.0) * image_size
            # Only convert if predictions are in normalized range (check if max coord > image_size)
            # ================================================================================================
            if len(pred_bboxes) > 0 and pred_bboxes.max() > image_size * 1.5:
                # Predictions are likely normalized (0-1000), convert to pixels
                pred_bboxes = (pred_bboxes / 1000.0 * image_size).astype(np.float32)
                pred_points = (pred_points / 1000.0 * image_size).astype(np.float32)

            # Ground truth is already in pixel coordinates, no conversion needed
            
            # 并行计算所有指标
            iou_matrix = batch_iou(pred_bboxes, gt_bboxes)  # (M,N)
            l1_matrix = batch_l1_distance(pred_bboxes, gt_bboxes)  # (M,N)
            points_dist_matrix = batch_points_distance(pred_points, gt_points)  # (M,N)
            points_in_box = batch_points_in_box(pred_points, pred_bboxes)  # (M,)
            
            # 计算reward矩阵
            iou_reward = (iou_matrix > 0.5).astype(float)
            bbox_l1_reward = (l1_matrix < 10).astype(float)
            point_reward = ((points_dist_matrix < 30) & points_in_box[:,np.newaxis]).astype(float)
            
            # 构建最终的cost矩阵
            cost_matrix = 3.0 - (iou_reward + bbox_l1_reward + point_reward)
            
            # 使用匈牙利算法找最优匹配
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # 直接从cost_matrix计算总reward
            total_reward = len(row_indices) * 3.0 - cost_matrix[row_indices, col_indices].sum()
            
            # 计算平均reward
            max_length = max(len(pred_bboxes), len(gt_bboxes))
            max_accuracy_reward = total_reward / max_length
            
    except Exception:
        pass
    return max_accuracy_reward

def vision_reasoner_non_repeat_reward(predict_str: str) -> float:
    non_repeat_reward = 1.0  # 初始满分
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
            if repeats >=2:
                non_repeat_reward = 0
                break
            seen.add(sentence)
            
    except Exception:
        pass
    
    return non_repeat_reward

def vision_reasoner_compute_score(predict_str: str, ground_truth: str, return_details: bool = False):
    """
    Compute vision reasoner reward score (UNCHANGED LOGIC)

    Components (max ~6.0):
    - format_reward: ~3.0 (thinking 1.0 + segmentation format ~2.0)
    - accuracy_reward: ~3.0 (bbox/point accuracy via Hungarian matching)
    - non_repeat_reward: 1.0 (penalize repetitive text)

    Args:
        predict_str: Model prediction string
        ground_truth: Ground truth JSON string
        return_details: If True, return (score, detailed_dict); else return score only

    Returns:
        float or (float, dict): Total score or (total_score, detailed_breakdown with 7 metrics)
    """
    # Original reward calculations (UNCHANGED)
    format_reward = vision_reasoner_format_reward(predict_str)
    accuracy_reward = vision_reasoner_accuracy_reward(predict_str, ground_truth)
    non_repeat_reward = vision_reasoner_non_repeat_reward(predict_str)

    total_reward = format_reward + accuracy_reward + non_repeat_reward

    if return_details:
        # Extract detailed breakdown for wandb monitoring
        details = {}

        # 1. Thinking tag (0 or 1.0)
        pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
        match = re.fullmatch(pattern, predict_str, re.DOTALL)
        details['thinking_tag'] = 1.0 if match else 0.0

        # 2-4. JSON format components
        details['json_parseable'] = 0.0
        details['bbox_format'] = 0.0
        details['point_format'] = 0.0

        try:
            json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
                details['json_parseable'] = 1.0

                if isinstance(data, list) and len(data) > 0:
                    # Check first item
                    item = data[0]
                    if 'bbox_2d' in item and isinstance(item['bbox_2d'], list) and len(item['bbox_2d']) == 4:
                        details['bbox_format'] = 1.0
                    if 'point_2d' in item and isinstance(item['point_2d'], list) and len(item['point_2d']) == 2:
                        details['point_format'] = 1.0
        except:
            pass

        # 5-6. Accuracy components (extract from accuracy_reward calculation)
        details['bbox_iou'] = 0.0
        details['point_distance'] = 0.0

        try:
            gt_data = json.loads(ground_truth)

            # Multi-image support: select largest bbox if multiple exist
            if len(gt_data) > 1:
                max_area = 0
                largest_idx = 0
                for idx, item in enumerate(gt_data):
                    bbox = item['bbox_2d']
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if area > max_area:
                        max_area = area
                        largest_idx = idx
                gt_data = [gt_data[largest_idx]]

            json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
            if json_match:
                pred_data = json.loads(json_match.group(1))
                if pred_data and gt_data:
                    pred_bbox = np.array([pred_data[0]['bbox_2d']])
                    gt_bbox = np.array([gt_data[0]['bbox_2d']])
                    pred_point = np.array([pred_data[0]['point_2d']])
                    gt_point = np.array([gt_data[0]['point_2d']])

                    # Convert Qwen3-VL normalized coords to pixels (same logic as accuracy_reward)
                    image_size = 280  # Default image size
                    if pred_bbox.max() > image_size * 1.5:
                        pred_bbox = (pred_bbox / 1000.0 * image_size).astype(np.float32)
                        pred_point = (pred_point / 1000.0 * image_size).astype(np.float32)

                    iou = batch_iou(pred_bbox, gt_bbox)[0,0]
                    details['bbox_iou'] = float(iou)

                    dist = batch_points_distance(pred_point, gt_point)[0,0]
                    details['point_distance'] = float(dist)
        except:
            pass

        # 7. Non-repeat (0 or 1.0)
        details['non_repeat'] = non_repeat_reward

        return total_reward, details
    else:
        return total_reward

def batch_iou(boxes1, boxes2):
    # boxes1: (M,4), boxes2: (N,4)
    # 广播机制自动扩展维度
    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)  # (M,1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)  # (N,1)
    
    xA = np.maximum(x11, np.transpose(x21))  # (M,N)
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    
    interArea = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)
    box1Area = (x12 - x11 + 1) * (y12 - y11 + 1)  # (M,1)
    box2Area = (x22 - x21 + 1) * (y22 - y21 + 1)  # (N,1)
    
    unionArea = box1Area + np.transpose(box2Area) - interArea
    iou = interArea / unionArea  # (M,N)
    return iou

def batch_l1_distance(boxes1, boxes2):
    # boxes1: (M,4), boxes2: (N,4)
    boxes1 = boxes1[:, np.newaxis, :]  # (M,1,4)
    boxes2 = boxes2[np.newaxis, :, :]  # (1,N,4)
    return np.mean(np.abs(boxes1 - boxes2), axis=2)  # (M,N)

def batch_points_distance(points1, points2):
    # points1: (M,2), points2: (N,2)
    points1 = points1[:, np.newaxis, :]  # (M,1,2)
    points2 = points2[np.newaxis, :, :]  # (1,N,2)
    
    # 计算欧氏距离
    dist = np.sqrt(np.sum((points1 - points2)**2, axis=2))  # (M,N)
    return dist

def batch_points_in_box(points, boxes):
    """
    检查每个点是否在对应的框内
    points: (M,2) - M个点的坐标
    boxes: (M,4) - M个框的坐标 [x1,y1,x2,y2]
    返回: (M,) 布尔数组
    """
    x_check = (points[:,0] >= boxes[:,0]) & (points[:,0] <= boxes[:,2])
    y_check = (points[:,1] >= boxes[:,1]) & (points[:,1] <= boxes[:,3])
    return x_check & y_check

if __name__ == "__main__":
    predict_str = """
<answer>
[{"bbox_2d": [10, 100, 398, 423], "point_2d": [283, 169]}]
</answer>
"""
    ground_truth = """
[{"bbox_2d": [416, 7, 833, 553], "point_2d": [648, 249]}]"""
    print(predict_str)
    print(ground_truth)
    print(vision_reasoner_compute_score(predict_str, ground_truth))

def vision_reasoner_smooth_existence_compute_score(
    predict_str: str,
    ground_truth: str,
    has_tumor: bool,
    image_size: int = 280,
    area_threshold: int = 100,
    sigmoid_scale: float = 50.0,
    return_details: bool = False
):
    """Smooth existence detection with threshold-based coordinate conversion (stable version)."""
    format_reward = vision_reasoner_format_reward(predict_str)
    existence_reward = 0.0
    localization_reward = 0.0
    pred_area = 0.0
    pred_iou = 0.0
    pred_point_dist = 0.0

    try:
        json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
        if json_match:
            pred_data = json.loads(json_match.group(1))
            if pred_data and len(pred_data) > 0:
                pred_bbox = pred_data[0]['bbox_2d']
                pred_bbox_arr = np.array(pred_bbox)
                
                # Threshold-based conversion (stable version logic)
                if pred_bbox_arr.max() > image_size * 1.5:
                    pred_bbox_arr = (pred_bbox_arr / 1000.0 * image_size).astype(np.float32)
                    pred_bbox = pred_bbox_arr.tolist()

                width = abs(pred_bbox[2] - pred_bbox[0])
                height = abs(pred_bbox[3] - pred_bbox[1])
                pred_area = width * height
                prob_has_tumor = 1.0 / (1.0 + np.exp(-(pred_area - area_threshold) / sigmoid_scale))

                if has_tumor:
                    existence_reward = prob_has_tumor * 3.0
                    if pred_area > area_threshold * 0.5:
                        gt_data = json.loads(ground_truth)

                        # Multi-image support: select largest bbox if multiple exist
                        if len(gt_data) > 1:
                            max_area = 0
                            largest_idx = 0
                            for idx, item in enumerate(gt_data):
                                bbox = item['bbox_2d']
                                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                                if area > max_area:
                                    max_area = area
                                    largest_idx = idx
                            gt_data = [gt_data[largest_idx]]

                        if gt_data and len(gt_data) > 0:
                            gt_bbox = np.array([gt_data[0]['bbox_2d']])
                            pred_bbox_for_iou = np.array([pred_bbox])
                            iou = batch_iou(pred_bbox_for_iou, gt_bbox)[0, 0]
                            pred_iou = float(iou)
                            localization_reward = pred_iou
                else:
                    existence_reward = (1.0 - prob_has_tumor) * 3.0
    except Exception:
        pass

    non_repeat_reward = vision_reasoner_non_repeat_reward(predict_str)
    total_reward = format_reward + existence_reward + localization_reward + non_repeat_reward

    if return_details:
        return total_reward, {'format': format_reward, 'existence': existence_reward, 
                             'localization': localization_reward, 'non_repeat': non_repeat_reward,
                             'pred_area': pred_area, 'bbox_iou': pred_iou, 'point_distance': pred_point_dist}
    return total_reward
