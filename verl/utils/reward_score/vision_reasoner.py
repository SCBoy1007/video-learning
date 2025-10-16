import re
import json
from scipy.optimize import linear_sum_assignment
import numpy as np

# ==================== TUMOR EXISTENCE DETECTION (SIMPLIFIED) ====================
# Task: Only detect whether tumor exists (yes/no), NO localization required
# Expected format: <think>...</think><answer>yes</answer> or <answer>no</answer>
# ==================================================================================

def vision_reasoner_format_reward(predict_str: str) -> float:
    """
    Format reward for tumor existence detection (1 point)

    Checks for:
    - <think>...</think><answer>...</answer> structure

    Returns:
        1.0 if format is correct, 0.0 otherwise
    """
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    match = re.fullmatch(pattern, predict_str, re.DOTALL)
    return 1.0 if match else 0.0

# ==================== LOCALIZATION FUNCTIONS (COMMENTED OUT) ====================
# These functions are for bbox/point localization, which we'll add back later
# For now, we only focus on tumor existence detection
# ===============================================================================

# def segmentation_format(predict_str: str) -> float:
#     """Check if output has valid bbox_2d and point_2d format"""
#     segmentation_format_reward = 0.0
#     try:
#         json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
#         if not json_match:
#             return segmentation_format_reward
#         data = json.loads(json_match.group(1))
#
#         data_cnt = len(data)
#         if data_cnt == 0:
#             return 0.0
#
#         for item in data:
#             cur_reward = 0.0
#             if 'bbox_2d' in item:
#                 bbox_2d = item['bbox_2d']
#                 if isinstance(bbox_2d, list) and len(bbox_2d) == 4:
#                     cur_reward += 1.0
#             if 'point_2d' in item:
#                 point_2d = item['point_2d']
#                 if isinstance(point_2d, list) and len(point_2d) == 2:
#                     cur_reward += 1.0
#             segmentation_format_reward += cur_reward / data_cnt
#     except Exception:
#         pass
#     return segmentation_format_reward

def check_model_prediction(predict_str: str) -> bool:
    """
    Extract model's tumor existence prediction from answer tag

    Returns:
        True if model predicts tumor exists, False otherwise
    """
    try:
        # Extract content from <answer>...</answer>
        json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
        if not json_match:
            return False

        answer_content = json_match.group(1).strip().lower()

        # Check for "yes" or "no"
        if "yes" in answer_content:
            return True
        elif "no" in answer_content:
            return False
        else:
            # Fallback: empty or invalid means no tumor
            return False

    except Exception:
        return False


def vision_reasoner_existence_accuracy_reward(predict_str: str, has_tumor: bool) -> float:
    """
    Reward for tumor existence detection accuracy (3 points)

    Args:
        predict_str: Model's prediction
        has_tumor: Ground truth (True if tumor exists)

    Returns:
        3.0 if correct, 0.0 if incorrect
    """
    predicted_has_tumor = check_model_prediction(predict_str)

    # Correct prediction: 3 points
    # Incorrect prediction: 0 points
    if predicted_has_tumor == has_tumor:
        return 3.0
    else:
        return 0.0


# ==================== LOCALIZATION ACCURACY (COMMENTED OUT) ====================
# This function computes bbox/point localization accuracy using Hungarian matching
# We'll uncomment and use this when adding localization task back
# ===============================================================================

# def vision_reasoner_localization_accuracy_reward(predict_str: str, ground_truth: str) -> float:
#     """Compute localization accuracy using IoU and point distance"""
#     max_accuracy_reward = 0.0
#     MAX_OBJECTS = 120
#
#     try:
#         gt_data = json.loads(ground_truth)
#         gt_bboxes = [item['bbox_2d'] for item in gt_data]
#         gt_points = [item['point_2d'] for item in gt_data]
#
#         json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
#         if json_match:
#             data = json.loads(json_match.group(1))
#             pred_bboxes = [item['bbox_2d'] for item in data]
#             pred_points = [item['point_2d'] for item in data]
#
#             if len(pred_bboxes) > MAX_OBJECTS:
#                 pred_bboxes = pred_bboxes[:MAX_OBJECTS]
#                 pred_points = pred_points[:MAX_OBJECTS]
#
#             if len(gt_bboxes) > MAX_OBJECTS:
#                 gt_bboxes = gt_bboxes[:MAX_OBJECTS]
#                 gt_points = gt_points[:MAX_OBJECTS]
#
#             pred_bboxes = np.array(pred_bboxes)
#             pred_points = np.array(pred_points)
#             gt_bboxes = np.array(gt_bboxes)
#             gt_points = np.array(gt_points)
#
#             iou_matrix = batch_iou(pred_bboxes, gt_bboxes)
#             l1_matrix = batch_l1_distance(pred_bboxes, gt_bboxes)
#             points_dist_matrix = batch_points_distance(pred_points, gt_points)
#             points_in_box = batch_points_in_box(pred_points, pred_bboxes)
#
#             iou_reward = (iou_matrix > 0.5).astype(float)
#             bbox_l1_reward = (l1_matrix < 10).astype(float)
#             point_reward = ((points_dist_matrix < 30) & points_in_box[:,np.newaxis]).astype(float)
#
#             cost_matrix = 3.0 - (iou_reward + bbox_l1_reward + point_reward)
#             row_indices, col_indices = linear_sum_assignment(cost_matrix)
#
#             total_reward = len(row_indices) * 3.0 - cost_matrix[row_indices, col_indices].sum()
#             max_length = max(len(pred_bboxes), len(gt_bboxes))
#             max_accuracy_reward = total_reward / max_length
#
#     except Exception:
#         pass
#     return max_accuracy_reward

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

def vision_reasoner_compute_score(predict_str: str, has_tumor: bool, return_details: bool = False):
    """
    Compute tumor existence detection reward score (SIMPLIFIED FOR EXISTENCE DETECTION ONLY)

    Components (max 5.0):
    - format_reward: 1.0 (has <think>...</think><answer>...</answer> structure)
    - existence_accuracy_reward: 3.0 (correct yes/no prediction)
    - non_repeat_reward: 1.0 (penalize repetitive text)

    Args:
        predict_str: Model prediction string
        has_tumor: Ground truth tumor existence (bool)
        return_details: If True, return (score, detailed_dict); else return score only

    Returns:
        float or (float, dict): Total score or (total_score, detailed_breakdown with 3 metrics)
    """
    # New simplified reward calculation
    format_reward = vision_reasoner_format_reward(predict_str)
    existence_reward = vision_reasoner_existence_accuracy_reward(predict_str, has_tumor)
    non_repeat_reward = vision_reasoner_non_repeat_reward(predict_str)

    total_reward = format_reward + existence_reward + non_repeat_reward

    if return_details:
        # Simplified detailed breakdown for wandb monitoring
        details = {}

        # 1. Format (thinking tag): 0 or 1.0
        details['format'] = format_reward

        # 2. Existence accuracy: 0 or 3.0
        details['existence_accuracy'] = existence_reward

        # 3. Non-repeat: 0 or 1.0
        details['non_repeat'] = non_repeat_reward

        # 4. Prediction correctness (for debugging)
        predicted_has_tumor = check_model_prediction(predict_str)
        details['predicted_has_tumor'] = 1.0 if predicted_has_tumor else 0.0
        details['ground_truth_has_tumor'] = 1.0 if has_tumor else 0.0

        return total_reward, details
    else:
        return total_reward


# ==================== DETAILED METRICS (COMMENTED OUT) ====================
# These detailed metrics are for localization task, which we'll add back later
# For now, we only track existence detection metrics
# ===========================================================================

# if return_details:
#     details = {}
#     pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
#     match = re.fullmatch(pattern, predict_str, re.DOTALL)
#     details['thinking_tag'] = 1.0 if match else 0.0
#
#     details['json_parseable'] = 0.0
#     details['bbox_format'] = 0.0
#     details['point_format'] = 0.0
#     try:
#         json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
#         if json_match:
#             data = json.loads(json_match.group(1))
#             details['json_parseable'] = 1.0
#             if isinstance(data, list) and len(data) > 0:
#                 item = data[0]
#                 if 'bbox_2d' in item and isinstance(item['bbox_2d'], list) and len(item['bbox_2d']) == 4:
#                     details['bbox_format'] = 1.0
#                 if 'point_2d' in item and isinstance(item['point_2d'], list) and len(item['point_2d']) == 2:
#                     details['point_format'] = 1.0
#     except:
#         pass
#
#     details['bbox_iou'] = 0.0
#     details['point_distance'] = 0.0
#     try:
#         gt_data = json.loads(ground_truth)
#         json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
#         if json_match:
#             pred_data = json.loads(json_match.group(1))
#             if pred_data and gt_data:
#                 pred_bbox = np.array([pred_data[0]['bbox_2d']])
#                 gt_bbox = np.array([gt_data[0]['bbox_2d']])
#                 pred_point = np.array([pred_data[0]['point_2d']])
#                 gt_point = np.array([gt_data[0]['point_2d']])
#                 iou = batch_iou(pred_bbox, gt_bbox)[0,0]
#                 details['bbox_iou'] = float(iou)
#                 dist = batch_points_distance(pred_point, gt_point)[0,0]
#                 details['point_distance'] = float(dist)
#     except:
#         pass
#     details['non_repeat'] = non_repeat_reward
#     return total_reward, details

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
    # Test case 1: Correct positive prediction
    predict_str_yes = """<think>I can see abnormal bright regions in the image that appear to be tumor tissue.</think>
<answer>yes</answer>"""

    # Test case 2: Correct negative prediction
    predict_str_no = """<think>The brain tissue appears normal with no visible tumor regions.</think>
<answer>no</answer>"""

    # Test case 3: Wrong prediction
    predict_str_wrong = """<think>Analyzing the image...</think>
<answer>yes</answer>"""

    print("Test 1 - Correct Positive (has_tumor=True, predict=yes):")
    score1, details1 = vision_reasoner_compute_score(predict_str_yes, has_tumor=True, return_details=True)
    print(f"  Score: {score1}/5.0")
    print(f"  Details: {details1}\n")

    print("Test 2 - Correct Negative (has_tumor=False, predict=no):")
    score2, details2 = vision_reasoner_compute_score(predict_str_no, has_tumor=False, return_details=True)
    print(f"  Score: {score2}/5.0")
    print(f"  Details: {details2}\n")

    print("Test 3 - Wrong Prediction (has_tumor=False, predict=yes):")
    score3, details3 = vision_reasoner_compute_score(predict_str_wrong, has_tumor=False, return_details=True)
    print(f"  Score: {score3}/5.0")
    print(f"  Details: {details3}")
    