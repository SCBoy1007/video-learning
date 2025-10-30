#!/usr/bin/env python3
"""
Brain Tumor Detection Evaluation Script

Evaluates bbox localization performance on brain tumor MRI datasets.
Uses the same prompt format as training to ensure consistency.
No SAM segmentation - only bbox IoU evaluation.

Usage:
    python evaluation_scripts/eval_brain_tumor.py \
        --model_path models/Qwen3-VL-8B-Thinking \
        --data_path data/BraTS_GLI_Additional_Image_280/T1C \
        --output_path eval_results/baseline_T1C \
        --batch_size 8
"""

import argparse
import json
import os
import re
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import torch
from datasets import load_from_disk
from PIL import Image as PILImage
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate brain tumor detection model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model (base model or checkpoint)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to evaluation dataset (e.g., data/BraTS_GLI_Additional_Image_280/T1C)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for inference")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum tokens to generate")
    parser.add_argument("--image_size", type=int, default=280,
                        help="Image size (280 or 840)")
    parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.",
                        help="System prompt")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to evaluate (for quick testing)")
    parser.add_argument("--num_images", type=int, default=8,
                        help="Number of images to use per sample (1=single-image, 8=multi-image)")
    return parser.parse_args()


def extract_bbox_and_point(output_text: str, image_size: int = 280) -> Tuple[List, List]:
    """
    Extract bbox and point from model output.
    Handles Qwen3-VL's normalized coordinates (0-1000) and converts to pixels.

    Returns:
        pred_bboxes: List of [x1, y1, x2, y2] in pixel coordinates
        pred_points: List of [x, y] in pixel coordinates
    """
    try:
        json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', output_text, re.DOTALL)
        if not json_match:
            return [], []

        data = json.loads(json_match.group(1))

        pred_bboxes = [item['bbox_2d'] for item in data if 'bbox_2d' in item]
        pred_points = [item['point_2d'] for item in data if 'point_2d' in item]

        # Convert to numpy for easier processing
        pred_bboxes = np.array(pred_bboxes, dtype=np.float32)  # (N, 4)
        pred_points = np.array(pred_points, dtype=np.float32)  # (N, 2)

        # Convert Qwen3-VL normalized coords (0-1000) to pixels if needed
        if len(pred_bboxes) > 0 and pred_bboxes.max() > image_size * 1.5:
            pred_bboxes = (pred_bboxes / 1000.0 * image_size).astype(np.float32)
            pred_points = (pred_points / 1000.0 * image_size).astype(np.float32)

        return pred_bboxes.tolist(), pred_points.tolist()

    except Exception as e:
        print(f"Error parsing output: {e}")
        print(f"Output text: {output_text[:200]}...")
        return [], []


def compute_bbox_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """Compute IoU between two bboxes [x1, y1, x2, y2]"""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def compute_point_distance(point1: List[float], point2: List[float]) -> float:
    """Compute Euclidean distance between two points [x, y]"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def check_format(output_text: str) -> Dict[str, float]:
    """Check if output follows the required format"""
    scores = {}

    # Check think tag
    think_pattern = r'<think>.*?</think>'
    scores['has_think'] = 1.0 if re.search(think_pattern, output_text, re.DOTALL) else 0.0

    # Check answer tag with JSON
    answer_pattern = r'<answer>\s*\[.*?\]\s*</answer>'
    scores['has_answer'] = 1.0 if re.search(answer_pattern, output_text, re.DOTALL) else 0.0

    # Check JSON parseable
    try:
        json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', output_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
            scores['json_valid'] = 1.0

            # Check bbox and point format
            if len(data) > 0:
                item = data[0]
                scores['has_bbox'] = 1.0 if 'bbox_2d' in item and len(item['bbox_2d']) == 4 else 0.0
                scores['has_point'] = 1.0 if 'point_2d' in item and len(item['point_2d']) == 2 else 0.0
            else:
                scores['has_bbox'] = 0.0
                scores['has_point'] = 0.0
        else:
            scores['json_valid'] = 0.0
            scores['has_bbox'] = 0.0
            scores['has_point'] = 0.0
    except:
        scores['json_valid'] = 0.0
        scores['has_bbox'] = 0.0
        scores['has_point'] = 0.0

    return scores


def create_prompt(problem: str, image_size: int, system_prompt: str, num_images: int = 1) -> str:
    """
    Create prompt exactly matching training format.
    Based on verl/utils/rl_dataset.py

    Args:
        problem: Task description
        image_size: Image dimension (e.g., 280)
        system_prompt: System message
        num_images: Number of images (1 for single-image, 8 for multi-image)
    """
    if num_images == 1:
        # Single-image prompt (original)
        user_prompt = (
            f"<image>\n"
            f"Task: {problem}\n\n"
            f"Instructions:\n"
            f"1. This is a brain MRI scan. Look for abnormal regions that appear different from normal brain tissue.\n"
            f"2. Brain tumors typically appear as areas with altered intensity (brighter or darker regions) or irregular shapes.\n"
            f"3. Locate the tumor region and determine its 2D bounding box [x_min, y_min, x_max, y_max] and center point [x, y].\n"
            f"4. Use normalized coordinates in range [0, 1000] for bbox_2d and point_2d.\n"
            f"5. Output your analysis in <think></think> tags, then provide the final answer in <answer></answer> tags.\n\n"
            f"Output format example:\n"
            f"<think>Analysis of the image shows...</think>\n"
            f'<answer>[{{"bbox_2d": [10,100,200,210], "point_2d": [120,155]}}]</answer>'
        )
    else:
        # Multi-image prompt - MUST match training prompt exactly for consistency
        image_tags = "<image>" * num_images
        user_prompt = (
            f"{image_tags}\n"
            f"Task: {problem}\n\n"
            f"Instructions:\n"
            f"1. You are viewing {num_images} MRI slices sampled uniformly from a 3D brain scan (from shallow to deep).\n"
            f"2. Each slice may contain brain tumor with varying sizes. Your task is to identify the slice with the LARGEST tumor.\n"
            f"3. Brain tumors typically appear as areas with altered intensity (brighter or darker regions) or irregular shapes.\n"
            f"4. Locate the largest tumor region and determine its 2D bounding box [x_min, y_min, x_max, y_max] and center point [x, y].\n"
            f"5. Use normalized coordinates in range [0, 1000] for bbox_2d and point_2d.\n"
            f"6. Output your analysis in <think></think> tags, then provide the final answer in <answer></answer> tags.\n\n"
            f"Output format example:\n"
            f"<think>Analyzing all {num_images} slices... Slice {num_images//2} shows the largest tumor region...</think>\n"
            f'<answer>[{{"bbox_2d": [10,100,200,210], "point_2d": [120,155]}}]</answer>'
        )

    # Add image size hint (as done in training)
    size_hint = f"Image size: {image_size}x{image_size} pixels. "

    return [{
        "role": "system",
        "content": system_prompt
    }, {
        "role": "user",
        "content": size_hint + user_prompt
    }]


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)

    print("=" * 80)
    print("Brain Tumor Detection Evaluation (Multi-Image)")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.data_path}")
    print(f"Output: {args.output_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Images per sample: {args.num_images} {'(single-image mode)' if args.num_images == 1 else '(multi-image mode)'}")
    print("=" * 80)

    # Load model
    print("\n[1/4] Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_path, padding_side="left")

    # Load dataset
    print(f"\n[2/4] Loading dataset from {args.data_path}...")
    try:
        # Try loading as DatasetDict first
        dataset_dict = load_from_disk(args.data_path)
        if hasattr(dataset_dict, 'keys') and 'train' in dataset_dict:
            dataset = dataset_dict['train']
        else:
            dataset = dataset_dict
    except:
        # Try loading train subdirectory
        dataset = load_from_disk(os.path.join(args.data_path, 'train'))

    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    print(f"Dataset loaded: {len(dataset)} samples")

    # Prepare messages
    print("\n[3/4] Preparing messages...")
    messages = []
    metadata = []

    for idx, item in enumerate(tqdm(dataset, desc="Preparing")):
        # Get images (support both single-image and multi-image)
        images = []
        if args.num_images == 1:
            # Single-image mode: take first image only
            if "image" in item:
                images = [item["image"]]
            elif "images" in item and len(item["images"]) > 0:
                images = [item["images"][0]]
        else:
            # Multi-image mode: take all images (up to num_images)
            if "images" in item:
                images = item["images"][:args.num_images]
            elif "image" in item:
                # Fallback: single image repeated (not ideal, but prevents crash)
                images = [item["image"]]
                print(f"Warning: Sample {idx} has single 'image' but num_images={args.num_images}, using single image")

        if not images:
            print(f"Warning: No images found in sample {idx}")
            continue

        # Process all images
        processed_images = []
        for img in images:
            # Ensure RGB
            if img.mode != "RGB":
                img = img.convert("RGB")
            # Resize if needed
            if img.size != (args.image_size, args.image_size):
                img = img.resize((args.image_size, args.image_size), PILImage.Resampling.BILINEAR)
            processed_images.append(img)

        # Get problem text
        problem = item.get("problem", "locate brain tumor in this mri image")

        # Create message with multi-image support
        msg = create_prompt(problem, args.image_size, args.system_prompt, num_images=args.num_images)

        # Build content with images + text
        content = []
        for img in processed_images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": msg[1]["content"]})

        msg[1]["content"] = content
        messages.append(msg)

        # Store metadata
        metadata.append({
            "index": idx,
            "problem": problem,
            "ground_truth": item.get("solution", "[]"),
            "image_width": processed_images[0].size[0],
            "image_height": processed_images[0].size[1],
            "num_images": len(processed_images),
        })

    # Run inference
    print(f"\n[4/4] Running inference on {len(messages)} samples...")
    all_results = []

    for i in tqdm(range(0, len(messages), args.batch_size), desc="Evaluating"):
        batch_messages = messages[i:i + args.batch_size]
        batch_metadata = metadata[i:i + args.batch_size]

        # Prepare inputs
        texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                 for msg in batch_messages]

        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Generate
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True
            )

        # Decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        outputs = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        # Evaluate each output
        for j, output_text in enumerate(outputs):
            meta = batch_metadata[j]

            # Extract predictions
            pred_bboxes, pred_points = extract_bbox_and_point(output_text, args.image_size)

            # Parse ground truth
            try:
                gt_data = json.loads(meta["ground_truth"])
                gt_bboxes = [item["bbox_2d"] for item in gt_data if "bbox_2d" in item]
                gt_points = [item["point_2d"] for item in gt_data if "point_2d" in item]
            except:
                gt_bboxes = []
                gt_points = []

            # Compute metrics
            format_scores = check_format(output_text)

            # Compute IoU (take best match for simplicity)
            max_iou = 0.0
            min_point_dist = float('inf')

            if len(pred_bboxes) > 0 and len(gt_bboxes) > 0:
                for pred_bbox in pred_bboxes:
                    for gt_bbox in gt_bboxes:
                        iou = compute_bbox_iou(pred_bbox, gt_bbox)
                        max_iou = max(max_iou, iou)

                for pred_point in pred_points:
                    for gt_point in gt_points:
                        dist = compute_point_distance(pred_point, gt_point)
                        min_point_dist = min(min_point_dist, dist)

            if min_point_dist == float('inf'):
                min_point_dist = args.image_size  # Maximum possible distance

            # Store result
            result = {
                "index": meta["index"],
                "problem": meta["problem"],
                "prediction": output_text,
                "ground_truth": meta["ground_truth"],
                "pred_bboxes": pred_bboxes,
                "pred_points": pred_points,
                "gt_bboxes": gt_bboxes,
                "gt_points": gt_points,
                "bbox_iou": max_iou,
                "point_distance": min_point_dist,
                **format_scores
            }
            all_results.append(result)

        # Clean up
        del inputs, generated_ids, generated_ids_trimmed
        torch.cuda.empty_cache()

    # Calculate summary statistics
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    # Format metrics
    format_metrics = {
        "has_think": np.mean([r["has_think"] for r in all_results]),
        "has_answer": np.mean([r["has_answer"] for r in all_results]),
        "json_valid": np.mean([r["json_valid"] for r in all_results]),
        "has_bbox": np.mean([r["has_bbox"] for r in all_results]),
        "has_point": np.mean([r["has_point"] for r in all_results]),
    }

    print("\nFormat Metrics:")
    for key, value in format_metrics.items():
        print(f"  {key:20s}: {value:.3f} ({value*100:.1f}%)")

    # Accuracy metrics
    ious = [r["bbox_iou"] for r in all_results]
    point_dists = [r["point_distance"] for r in all_results]

    print("\nAccuracy Metrics:")
    print(f"  {'Bbox IoU (mean)':20s}: {np.mean(ious):.3f}")
    print(f"  {'Bbox IoU (median)':20s}: {np.median(ious):.3f}")
    print(f"  {'Bbox IoU (max)':20s}: {np.max(ious):.3f}")
    print(f"  {'IoU > 0.5':20s}: {np.mean([iou > 0.5 for iou in ious]):.3f} ({np.mean([iou > 0.5 for iou in ious])*100:.1f}%)")
    print(f"  {'IoU > 0.7':20s}: {np.mean([iou > 0.7 for iou in ious]):.3f} ({np.mean([iou > 0.7 for iou in ious])*100:.1f}%)")
    print(f"  {'Point dist (mean)':20s}: {np.mean(point_dists):.1f} pixels")
    print(f"  {'Point dist (median)':20s}: {np.median(point_dists):.1f} pixels")

    # Distribution by IoU ranges
    print("\nIoU Distribution:")
    iou_ranges = [
        (0.0, 0.1, "0.0-0.1 (poor)"),
        (0.1, 0.3, "0.1-0.3 (low)"),
        (0.3, 0.5, "0.3-0.5 (medium)"),
        (0.5, 0.7, "0.5-0.7 (good)"),
        (0.7, 0.9, "0.7-0.9 (very good)"),
        (0.9, 1.0, "0.9-1.0 (excellent)"),
    ]
    for low, high, label in iou_ranges:
        count = sum(1 for iou in ious if low <= iou < high)
        pct = count / len(ious) * 100
        print(f"  {label:20s}: {count:4d} samples ({pct:5.1f}%)")

    # Save results
    output_file = os.path.join(args.output_path, "evaluation_results.json")
    with open(output_file, "w") as f:
        json.dump({
            "config": vars(args),
            "summary": {
                "total_samples": len(all_results),
                "format_metrics": format_metrics,
                "accuracy_metrics": {
                    "bbox_iou_mean": float(np.mean(ious)),
                    "bbox_iou_median": float(np.median(ious)),
                    "bbox_iou_max": float(np.max(ious)),
                    "bbox_iou_gt_0.5": float(np.mean([iou > 0.5 for iou in ious])),
                    "bbox_iou_gt_0.7": float(np.mean([iou > 0.7 for iou in ious])),
                    "point_distance_mean": float(np.mean(point_dists)),
                    "point_distance_median": float(np.median(point_dists)),
                },
            },
            "detailed_results": all_results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Results saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
