import argparse
from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from datasets import load_from_disk, load_dataset
from PIL import Image as PILImage
from tqdm import tqdm
import os
import re

# ==================== TUMOR EXISTENCE DETECTION EVALUATION ====================
# This script evaluates tumor existence detection (yes/no), NOT localization
# Metrics: Accuracy, Precision, Recall, F1-Score
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Tumor existence detection evaluation")
    parser.add_argument("--reasoning_model_path", type=str, default="Ricky06662/Seg-Zero-7B",
                        help="Path to the trained VLM model")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Directory to save evaluation results")
    parser.add_argument("--test_data_path", type=str, required=True,
                        help="Path to test dataset")
    parser.add_argument("--idx", type=int, required=True,
                        help="Partition index for parallel evaluation")
    parser.add_argument("--num_parts", type=int, required=True,
                        help="Total number of partitions")
    parser.add_argument("--batch_size", type=int, default=50,
                        help="Batch size for inference")
    return parser.parse_args()


def extract_existence_answer(output_text: str) -> tuple[str, str]:
    """
    Extract tumor existence prediction from model output

    Returns:
        tuple: (answer, thinking)
            - answer: "yes", "no", or "unknown"
            - thinking: content inside <think> tags
    """
    # Extract thinking process
    think_pattern = r'<think>(.*?)</think>'
    think_match = re.search(think_pattern, output_text, re.DOTALL)
    thinking = think_match.group(1).strip() if think_match else ""

    # Extract answer
    answer_pattern = r'<answer>(.*?)</answer>'
    answer_match = re.search(answer_pattern, output_text, re.DOTALL)

    if answer_match:
        answer_content = answer_match.group(1).strip().lower()
        if "yes" in answer_content:
            answer = "yes"
        elif "no" in answer_content:
            answer = "no"
        else:
            answer = "unknown"
    else:
        answer = "unknown"

    return answer, thinking


def main():
    args = parse_args()

    print("="*80)
    print(f"Tumor Existence Detection Evaluation - Partition {args.idx+1}/{args.num_parts}")
    print("="*80)

    # Load model
    print("\n[1/5] Loading model...")
    reasoning_model = AutoModelForVision2Seq.from_pretrained(
        args.reasoning_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    reasoning_model.eval()

    processor = AutoProcessor.from_pretrained(args.reasoning_model_path, padding_side="left")
    print(f"✓ Model loaded: {args.reasoning_model_path}")

    # Load dataset
    print("\n[2/5] Loading dataset...")
    resize_size = 840

    try:
        dataset = load_dataset(args.test_data_path, split='test')
    except:
        dataset = load_from_disk(args.test_data_path)
        if hasattr(dataset, 'keys') and 'test' in dataset:
            dataset = dataset['test']

    total_len = len(dataset)
    part_size = total_len // args.num_parts
    start_idx = args.idx * part_size
    end_idx = start_idx + part_size if args.idx < args.num_parts - 1 else total_len

    dataset = dataset.select(range(start_idx, end_idx))
    print(f"✓ Dataset loaded: {len(dataset)} samples (partition {args.idx+1}/{args.num_parts})")

    # Check if dataset has has_tumor field
    if 'has_tumor' not in dataset[0]:
        print("WARNING: 'has_tumor' field not found in dataset. Assuming all samples have tumors.")
        has_ground_truth = False
    else:
        has_ground_truth = True
        print("✓ Dataset has ground truth 'has_tumor' field")

    # Prepare prompts
    print("\n[3/5] Preparing prompts...")
    QUESTION_TEMPLATE = \
        "<image>\n" \
        "Task: {Question}\n\n" \
        "Instructions:\n" \
        "1. This is a brain MRI scan. Carefully examine the entire image.\n" \
        "2. Brain tumors typically appear as abnormal regions with altered intensity (brighter or darker areas) or irregular shapes.\n" \
        "3. Determine whether a tumor exists in this image. Answer 'yes' if tumor is present, 'no' if not.\n" \
        "4. Output your analysis in <think></think> tags, then provide your answer in <answer></answer> tags.\n\n" \
        "Output format example:\n" \
        "<think>Analyzing the image, I observe...</think>\n" \
        "<answer>{Answer}</answer>"

    messages = []
    metadata_list = []

    for item in dataset:
        image = item["image"].convert("RGB") if hasattr(item["image"], 'convert') else item["image"]
        question = item.get("problem", item.get("text", "Is there a tumor in this image?"))

        message = [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image.resize((resize_size, resize_size), PILImage.BILINEAR)
                },
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(
                        Question=question.lower().strip(".\"?!"),
                        Answer="yes"  # Example format
                    )
                }
            ]
        }]
        messages.append(message)

        metadata = {
            "image_id": item.get("image_id", "unknown"),
            "ann_id": item.get("ann_id", "unknown"),
        }
        if has_ground_truth:
            metadata["has_tumor"] = item["has_tumor"]

        metadata_list.append(metadata)

    print(f"✓ Prepared {len(messages)} prompts")

    # Run inference
    print(f"\n[4/5] Running inference (batch_size={args.batch_size})...")
    all_outputs = []

    for i in tqdm(range(0, len(messages), args.batch_size), desc="Inference"):
        batch_messages = messages[i:i + args.batch_size]
        batch_metadata = metadata_list[i:i + args.batch_size]

        # Prepare inputs
        text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in batch_messages]

        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Generate predictions
        with torch.inference_mode():
            generated_ids = reasoning_model.generate(
                **inputs,
                use_cache=True,
                max_new_tokens=512,
                do_sample=False
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        batch_output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # Parse outputs
        for output_idx, output_text in enumerate(batch_output_text):
            try:
                answer, thinking = extract_existence_answer(output_text)

                result = {
                    "image_id": batch_metadata[output_idx]["image_id"],
                    "ann_id": batch_metadata[output_idx]["ann_id"],
                    "prediction": answer,  # "yes", "no", or "unknown"
                    "thinking": thinking,
                    "full_output": output_text
                }

                if has_ground_truth:
                    result["ground_truth"] = "yes" if batch_metadata[output_idx]["has_tumor"] else "no"
                    result["correct"] = (result["prediction"] == result["ground_truth"])

                all_outputs.append(result)

            except Exception as e:
                print(f"\nError processing sample {batch_metadata[output_idx]['image_id']}: {e}")
                # Add failed result
                result = {
                    "image_id": batch_metadata[output_idx]["image_id"],
                    "ann_id": batch_metadata[output_idx]["ann_id"],
                    "prediction": "error",
                    "thinking": "",
                    "full_output": output_text,
                    "error": str(e)
                }
                if has_ground_truth:
                    result["ground_truth"] = "yes" if batch_metadata[output_idx]["has_tumor"] else "no"
                    result["correct"] = False
                all_outputs.append(result)

        # Clean GPU memory
        del inputs, generated_ids, generated_ids_trimmed
        torch.cuda.empty_cache()

    # Compute metrics
    print("\n[5/5] Computing metrics...")
    if has_ground_truth:
        total = len(all_outputs)
        correct = sum(1 for r in all_outputs if r.get("correct", False))

        # Confusion matrix
        tp = sum(1 for r in all_outputs if r["prediction"] == "yes" and r["ground_truth"] == "yes")
        tn = sum(1 for r in all_outputs if r["prediction"] == "no" and r["ground_truth"] == "no")
        fp = sum(1 for r in all_outputs if r["prediction"] == "yes" and r["ground_truth"] == "no")
        fn = sum(1 for r in all_outputs if r["prediction"] == "no" and r["ground_truth"] == "yes")

        accuracy = correct / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics = {
            "total_samples": total,
            "correct": correct,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": {
                "true_positive": tp,
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn
            }
        }

        print("\n" + "="*80)
        print("EVALUATION METRICS")
        print("="*80)
        print(f"Total Samples:  {total}")
        print(f"Correct:        {correct}")
        print(f"Accuracy:       {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision:      {precision:.4f}")
        print(f"Recall:         {recall:.4f}")
        print(f"F1-Score:       {f1:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TP: {tp:4d}  |  FP: {fp:4d}")
        print(f"  FN: {fn:4d}  |  TN: {tn:4d}")
        print("="*80)
    else:
        print("WARNING: No ground truth available, skipping metric computation")
        metrics = None

    # Save results
    os.makedirs(args.output_path, exist_ok=True)
    output_file = os.path.join(args.output_path, f"output_{args.idx}.json")

    output_data = {
        "predictions": all_outputs,
        "metrics": metrics
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Results saved to: {output_file}")
    print("="*80)


if __name__ == "__main__":
    main()
