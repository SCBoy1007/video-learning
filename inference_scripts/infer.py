import argparse
from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image as PILImage
import re
import matplotlib.pyplot as plt

# ==================== TUMOR EXISTENCE DETECTION INFERENCE ====================
# This script performs tumor existence detection (yes/no), NOT localization
# For localization (bbox/point + SAM2), see infer_localization.py (archived)
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Tumor existence detection inference")
    parser.add_argument("--reasoning_model_path", type=str, default="Ricky06662/Seg-Zero-7B",
                        help="Path to the trained VLM model")
    parser.add_argument("--text", type=str, default="Is there a tumor in this brain MRI scan?",
                        help="Question to ask about the image")
    parser.add_argument("--image_path", type=str, default="./assets/test_image.png",
                        help="Path to input image")
    parser.add_argument("--output_path", type=str, default="./inference_scripts/test_output.png",
                        help="Path to save visualization")
    return parser.parse_args()


def extract_existence_answer(output_text: str) -> tuple[str, str, str]:
    """
    Extract tumor existence prediction from model output

    Returns:
        tuple: (answer, thinking, full_output)
            - answer: "yes", "no", or "unknown"
            - thinking: content inside <think> tags
            - full_output: complete model output
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

    return answer, thinking, output_text


def main():
    args = parse_args()

    print("="*60)
    print("Tumor Existence Detection Inference")
    print("="*60)

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

    # Load and prepare image
    print("\n[2/5] Loading image...")
    image = PILImage.open(args.image_path).convert("RGB")
    original_width, original_height = image.size
    resize_size = 840
    print(f"✓ Image loaded: {args.image_path} ({original_width}×{original_height})")

    # Prepare prompt for existence detection
    print("\n[3/5] Preparing prompt...")
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

    messages = [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image.resize((resize_size, resize_size), PILImage.BILINEAR)
            },
            {
                "type": "text",
                "text": QUESTION_TEMPLATE.format(
                    Question=args.text.lower().strip("."),
                    Answer="yes"  # Example format
                )
            }
        ]
    }]

    print(f"✓ Question: {args.text}")

    # Inference
    print("\n[4/5] Running inference...")
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info([messages])

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    with torch.inference_mode():
        generated_ids = reasoning_model.generate(
            **inputs,
            use_cache=True,
            max_new_tokens=512,  # Reduced from 1024 since we only need yes/no
            do_sample=False
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # Parse results
    answer, thinking, full_output = extract_existence_answer(output_text)

    print("✓ Inference complete")

    # Display results
    print("\n[5/5] Results:")
    print("="*60)
    print(f"Answer: {answer.upper()}")
    print(f"\nThinking process:\n{thinking}")
    print(f"\nFull output:\n{full_output}")
    print("="*60)

    # Visualize
    print(f"\n[6/6] Saving visualization to {args.output_path}...")

    plt.figure(figsize=(10, 5))

    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Input Image', fontsize=14)
    plt.axis('off')

    # Result display
    plt.subplot(1, 2, 2)
    plt.imshow(image, alpha=0.3)

    # Color-code the result
    color = 'red' if answer == 'yes' else 'green' if answer == 'no' else 'gray'
    result_text = f"Tumor Detected: {answer.upper()}"

    plt.text(
        0.5, 0.5, result_text,
        horizontalalignment='center',
        verticalalignment='center',
        transform=plt.gca().transAxes,
        fontsize=20,
        color=color,
        weight='bold',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    plt.title('Prediction', fontsize=14)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(args.output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Visualization saved")
    print("\n" + "="*60)
    print("Inference complete!")
    print("="*60)


if __name__ == "__main__":
    main()
