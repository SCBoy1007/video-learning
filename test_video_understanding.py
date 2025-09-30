#!/usr/bin/env python3
"""
Test script to verify Qwen2.5-VL-7B's video understanding capability
Usage: python test_video_understanding.py
"""

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from decord import VideoReader, cpu
import numpy as np

def load_video(video_path, max_frames=32):
    """Load video frames using decord"""
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)

    # Sample frames uniformly
    if total_frames > max_frames:
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    else:
        indices = list(range(total_frames))

    frames = vr.get_batch(indices).asnumpy()
    print(f"Loaded {len(frames)} frames from video (total {total_frames} frames)")
    return frames

def main():
    print("=" * 70)
    print("Testing Qwen2.5-VL-7B Video Understanding")
    print("=" * 70)

    # Configuration
    model_path = "pretrained_models/Qwen2.5-VL-7B-Instruct"
    video_path = "data/Animated-Logo.mp4"

    print(f"\n[1/4] Loading model from: {model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)
    print("✓ Model loaded successfully")

    print(f"\n[2/4] Loading video from: {video_path}")
    video_frames = load_video(video_path)
    print("✓ Video loaded successfully")

    # Test 1: Simple description
    print("\n[3/4] Test 1: Simple video description")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": "Describe what you see in this video."}
            ]
        }
    ]

    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text_prompt], videos=[video_frames], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    print("Generating response...")
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False
        )

    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    print("\n" + "=" * 70)
    print("MODEL RESPONSE:")
    print("=" * 70)
    print(response)
    print("=" * 70)

    # Test 2: Specific question about video content
    print("\n[4/4] Test 2: Asking if this is a video")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": "Is this a video or an image? Please explain."}
            ]
        }
    ]

    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text_prompt], videos=[video_frames], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    print("Generating response...")
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False
        )

    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    print("\n" + "=" * 70)
    print("MODEL RESPONSE:")
    print("=" * 70)
    print(response)
    print("=" * 70)

    print("\n✓ All tests completed!")
    print("\nNOTE: If the model describes motion/animation, it understands this is a video.")
    print("      If it describes static content, it may be treating frames as images.")

if __name__ == "__main__":
    main()