#!/usr/bin/env python3
"""Test script to diagnose video loading issues with Qwen2.5-VL processor"""

import os
import sys

print("=" * 80)
print("Video Loading Diagnostics for Qwen2.5-VL")
print("=" * 80)

# Test 1: Check if video file exists
print("\n[Test 1] Checking video file existence...")
video_path = 'data/BraTS_GLI_TrainingData_video/train/videos/BraTS-GLI-03021-101.mp4'
print(f"Video path: {video_path}")
print(f"File exists: {os.path.exists(video_path)}")
if os.path.exists(video_path):
    file_size = os.path.getsize(video_path)
    print(f"File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
else:
    print("ERROR: Video file not found!")
    sys.exit(1)

# Test 2: Test decord video reading
print("\n[Test 2] Testing decord video reader...")
try:
    from decord import VideoReader
    vr = VideoReader(video_path)
    print(f"✓ Decord successfully loaded video")
    print(f"  - Total frames: {len(vr)}")
    print(f"  - Frame shape: {vr[0].shape}")
    print(f"  - Frame dtype: {vr[0].dtype}")
except ImportError:
    print("✗ decord not installed!")
    print("  Install with: pip install decord")
    sys.exit(1)
except Exception as e:
    print(f"✗ Decord failed to read video: {e}")
    sys.exit(1)

# Test 3: Test Qwen2.5-VL processor
print("\n[Test 3] Testing Qwen2.5-VL processor...")
try:
    from transformers import Qwen2VLProcessor
    print("Attempting to load processor from pretrained_models/Qwen2.5-VL-7B-Instruct...")
    processor = Qwen2VLProcessor.from_pretrained("pretrained_models/Qwen2.5-VL-7B-Instruct")
    print("✓ Processor loaded successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Failed to load processor: {e}")
    sys.exit(1)

# Test 4: Test processor with video path (string)
print("\n[Test 4] Testing processor with video path string...")
try:
    result = processor(videos=[video_path], return_tensors="pt")
    print(f"✓ Processor accepted video path string")
    print(f"  - Output keys: {list(result.keys())}")
    for key, value in result.items():
        if hasattr(value, 'shape'):
            print(f"  - {key} shape: {value.shape}")
except Exception as e:
    print(f"✗ Processor failed with path string: {e}")
    print(f"  Error type: {type(e).__name__}")
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()

# Test 5: Test processor with loaded video frames
print("\n[Test 5] Testing processor with pre-loaded video frames...")
try:
    from decord import VideoReader
    vr = VideoReader(video_path)
    # Load all frames as numpy array
    frames = vr.get_batch(list(range(len(vr)))).asnumpy()
    print(f"  - Loaded {len(frames)} frames, shape: {frames.shape}")

    # Try processing the numpy array
    result = processor(videos=[frames], return_tensors="pt")
    print(f"✓ Processor accepted numpy array")
    print(f"  - Output keys: {list(result.keys())}")
    for key, value in result.items():
        if hasattr(value, 'shape'):
            print(f"  - {key} shape: {value.shape}")
except Exception as e:
    print(f"✗ Processor failed with numpy array: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Check transformers version
print("\n[Test 6] Checking library versions...")
try:
    import transformers
    print(f"  - transformers version: {transformers.__version__}")
except:
    pass

try:
    import decord
    print(f"  - decord version: {decord.__version__}")
except:
    pass

try:
    import torch
    print(f"  - torch version: {torch.__version__}")
except:
    pass

print("\n" + "=" * 80)
print("Diagnostics complete!")
print("=" * 80)