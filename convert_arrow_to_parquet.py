#!/usr/bin/env python3
"""
Convert HuggingFace Arrow datasets to Parquet format for verl v0.5.0 compatibility.

This script converts the brain tumor dataset from Arrow format (data/**/train/)
to Parquet format (data/**/train.parquet) required by verl v0.5.0.

Usage:
    python3 convert_arrow_to_parquet.py
"""

import os
import datasets

# List of Arrow dataset directories to convert
ARROW_DATASETS = [
    "data/BraTS2024-BraTS-GLI-T1C/train",
    "data/BraTS2024-BraTS-GLI-T2F/train",
    "data/MSD-Task01-BrainTumour-T1Gd/train",
    "data/MSD-Task01-BrainTumour-FLAIR/train",
    "data/BraTS2024_MEN_RT_TrainingData_video/train",
    "data/BraTS2024-BraTS-GLI-Additional-T1C/train",
    "data/BraTS2024-BraTS-GLI-Additional-T2F/train",
]


def convert_arrow_to_parquet(arrow_dir: str) -> str:
    """
    Convert a HuggingFace Arrow dataset directory to Parquet file.

    Args:
        arrow_dir: Path to Arrow dataset directory (e.g., "data/xxx/train")

    Returns:
        Path to generated Parquet file (e.g., "data/xxx/train.parquet")
    """
    if not os.path.exists(arrow_dir):
        print(f"⚠️  Skipping {arrow_dir} (directory not found)")
        return None

    # Load Arrow dataset
    print(f"📂 Loading {arrow_dir}...", end=" ", flush=True)
    try:
        ds = datasets.load_from_disk(arrow_dir)
        print(f"✓ ({len(ds)} samples)")
    except Exception as e:
        print(f"✗ Failed to load: {e}")
        return None

    # Generate Parquet file path
    parquet_path = arrow_dir.replace("/train", "/train.parquet")

    # Convert to Parquet
    print(f"💾 Saving to {parquet_path}...", end=" ", flush=True)
    try:
        ds.to_parquet(parquet_path)
        file_size_mb = os.path.getsize(parquet_path) / (1024 * 1024)
        print(f"✓ ({file_size_mb:.2f} MB)")
        return parquet_path
    except Exception as e:
        print(f"✗ Failed to save: {e}")
        return None


def main():
    """Convert all Arrow datasets to Parquet format."""
    print("=" * 70)
    print("Converting Arrow datasets to Parquet for verl v0.5.0")
    print("=" * 70)

    converted_count = 0
    failed_count = 0
    skipped_count = 0

    for arrow_dir in ARROW_DATASETS:
        result = convert_arrow_to_parquet(arrow_dir)
        if result:
            converted_count += 1
        elif result is None and not os.path.exists(arrow_dir):
            skipped_count += 1
        else:
            failed_count += 1
        print()

    # Summary
    print("=" * 70)
    print(f"✓ Converted: {converted_count}")
    print(f"⚠️  Skipped:   {skipped_count}")
    print(f"✗ Failed:    {failed_count}")
    print("=" * 70)

    if converted_count > 0:
        print("\n✅ Conversion completed successfully!")
        print("You can now run the training script with Parquet files.")
    elif failed_count > 0:
        print("\n❌ Some conversions failed. Please check the errors above.")
        exit(1)
    else:
        print("\n⚠️  No datasets were converted.")


if __name__ == "__main__":
    main()
