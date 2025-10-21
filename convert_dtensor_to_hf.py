#!/usr/bin/env python3
"""
Convert verl's DTensor FSDP checkpoint to HuggingFace format.
This script correctly handles DTensor by extracting local tensors and concatenating them.
"""

import os
import torch
import argparse
import shutil
from collections import OrderedDict


def load_all_shards(checkpoint_dir, world_size=4):
    """Load all FSDP shards."""
    print(f"Loading {world_size} shards from: {checkpoint_dir}")
    shards = []
    for rank in range(world_size):
        shard_path = os.path.join(checkpoint_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
        print(f"  Loading shard {rank}: {shard_path}")
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        shards.append(shard)
    print(f"✓ Loaded {len(shards)} shards\n")
    return shards


def convert_dtensor_to_tensor(dtensor_list):
    """
    Convert list of DTensors (one per rank) to a single merged tensor.

    DTensor stores:
    - _local_tensor: the actual tensor data for this rank
    - placements: how the tensor is distributed (e.g., Shard(dim=0))
    - Global shape vs local shape

    For Shard(dim=0), we concatenate local tensors along dim 0.
    For Replicate, all ranks have the same tensor, so we take rank 0.
    """
    # Get first DTensor to check type and placements
    first = dtensor_list[0]

    # If it's a regular tensor (not DTensor), all ranks should be identical
    if not hasattr(first, '_local_tensor'):
        # Regular tensor - all ranks have the same value
        return first.clone()

    # It's a DTensor - check placements
    placements = first.placements

    # Extract local tensors from all ranks
    local_tensors = [dt._local_tensor for dt in dtensor_list]

    # Check placement type
    if len(placements) > 0 and hasattr(placements[0], 'dim'):
        # Sharded - concatenate along the sharded dimension
        shard_dim = placements[0].dim
        merged = torch.cat(local_tensors, dim=shard_dim)
        return merged
    else:
        # Replicated - all ranks have the same tensor, use rank 0
        return local_tensors[0].clone()


def merge_shards(shards):
    """Merge all shards into a single state dict."""
    print("Merging shards...")

    # Get all keys from first shard
    keys = list(shards[0].keys())
    merged_state_dict = OrderedDict()

    for key in keys:
        # Get this parameter from all shards
        dtensors = [shard[key] for shard in shards]

        # Convert to single tensor
        try:
            merged_tensor = convert_dtensor_to_tensor(dtensors)
            merged_state_dict[key] = merged_tensor

            # Print info for first few and some important layers
            if len(merged_state_dict) <= 5 or 'q_proj' in key or 'lm_head' in key:
                local_shape = dtensors[0]._local_tensor.shape if hasattr(dtensors[0], '_local_tensor') else dtensors[0].shape
                print(f"  {key}:")
                print(f"    Local shape: {local_shape} -> Merged shape: {merged_tensor.shape}")

        except Exception as e:
            print(f"  ERROR merging {key}: {e}")
            # Fallback to rank 0
            merged_state_dict[key] = shards[0][key]

    print(f"\n✓ Merged {len(merged_state_dict)} parameters\n")
    return merged_state_dict


def save_huggingface_checkpoint(state_dict, output_dir, config_dir):
    """Save as HuggingFace checkpoint."""
    print(f"Saving to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Save weights
    model_path = os.path.join(output_dir, "pytorch_model.bin")
    torch.save(state_dict, model_path)

    file_size_gb = os.path.getsize(model_path) / (1024 ** 3)
    print(f"  ✓ Saved weights: {model_path} ({file_size_gb:.2f} GB)")

    # Copy config files
    if os.path.exists(config_dir):
        print(f"\nCopying config files from: {config_dir}")
        for file in os.listdir(config_dir):
            if file.endswith(('.json', '.txt', '.jinja')):
                src = os.path.join(config_dir, file)
                dst = os.path.join(output_dir, file)
                shutil.copy2(src, dst)
                print(f"  ✓ {file}")

    print("\n" + "=" * 80)
    print("✓ Conversion complete!")
    print(f"✓ Model saved to: {output_dir}")
    print(f"✓ Model size: {file_size_gb:.2f} GB")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Convert FSDP DTensor checkpoint to HuggingFace")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="brain_tumor_workdir/run_brain_tumor_image_4x80G/global_step_200/actor",
        help="FSDP checkpoint directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/Qwen3-VL-8B-BrainTumor-Step200",
        help="Output directory"
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=4,
        help="Number of shards"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("DTensor FSDP to HuggingFace Converter")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Output: {args.output_dir}")
    print(f"World size: {args.world_size}")
    print()

    # Load shards
    shards = load_all_shards(args.checkpoint_dir, args.world_size)

    # Merge shards
    merged_state_dict = merge_shards(shards)

    # Save
    config_dir = os.path.join(args.checkpoint_dir, "huggingface")
    save_huggingface_checkpoint(merged_state_dict, args.output_dir, config_dir)


if __name__ == "__main__":
    main()
