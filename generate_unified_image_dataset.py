#!/usr/bin/env python3
"""
Unified Image Dataset Generator for Brain Tumor Detection
Extracts 8 uniformly distributed slices from 3D MRI, resizes to 280×280,
and generates Seg-Zero format annotations with tumor existence labels

Sampling Strategy:
- 8 slices at positions: 15%, 25%, 35%, 45%, 55%, 65%, 75%, 85% of volume depth
- Avoids first/last slices which typically contain less relevant tissue

Supports three data sources:
- BraTS-GLI: BraTS-GLI-*/-seg.nii.gz + -t1c/t1n/t2f/t2w.nii.gz
- MSD: imagesTr/BRATS_*.nii.gz (4D) + labelsTr/BRATS_*.nii.gz
- MEN-RT: BraTS-MEN-RT-*/_gtv.nii.gz + _t1c.nii.gz

Output format (Seg-Zero compatible):
{
    "id": str,                          # e.g., "BraTS-GLI-00001-t1c-slice0"
    "case_id": str,                     # e.g., "BraTS-GLI-00001-t1c"
    "slice_idx": int,                   # 0-7 (which of the 8 slices)
    "slice_position": float,            # 0.15-0.85 (position in volume)
    "problem": str,                     # Task description
    "has_tumor": bool,                  # Whether tumor exists in this slice
    "solution": str,                    # '[{"bbox_2d": [...], "point_2d": [...]}]' or empty
    "image": PIL.Image (280×280),
    "img_width": 280,
    "img_height": 280
}
"""

import argparse
import os
import json
import nibabel as nib
import numpy as np
from pathlib import Path
from PIL import Image
from datasets import Features, Value
from datasets import Image as DatasetImage
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm
import multiprocessing as mp
import gc
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass


# ==================== Configuration ====================
# Get script directory for resolving relative paths
SCRIPT_DIR = Path(__file__).parent.resolve()

TARGET_SIZE = 280  # Single dimension for square images (280×280)
_CPU_COUNT = os.cpu_count() or 1
_DEFAULT_WORKERS = 1
if _CPU_COUNT > 2:
    _DEFAULT_WORKERS = min(4, max(1, _CPU_COUNT // 2))
NUM_WORKERS = int(os.environ.get("DATASET_NUM_WORKERS", _DEFAULT_WORKERS))


class StopProcessing(Exception):
    """Raised to break out once the modality-per-run limit is reached."""


@dataclass
class ProcessingState:
    limit: Optional[int]
    modalities_finished: int = 0

    def should_stop(self) -> bool:
        return self.limit is not None and self.modalities_finished >= self.limit

    def register_modality(self, finished: bool) -> bool:
        """Record a finished modality and report whether the limit is reached."""
        if not finished:
            return False
        self.modalities_finished += 1
        return self.should_stop()

# Slice sampling positions (8 uniformly distributed slices)
SLICE_POSITIONS = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
NUM_SLICES = len(SLICE_POSITIONS)  # 8

# Tumor existence threshold (minimum pixels to consider tumor present)
TUMOR_EXISTENCE_THRESHOLD = 10

# Data source configurations (relative to script directory)
DATA_SOURCES = {
    "brats_gli_main": {
        "input_dir": "./BraTS2024-BraTS-GLI-TrainingData/training_data1_v2",
        "output_dir": "./BraTS_GLI_Main_Image_280_MultiSlice",
        "modalities": ["t1c", "t1n", "t2f", "t2w"],
        "modality_names": ["T1C", "T1N", "T2F", "T2W"],
        "seg_suffix": "-seg.nii.gz",
        "pattern": "BraTS-GLI-*",
        "type": "brats",
    },
    "brats_gli_additional": {
        "input_dir": "./BraTS2024-BraTS-GLI-AdditionalTrainingData/training_data_additional",
        "output_dir": "./BraTS_GLI_Additional_Image_280_MultiSlice",
        "modalities": ["t1c", "t1n", "t2f", "t2w"],
        "modality_names": ["T1C", "T1N", "T2F", "T2W"],
        "seg_suffix": "-seg.nii.gz",
        "pattern": "BraTS-GLI-*",
        "type": "brats",
    },
    "menrt": {
        "input_dir": "./BraTS2024-MEN-RT-TrainingData/BraTS-MEN-RT-Train-v2",
        "output_dir": "./BraTS_MEN_RT_Image_280_MultiSlice",
        "modalities": ["t1c"],
        "modality_names": ["T1C"],
        "seg_suffix": "_gtv.nii.gz",
        "pattern": "BraTS-MEN-RT-*",
        "type": "menrt",
    },
    "msd": {
        "input_images": "./MSD_Task01_BrainTumour/imagesTr",
        "input_labels": "./MSD_Task01_BrainTumour/labelsTr",
        "output_dir": "./MSD_BrainTumour_Image_280_MultiSlice",
        "modalities": ["flair", "t1w", "t1gd", "t2w"],
        "modality_names": ["FLAIR", "T1w", "T1Gd", "T2w"],
        "pattern": "BRATS_*.nii.gz",
        "type": "msd_4d",
    },
}


# ==================== Core Functions ====================


def normalize_slice(slice_data: np.ndarray, percentile: int = 99) -> np.ndarray:
    """Normalize slice to 0-255 uint8 using robust percentile method"""
    if slice_data.max() == slice_data.min():
        return np.zeros_like(slice_data, dtype=np.uint8)

    # Use brain tissue (non-zero) for normalization
    brain_mask = slice_data > slice_data.mean() * 0.1  # Filter out background noise
    brain_values = slice_data[brain_mask]

    if len(brain_values) > 0:
        # Use 1st and 99th percentile for robust normalization
        vmin = np.percentile(brain_values, 1)
        vmax = np.percentile(brain_values, percentile)

        if vmax <= vmin:
            # Fallback to global min/max
            vmin = slice_data.min()
            vmax = slice_data.max()

        if vmax > vmin:
            normalized = np.clip((slice_data - vmin) / (vmax - vmin), 0, 1)
            return (normalized * 255).astype(np.uint8)

    # Fallback: simple min-max normalization
    vmin, vmax = slice_data.min(), slice_data.max()
    if vmax > vmin:
        normalized = (slice_data - vmin) / (vmax - vmin)
        return (normalized * 255).astype(np.uint8)

    return np.zeros_like(slice_data, dtype=np.uint8)


def get_uniform_slice_indices(total_depth: int) -> List[int]:
    """
    Get 8 uniformly distributed slice indices from volume

    Args:
        total_depth: Total number of slices in the volume (z-dimension)

    Returns:
        List of 8 slice indices
    """
    slice_indices = [int(total_depth * pos) for pos in SLICE_POSITIONS]
    # Ensure indices are within valid range [0, total_depth-1]
    slice_indices = [min(max(idx, 0), total_depth - 1) for idx in slice_indices]
    return slice_indices


def check_tumor_existence(
    seg_slice: np.ndarray, threshold: int = TUMOR_EXISTENCE_THRESHOLD
) -> bool:
    """
    Check if tumor exists in a 2D slice

    Args:
        seg_slice: 2D segmentation mask (H, W)
        threshold: Minimum number of tumor pixels to consider existence

    Returns:
        True if tumor pixels > threshold, False otherwise
    """
    tumor_pixels = (seg_slice > 0).sum()
    return tumor_pixels > threshold


def extract_bbox_and_point_from_mask(
    mask_slice: np.ndarray,
) -> Optional[Tuple[List[int], List[int]]]:
    """
    Extract 2D bounding box and center point from binary mask

    Args:
        mask_slice: 2D binary mask (H, W)

    Returns:
        bbox_2d: [x1, y1, x2, y2]
        point_2d: [center_x, center_y]
    """
    tumor_pixels = mask_slice > 0

    if not tumor_pixels.any():
        return None

    coords = np.where(tumor_pixels)
    y_coords, x_coords = coords[0], coords[1]  # Note: numpy returns (row, col) = (y, x)

    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    bbox_2d = [int(x_min), int(y_min), int(x_max), int(y_max)]

    # Calculate center point
    center_x = int((x_min + x_max) / 2)
    center_y = int((y_min + y_max) / 2)
    point_2d = [center_x, center_y]

    return bbox_2d, point_2d


def scale_bbox_and_point(
    bbox_2d: List[int],
    point_2d: List[int],
    orig_size: Tuple[int, int],
    target_size: int,
) -> Tuple[List[int], List[int]]:
    """
    Scale bbox and point coordinates to target size

    Args:
        bbox_2d: [x1, y1, x2, y2] in original coordinates
        point_2d: [x, y] in original coordinates
        orig_size: (orig_height, orig_width)
        target_size: target dimension (square)

    Returns:
        scaled_bbox_2d, scaled_point_2d
    """
    orig_h, orig_w = orig_size

    scale_x = target_size / orig_w
    scale_y = target_size / orig_h

    # Scale bbox
    x1, y1, x2, y2 = bbox_2d
    scaled_bbox = [
        int(x1 * scale_x + 0.5),
        int(y1 * scale_y + 0.5),
        int(x2 * scale_x + 0.5),
        int(y2 * scale_y + 0.5),
    ]

    # Scale point
    px, py = point_2d
    scaled_point = [int(px * scale_x + 0.5), int(py * scale_y + 0.5)]

    return scaled_bbox, scaled_point


def extract_slice_image(
    modality_data: np.ndarray, slice_idx: int, target_size: int
) -> Image.Image:
    """Extract and resize a single slice to target size"""
    slice_data = modality_data[:, :, slice_idx]
    normalized = normalize_slice(slice_data)

    # Convert to PIL Image and resize
    pil_img = Image.fromarray(normalized, "L")  # Grayscale
    pil_img = pil_img.resize((target_size, target_size), Image.LANCZOS)

    # Convert to RGB (for consistency with Seg-Zero dataset)
    pil_img_rgb = pil_img.convert("RGB")

    return pil_img_rgb


# ==================== Data Source Handlers ====================


def process_brats_case(
    case_dir: Path, modality: str, modality_name: str, seg_suffix: str
) -> Optional[List[Dict]]:
    """
    Process BraTS-GLI or MEN-RT case - Extract 8 uniformly distributed slices

    Returns:
        List of 8 dicts (one per slice), or None if case cannot be processed
    """
    case_name = case_dir.name

    try:
        # Load segmentation
        seg_file = case_dir / f"{case_name}{seg_suffix}"
        if not seg_file.exists():
            return None

        seg_data = nib.load(seg_file).get_fdata()
        total_depth = seg_data.shape[2]  # Z dimension

        # Get 8 uniformly distributed slice indices
        slice_indices = get_uniform_slice_indices(total_depth)

        # Load modality data
        modality_file = (
            case_dir / f"{case_name}-{modality}.nii.gz"
            if "-" in seg_suffix
            else case_dir / f"{case_name}_{modality}.nii.gz"
        )

        if not modality_file.exists():
            return None

        modality_data = nib.load(modality_file).get_fdata()

        # Create case ID
        case_id = f"{case_name}-{modality}"

        # Process each slice
        samples = []
        for slice_idx, z_pos in enumerate(slice_indices):
            # Extract segmentation mask for this slice
            seg_slice = seg_data[:, :, z_pos]
            orig_shape = seg_slice.shape  # (H, W)

            # Check if tumor exists in this slice
            has_tumor = check_tumor_existence(seg_slice)

            # Extract image
            slice_image = extract_slice_image(modality_data, z_pos, TARGET_SIZE)

            # Create solution based on tumor existence
            if has_tumor:
                result = extract_bbox_and_point_from_mask(seg_slice)
                if result is not None:
                    bbox_orig, point_orig = result
                    bbox_scaled, point_scaled = scale_bbox_and_point(
                        bbox_orig, point_orig, orig_shape, TARGET_SIZE
                    )
                    solution = [{"bbox_2d": bbox_scaled, "point_2d": point_scaled}]
                else:
                    # Mask extraction failed, treat as no tumor
                    has_tumor = False
                    solution = [{"bbox_2d": [0, 0, 1, 1], "point_2d": [0, 0]}]
            else:
                # No tumor: use degenerate bbox (area ≈ 0) at top-left corner
                # This provides a unified format and enables continuous reward from area
                # Area threshold in reward function will distinguish tumor vs no-tumor
                solution = [{"bbox_2d": [0, 0, 1, 1], "point_2d": [0, 0]}]

            # Create problem prompt (varies by slice to add diversity)
            if has_tumor:
                problem_templates = [
                    f"Does this {modality_name} MRI slice contain brain tumor? If yes, locate it.",
                    f"Analyze this {modality_name} brain MRI. Identify tumor location if present.",
                    f"Check for brain tumor in this {modality_name} image and provide bbox if found.",
                    f"Examine this {modality_name} MRI slice for tumor presence and location.",
                ]
            else:
                problem_templates = [
                    f"Does this {modality_name} MRI slice contain brain tumor? If yes, locate it.",
                    f"Analyze this {modality_name} brain MRI. Identify tumor location if present.",
                    f"Check for brain tumor in this {modality_name} image and provide bbox if found.",
                    f"Examine this {modality_name} MRI slice for tumor presence and location.",
                ]

            problem = problem_templates[
                hash(f"{case_name}-{slice_idx}") % len(problem_templates)
            ]

            # Create sample dict
            sample = {
                "id": f"{case_id}-slice{slice_idx}",
                "case_id": case_id,
                "slice_idx": slice_idx,
                "slice_position": SLICE_POSITIONS[slice_idx],
                "problem": problem,
                "has_tumor": has_tumor,
                "solution": json.dumps(solution),
                "image": slice_image,
                "img_width": TARGET_SIZE,
                "img_height": TARGET_SIZE,
            }
            samples.append(sample)

        return samples

    except Exception as e:
        print(f"Error processing {case_name}-{modality}: {e}")
        return None


def process_msd_case(
    case_name: str,
    modality_idx: int,
    modality_name: str,
    images_dir: Path,
    labels_dir: Path,
) -> Optional[List[Dict]]:
    """
    Process MSD case (4D data) - Extract 8 uniformly distributed slices

    Returns:
        List of 8 dicts (one per slice), or None if case cannot be processed
    """
    try:
        # Load segmentation
        seg_file = labels_dir / f"{case_name}.nii.gz"
        if not seg_file.exists():
            return None

        seg_data = nib.load(seg_file).get_fdata()
        total_depth = seg_data.shape[2]  # Z dimension

        # Get 8 uniformly distributed slice indices
        slice_indices = get_uniform_slice_indices(total_depth)

        # Load 4D image data and extract modality
        img_file = images_dir / f"{case_name}.nii.gz"
        if not img_file.exists():
            return None

        img_4d = nib.load(img_file).get_fdata()  # (H, W, D, 4)
        modality_data = img_4d[:, :, :, modality_idx]

        # Create case ID
        case_id = f"{case_name}-{modality_name.lower()}"

        # Process each slice
        samples = []
        for slice_idx, z_pos in enumerate(slice_indices):
            # Extract segmentation mask for this slice
            seg_slice = seg_data[:, :, z_pos]
            orig_shape = seg_slice.shape  # (H, W)

            # Check if tumor exists in this slice
            has_tumor = check_tumor_existence(seg_slice)

            # Extract image
            slice_image = extract_slice_image(modality_data, z_pos, TARGET_SIZE)

            # Create solution based on tumor existence
            if has_tumor:
                result = extract_bbox_and_point_from_mask(seg_slice)
                if result is not None:
                    bbox_orig, point_orig = result
                    bbox_scaled, point_scaled = scale_bbox_and_point(
                        bbox_orig, point_orig, orig_shape, TARGET_SIZE
                    )
                    solution = [{"bbox_2d": bbox_scaled, "point_2d": point_scaled}]
                else:
                    # Mask extraction failed, treat as no tumor
                    has_tumor = False
                    solution = [{"bbox_2d": [0, 0, 1, 1], "point_2d": [0, 0]}]
            else:
                # No tumor: use degenerate bbox (unified format)
                solution = [{"bbox_2d": [0, 0, 1, 1], "point_2d": [0, 0]}]

            # Create problem prompt (same templates as BraTS)
            problem_templates = [
                f"Does this {modality_name} MRI slice contain brain tumor? If yes, locate it.",
                f"Analyze this {modality_name} brain MRI. Identify tumor location if present.",
                f"Check for brain tumor in this {modality_name} image and provide bbox if found.",
                f"Examine this {modality_name} MRI slice for tumor presence and location.",
            ]
            problem = problem_templates[
                hash(f"{case_name}-{slice_idx}") % len(problem_templates)
            ]

            # Create sample dict
            sample = {
                "id": f"{case_id}-slice{slice_idx}",
                "case_id": case_id,
                "slice_idx": slice_idx,
                "slice_position": SLICE_POSITIONS[slice_idx],
                "problem": problem,
                "has_tumor": has_tumor,
                "solution": json.dumps(solution),
                "image": slice_image,
                "img_width": TARGET_SIZE,
                "img_height": TARGET_SIZE,
            }
            samples.append(sample)

        return samples

    except Exception as e:
        print(f"Error processing {case_name}-{modality_name}: {e}")
        return None


# ==================== Multi-processing Wrappers ====================


def process_brats_worker(args):
    """Worker function for BraTS cases"""
    case_dir, modality, modality_name, seg_suffix = args
    result = process_brats_case(case_dir, modality, modality_name, seg_suffix)
    gc.collect()  # Force garbage collection after each case
    return result


def process_msd_worker(args):
    """Worker function for MSD cases"""
    case_name, modality_idx, modality_name, images_dir, labels_dir = args
    result = process_msd_case(
        case_name, modality_idx, modality_name, images_dir, labels_dir
    )
    gc.collect()  # Force garbage collection after each case
    return result


# ==================== Main Processing Pipeline ====================


def check_dataset_exists(output_dir: Path, modality_name: str) -> bool:
    """
    Check if dataset already exists and is complete

    Returns:
        True if dataset exists with all 8 shards, False otherwise
    """
    modality_dir = output_dir / modality_name
    train_dir = modality_dir / "train"

    # Check if directories exist
    if not train_dir.exists():
        return False

    # Check for all 8 arrow files
    expected_shards = [
        f"data-{i:05d}-of-{NUM_SLICES:05d}.arrow"
        for i in range(NUM_SLICES)
    ]

    for shard in expected_shards:
        if not (train_dir / shard).exists():
            return False

    # Check for metadata files
    required_files = [
        modality_dir / "dataset_dict.json",
        train_dir / "dataset_info.json",
        train_dir / "state.json"
    ]

    for file_path in required_files:
        if not file_path.exists():
            return False

    return True


def process_data_source(source_name: str, config: Dict, state: ProcessingState):
    """Process a single data source"""
    print(f"\n{'='*80}")
    print(f"🚀 Processing {source_name.upper()}")
    print(f"{'='*80}")

    if state.should_stop():
        raise StopProcessing

    source_type = config["type"]
    modalities = config["modalities"]
    modality_names = config["modality_names"]

    # Prepare tasks based on source type
    if source_type in ["brats", "menrt"]:
        # Resolve path relative to script directory
        input_path = SCRIPT_DIR / config["input_dir"]
        case_dirs = [
            d
            for d in input_path.iterdir()
            if d.is_dir() and d.name.startswith(config["pattern"].rstrip("*"))
        ]

        if not case_dirs:
            print(f"❌ No cases found matching pattern {config['pattern']}")
            return

        print(f"📊 Found {len(case_dirs)} cases")
        print(
            f"📊 Processing {len(modalities)} modalities: {', '.join(modality_names)}"
        )

        # Process each modality
        for modality, modality_name in zip(modalities, modality_names):
            if state.should_stop():
                raise StopProcessing

            output_dir = SCRIPT_DIR / config["output_dir"]

            # Check if dataset already exists (resume support)
            if check_dataset_exists(output_dir, modality_name):
                print(f"\n⏭️  Skipping {modality_name} (already exists)")
                continue

            print(f"\n🔄 Processing {modality_name}...")

            tasks = [
                (case_dir, modality, modality_name, config["seg_suffix"])
                for case_dir in case_dirs
            ]

            writer = IncrementalSliceDatasetWriter(str(output_dir), modality_name)
            valid_cases = 0

            if NUM_WORKERS == 1:
                for task in tqdm(tasks, desc=f"  {modality_name}"):
                    result = process_brats_worker(task)
                    if result:
                        writer.write_samples(result)
                        valid_cases += 1
            else:
                # maxtasksperchild: restart workers every 200 tasks to prevent memory leaks
                with mp.Pool(NUM_WORKERS, maxtasksperchild=200) as pool:
                    # Use iterator directly to avoid loading all results at once
                    # chunksize=1: process one task at a time to minimize memory usage
                    for result in tqdm(
                        pool.imap(process_brats_worker, tasks, chunksize=1),
                        total=len(tasks),
                        desc=f"  {modality_name}",
                    ):
                        if result:
                            writer.write_samples(result)
                            valid_cases += 1

            summary = writer.finalize()

            if summary["total_samples"] > 0:
                print(
                    f"  ✅ {modality_name}: {valid_cases} cases × {NUM_SLICES} slices = {summary['total_samples']} samples"
                )
            else:
                print(f"  ⚠️  {modality_name}: No valid samples")

            gc.collect()

            if state.register_modality(writer.has_written):
                raise StopProcessing

    elif source_type == "msd_4d":
        # Resolve paths relative to script directory
        images_dir = SCRIPT_DIR / config["input_images"]
        labels_dir = SCRIPT_DIR / config["input_labels"]

        image_files = sorted(images_dir.glob(config["pattern"]))
        case_names = [f.stem.replace(".nii", "") for f in image_files]

        if not case_names:
            print(f"❌ No cases found matching pattern {config['pattern']}")
            return

        print(f"📊 Found {len(case_names)} cases")
        print(
            f"📊 Processing {len(modalities)} modalities: {', '.join(modality_names)}"
        )

        # Process each modality
        for modality_idx, modality_name in enumerate(modality_names):
            if state.should_stop():
                raise StopProcessing

            output_dir = SCRIPT_DIR / config["output_dir"]

            # Check if dataset already exists (resume support)
            if check_dataset_exists(output_dir, modality_name):
                print(f"\n⏭️  Skipping {modality_name} (already exists)")
                continue

            print(f"\n🔄 Processing {modality_name} (channel {modality_idx})...")

            tasks = [
                (case_name, modality_idx, modality_name, images_dir, labels_dir)
                for case_name in case_names
            ]

            writer = IncrementalSliceDatasetWriter(str(output_dir), modality_name)
            valid_cases = 0

            if NUM_WORKERS == 1:
                for task in tqdm(tasks, desc=f"  {modality_name}"):
                    result = process_msd_worker(task)
                    if result:
                        writer.write_samples(result)
                        valid_cases += 1
            else:
                # maxtasksperchild: restart workers every 200 tasks to prevent memory leaks
                with mp.Pool(NUM_WORKERS, maxtasksperchild=200) as pool:
                    # Use iterator directly to avoid loading all results at once
                    # chunksize=1: process one task at a time to minimize memory usage
                    for result in tqdm(
                        pool.imap(process_msd_worker, tasks, chunksize=1),
                        total=len(tasks),
                        desc=f"  {modality_name}",
                    ):
                        if result:
                            writer.write_samples(result)
                            valid_cases += 1

            summary = writer.finalize()

            if summary["total_samples"] > 0:
                print(
                    f"  ✅ {modality_name}: {valid_cases} cases × {NUM_SLICES} slices = {summary['total_samples']} samples"
                )
            else:
                print(f"  ⚠️  {modality_name}: No valid samples")

            gc.collect()

            if state.register_modality(writer.has_written):
                raise StopProcessing


class IncrementalSliceDatasetWriter:
    """Incremental writer that streams samples to Arrow shards to keep memory usage low."""

    def __init__(self, output_base: str, modality_name: str):
        self.output_dir = Path(output_base) / modality_name
        self.train_dir = self.output_dir / "train"
        self.train_dir.mkdir(parents=True, exist_ok=True)

        print(f"    💾 Saving to: {self.output_dir}")

        self.features = Features(
            {
                "id": Value("string"),
                "case_id": Value("string"),
                "slice_idx": Value("int64"),
                "slice_position": Value("float32"),
                "problem": Value("string"),
                "has_tumor": Value("bool"),
                "solution": Value("string"),
                "image": DatasetImage(),
                "img_width": Value("int64"),
                "img_height": Value("int64"),
            }
        )

        self.slice_stats = [
            {
                "tumor": 0,
                "no_tumor": 0,
                "num_examples": 0,
                "size_bytes": 0,
                "shard_name": f"data-{idx:05d}-of-{NUM_SLICES:05d}.arrow",
            }
            for idx in range(NUM_SLICES)
        ]

        self.writers = {
            idx: ArrowWriter(
                features=self.features,
                path=str(self.train_dir / stats["shard_name"]),
                writer_batch_size=32,
            )
            for idx, stats in enumerate(self.slice_stats)
        }

        self.total_bytes = 0
        self.total_samples = 0
        self.has_written = False

    def write_samples(self, samples: List[Dict]):
        """Append a batch of samples returned by a worker."""
        if not samples:
            return

        self.has_written = True

        for sample in samples:
            slice_idx = sample["slice_idx"]
            self.writers[slice_idx].write(sample)

            if sample["has_tumor"]:
                self.slice_stats[slice_idx]["tumor"] += 1
            else:
                self.slice_stats[slice_idx]["no_tumor"] += 1

    def finalize(self) -> Dict[str, Any]:
        """Finalize all shard writers and write dataset metadata."""
        shard_files: List[str] = []
        total_bytes = 0
        total_samples = 0

        for slice_idx, stats in enumerate(self.slice_stats):
            num_examples, num_bytes = self.writers[slice_idx].finalize()
            stats["num_examples"] = num_examples
            stats["size_bytes"] = num_bytes

            if num_examples > 0:
                shard_files.append(stats["shard_name"])

            total_bytes += num_bytes
            total_samples += num_examples

        self.total_bytes = total_bytes
        self.total_samples = total_samples

        self._write_metadata(shard_files, total_bytes, total_samples)

        for slice_idx, stats in enumerate(self.slice_stats):
            num_examples = stats["num_examples"]
            if num_examples == 0:
                print(f"    ⚠️  Slice {slice_idx}: No samples")
                continue

            shard_size_mb = stats["size_bytes"] / (1024 * 1024)
            num_tumor = stats["tumor"]
            num_no_tumor = stats["no_tumor"]

            print(
                f"    📊 Slice {slice_idx} (pos={SLICE_POSITIONS[slice_idx]:.2f}): "
                f"{num_examples} samples "
                f"(✓ {num_tumor} tumor, ✗ {num_no_tumor} no-tumor) - "
                f"{shard_size_mb:.1f} MB"
            )

        total_size_mb = total_bytes / (1024 * 1024)
        print(f"    📦 Total: {total_samples} samples, {total_size_mb:.2f} MB")
        print(
            f"    📁 Shards: {len(shard_files)} files (data-00000 to data-{NUM_SLICES-1:05d})"
        )

        return {
            "shard_files": shard_files,
            "total_samples": total_samples,
            "total_size_mb": total_size_mb,
        }

    def _write_metadata(
        self, shard_files: List[str], total_bytes: int, total_samples: int
    ) -> None:
        """Write dataset metadata files expected by the Seg-Zero tooling."""
        with open(self.output_dir / "dataset_dict.json", "w") as f:
            json.dump({"splits": ["train"]}, f)

        dataset_info = {
            "features": {
                "id": {"dtype": "string", "_type": "Value"},
                "case_id": {"dtype": "string", "_type": "Value"},
                "slice_idx": {"dtype": "int64", "_type": "Value"},
                "slice_position": {"dtype": "float32", "_type": "Value"},
                "problem": {"dtype": "string", "_type": "Value"},
                "has_tumor": {"dtype": "bool", "_type": "Value"},
                "solution": {"dtype": "string", "_type": "Value"},
                "image": {"_type": "Image"},
                "img_width": {"dtype": "int64", "_type": "Value"},
                "img_height": {"dtype": "int64", "_type": "Value"},
            },
            "splits": {
                "train": {
                    "name": "train",
                    "num_bytes": int(total_bytes),
                    "num_examples": int(total_samples),
                }
            },
        }

        with open(self.train_dir / "dataset_info.json", "w") as f:
            json.dump(dataset_info, f, indent=2)

        with open(self.train_dir / "state.json", "w") as f:
            json.dump({"_data_files": [{"filename": f} for f in shard_files]}, f, indent=2)


def _normalize_limit(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    if value <= 0:
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified Brain Tumor Image Dataset Generator"
    )

    env_limit = os.environ.get("DATASET_MODALITIES_PER_RUN")
    default_limit = 1
    if env_limit is not None:
        try:
            default_limit = int(env_limit)
        except ValueError:
            default_limit = 1
    default_limit = default_limit if default_limit > 0 else None

    parser.add_argument(
        "--modalities-per-run",
        type=int,
        default=default_limit,
        help="每次运行最多处理的模态数（>0 限制；<=0 表示不限）。默认读取 DATASET_MODALITIES_PER_RUN 或 1。",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="覆盖默认的并行进程数。",
    )

    args = parser.parse_args()
    args.modalities_per_run = _normalize_limit(args.modalities_per_run)
    return args


def main():
    """Main entry point"""
    args = parse_args()

    global NUM_WORKERS
    if args.num_workers is not None:
        NUM_WORKERS = max(1, args.num_workers)

    state = ProcessingState(limit=args.modalities_per_run)

    print("🚀 Unified Brain Tumor Image Dataset Generator (Multi-Slice Version)")
    print("=" * 80)
    print(f"Target size: {TARGET_SIZE}×{TARGET_SIZE}")
    print(
        f"Sampling strategy: {NUM_SLICES} uniform slices at positions {SLICE_POSITIONS}"
    )
    print(f"Output format: Seg-Zero compatible with tumor existence labels")
    print(f"Workers: {NUM_WORKERS}")
    if state.limit is not None:
        print(f"Modalities per run: {state.limit}")
    print()

    try:
        for source_name, config in DATA_SOURCES.items():
            try:
                process_data_source(source_name, config, state)
            except StopProcessing:
                raise
            except Exception as e:
                print(f"❌ Error processing {source_name}: {e}")
                continue
    except StopProcessing:
        print("\n🛑 已达到每次运行的模态处理上限，退出以便下次续跑。")
        return

    print(f"\n{'='*80}")
    print("🎉 ALL DATA SOURCES PROCESSED")
    print(f"{'='*80}")
    print("\n✅ Datasets ready for training!")
    print("   Format: Seg-Zero compatible with existence detection")
    print("   Fields:")
    print("     - id: Unique sample identifier")
    print("     - case_id: Parent case identifier")
    print("     - slice_idx: Slice position index (0-7)")
    print("     - slice_position: Normalized position in volume (0.15-0.85)")
    print("     - problem: Task description")
    print("     - has_tumor: Boolean indicating tumor presence")
    print("     - solution: JSON with bbox_2d and point_2d (if has_tumor=True)")
    print("     - image: 280×280 RGB slice")
    print("   Storage: Each slice position saved in separate subdirectory")


if __name__ == "__main__":
    main()
