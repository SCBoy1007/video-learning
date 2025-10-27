#!/bin/bash
# Evaluate FSDP checkpoint directly without conversion
# Usage: bash evaluation_scripts/run_eval_checkpoint.sh [step_number]

set -e

# ============================================================================
# Configuration
# ============================================================================

# Get step number from argument or default to 200
STEP=${1:-200}

# FSDP checkpoint path (parent directory containing the FSDP shards)
CHECKPOINT_PATH="brain_tumor_workdir/run_brain_tumor_image_4x80G/global_step_${STEP}/actor"

# Validation datasets
DATASETS=(
    "data/BraTS_GLI_Additional_Image_280/T1C"
    "data/BraTS_GLI_Additional_Image_280/T2F"
)

# Output directory
OUTPUT_DIR="eval_results/step${STEP}_$(date +%Y%m%d_%H%M%S)"

# Evaluation parameters
BATCH_SIZE=16
IMAGE_SIZE=280
MAX_SAMPLES=""  # Set to a number for quick testing

# GPU configuration - use all 4 GPUs in parallel
NUM_GPUS=4
GPUS=(0 1 2 3)

# ============================================================================
# Functions
# ============================================================================

print_header() {
    echo ""
    echo "=========================================================================="
    echo "$1"
    echo "=========================================================================="
}

print_info() {
    echo "[INFO] $1"
}

# ============================================================================
# Main Execution
# ============================================================================

print_header "Brain Tumor Detection - Checkpoint Evaluation (Step ${STEP})"

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint path does not exist: $CHECKPOINT_PATH"
    echo ""
    echo "Available checkpoints:"
    ls -d brain_tumor_workdir/run_brain_tumor_image_4x80G/global_step_*/actor 2>/dev/null || echo "  No checkpoints found"
    echo ""
    echo "Usage: bash evaluation_scripts/run_eval_checkpoint.sh [step_number]"
    echo "Example: bash evaluation_scripts/run_eval_checkpoint.sh 200"
    exit 1
fi

print_info "Checkpoint: $CHECKPOINT_PATH"
print_info "Output directory: $OUTPUT_DIR"
print_info "Batch size: $BATCH_SIZE"
print_info "Image size: ${IMAGE_SIZE}x${IMAGE_SIZE}"
print_info "GPUs: ${GPUS[@]} (${NUM_GPUS} GPUs in parallel)"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run evaluation on both datasets in parallel using different GPUs
print_header "Starting Parallel Evaluation"

# Array to store background process IDs
PIDS=()

# Launch evaluation for each dataset on a different GPU
for i in "${!DATASETS[@]}"; do
    DATA_PATH="${DATASETS[$i]}"
    GPU_ID="${GPUS[$i]}"
    DATASET_NAME=$(basename $(dirname $DATA_PATH))/$(basename $DATA_PATH)
    OUTPUT_PATH="$OUTPUT_DIR/$DATASET_NAME"

    print_info "Launching evaluation on GPU $GPU_ID: $DATASET_NAME"

    # Check if dataset exists
    if [ ! -d "$DATA_PATH" ]; then
        print_info "WARNING: Dataset not found: $DATA_PATH (skipping)"
        continue
    fi

    # Build command
    CMD="CUDA_VISIBLE_DEVICES=$GPU_ID python evaluation_scripts/eval_brain_tumor.py \
        --model_path $CHECKPOINT_PATH \
        --data_path $DATA_PATH \
        --output_path $OUTPUT_PATH \
        --batch_size $BATCH_SIZE \
        --image_size $IMAGE_SIZE"

    # Add max_samples if specified
    if [ -n "$MAX_SAMPLES" ]; then
        CMD="$CMD --max_samples $MAX_SAMPLES"
    fi

    # Create log directory
    mkdir -p "$(dirname $OUTPUT_PATH.log)"

    # Run in background
    eval $CMD > "$OUTPUT_PATH.log" 2>&1 &
    PIDS+=($!)

    print_info "  → Process ID: ${PIDS[$i]}, Log: $OUTPUT_PATH.log"
done

# Wait for all processes to complete
print_header "Waiting for Evaluations to Complete"

for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    DATASET_NAME=$(basename $(dirname ${DATASETS[$i]}))/$(basename ${DATASETS[$i]})

    print_info "Waiting for $DATASET_NAME (PID: $PID)..."

    if wait $PID; then
        print_info "  ✓ $DATASET_NAME completed successfully"
    else
        print_info "  ✗ $DATASET_NAME failed (check log for details)"
    fi
done

# ============================================================================
# Collect and Display Results
# ============================================================================

print_header "Evaluation Results Summary"

for DATA_PATH in "${DATASETS[@]}"; do
    DATASET_NAME=$(basename $(dirname $DATA_PATH))/$(basename $DATA_PATH)
    OUTPUT_PATH="$OUTPUT_DIR/$DATASET_NAME"

    if [ -f "$OUTPUT_PATH/evaluation_results.json" ]; then
        print_info "Results for $DATASET_NAME:"
        python3 -c "
import json
import sys

try:
    with open('$OUTPUT_PATH/evaluation_results.json') as f:
        data = json.load(f)

    summary = data['summary']
    acc = summary['accuracy_metrics']

    print(f\"  Samples: {summary['total_samples']}\")
    print(f\"  Mean IoU: {acc['bbox_iou_mean']:.3f}\")
    print(f\"  Median IoU: {acc['bbox_iou_median']:.3f}\")
    print(f\"  IoU > 0.5: {acc['bbox_iou_gt_0.5']*100:.1f}%\")
    print(f\"  IoU > 0.7: {acc['bbox_iou_gt_0.7']*100:.1f}%\")
    print(f\"  Point dist: {acc['point_distance_mean']:.1f} pixels\")
except Exception as e:
    print(f\"  Error reading results: {e}\")
" || print_info "  Failed to parse results"
        echo ""
    else
        print_info "$DATASET_NAME: No results found (check $OUTPUT_PATH.log)"
        echo ""
    fi
done

# Calculate overall statistics
print_header "Overall Statistics"

python3 -c "
import json
import glob
import numpy as np

result_files = glob.glob('$OUTPUT_DIR/*/evaluation_results.json')
if not result_files:
    print('No results found!')
    exit(1)

all_ious = []
all_samples = 0

for file in result_files:
    with open(file) as f:
        data = json.load(f)

    # Get detailed results
    for result in data['detailed_results']:
        all_ious.append(result['bbox_iou'])

    all_samples += data['summary']['total_samples']

if all_ious:
    print(f'Total samples evaluated: {all_samples}')
    print(f'Overall mean IoU: {np.mean(all_ious):.3f}')
    print(f'Overall median IoU: {np.median(all_ious):.3f}')
    print(f'Overall IoU > 0.5: {np.mean([iou > 0.5 for iou in all_ious])*100:.1f}%')
    print(f'Overall IoU > 0.7: {np.mean([iou > 0.7 for iou in all_ious])*100:.1f}%')
    print('')
    print('IoU Distribution:')
    bins = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
    for low, high in bins:
        count = sum(1 for iou in all_ious if low <= iou < high)
        pct = count / len(all_ious) * 100
        print(f'  {low:.1f}-{high:.1f}: {count:4d} samples ({pct:5.1f}%)')
" || echo "Failed to calculate overall statistics"

# ============================================================================
# Summary
# ============================================================================

print_header "Evaluation Complete"

print_info "Checkpoint evaluated: $CHECKPOINT_PATH"
print_info "All results saved to: $OUTPUT_DIR"
print_info ""
print_info "Individual logs:"
for DATA_PATH in "${DATASETS[@]}"; do
    DATASET_NAME=$(basename $(dirname $DATA_PATH))/$(basename $DATA_PATH)
    echo "  - $OUTPUT_DIR/$DATASET_NAME.log"
done

print_info ""
print_info "To view detailed results:"
echo "  cat $OUTPUT_DIR/*/evaluation_results.json | jq '.summary'"

print_info ""
print_info "To evaluate a different checkpoint step:"
echo "  bash evaluation_scripts/run_eval_checkpoint.sh 150"
echo "  bash evaluation_scripts/run_eval_checkpoint.sh 200"

print_header "Done"
