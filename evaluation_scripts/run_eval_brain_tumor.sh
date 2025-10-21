#!/bin/bash
# Brain Tumor Evaluation Script
# Evaluates model performance on BraTS GLI Additional validation datasets

set -e

# ============================================================================
# Configuration
# ============================================================================

# Model path - default to base model (change to checkpoint path for trained model)
MODEL_PATH="${MODEL_PATH:-models/Qwen3-VL-8B-Thinking}"

# Validation datasets
DATASETS=(
    "data/BraTS_GLI_Additional_Image_280/T1C"
    "data/BraTS_GLI_Additional_Image_280/T2F"
)

# Output directory
OUTPUT_DIR="eval_results/$(basename $MODEL_PATH)_$(date +%Y%m%d_%H%M%S)"

# Evaluation parameters
BATCH_SIZE=8
IMAGE_SIZE=280
MAX_SAMPLES=""  # Set to a number for quick testing, e.g., MAX_SAMPLES=50

# GPU configuration
export CUDA_VISIBLE_DEVICES=0

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

print_header "Brain Tumor Detection Evaluation"

print_info "Model: $MODEL_PATH"
print_info "Output directory: $OUTPUT_DIR"
print_info "Batch size: $BATCH_SIZE"
print_info "Image size: ${IMAGE_SIZE}x${IMAGE_SIZE}"
print_info "GPU: $CUDA_VISIBLE_DEVICES"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model path does not exist: $MODEL_PATH"
    echo "Please set MODEL_PATH to a valid model directory, e.g.:"
    echo "  export MODEL_PATH=models/Qwen3-VL-8B-Thinking"
    echo "  export MODEL_PATH=brain_tumor_workdir/run_brain_tumor_image_4x80G/global_step_200/actor/default_0"
    exit 1
fi

# Evaluate each dataset
for DATA_PATH in "${DATASETS[@]}"; do
    DATASET_NAME=$(basename $(dirname $DATA_PATH))/$(basename $DATA_PATH)
    OUTPUT_PATH="$OUTPUT_DIR/$DATASET_NAME"

    print_header "Evaluating: $DATASET_NAME"

    # Check if dataset exists
    if [ ! -d "$DATA_PATH" ]; then
        print_info "WARNING: Dataset not found: $DATA_PATH (skipping)"
        continue
    fi

    # Build command
    CMD="python evaluation_scripts/eval_brain_tumor.py \
        --model_path $MODEL_PATH \
        --data_path $DATA_PATH \
        --output_path $OUTPUT_PATH \
        --batch_size $BATCH_SIZE \
        --image_size $IMAGE_SIZE"

    # Add max_samples if specified
    if [ -n "$MAX_SAMPLES" ]; then
        CMD="$CMD --max_samples $MAX_SAMPLES"
    fi

    # Run evaluation
    print_info "Running: $CMD"
    eval $CMD

    # Extract key metrics
    if [ -f "$OUTPUT_PATH/evaluation_results.json" ]; then
        print_info "Extracting metrics..."
        python3 -c "
import json
import sys

with open('$OUTPUT_PATH/evaluation_results.json') as f:
    data = json.load(f)

summary = data['summary']
acc = summary['accuracy_metrics']

print()
print('Quick Summary for $DATASET_NAME:')
print(f\"  Samples: {summary['total_samples']}\")
print(f\"  Mean IoU: {acc['bbox_iou_mean']:.3f}\")
print(f\"  IoU > 0.5: {acc['bbox_iou_gt_0.5']*100:.1f}%\")
print(f\"  IoU > 0.7: {acc['bbox_iou_gt_0.7']*100:.1f}%\")
print(f\"  Point dist: {acc['point_distance_mean']:.1f} pixels\")
"
    fi
done

# ============================================================================
# Summary
# ============================================================================

print_header "Evaluation Complete"

print_info "All results saved to: $OUTPUT_DIR"
print_info ""
print_info "To view detailed results:"
echo "  cat $OUTPUT_DIR/*/evaluation_results.json | jq '.summary'"
print_info ""
print_info "To evaluate a trained checkpoint instead of base model:"
echo "  export MODEL_PATH=brain_tumor_workdir/run_brain_tumor_image_4x80G/global_step_200/actor/default_0"
echo "  bash evaluation_scripts/run_eval_brain_tumor.sh"

print_header "Done"
