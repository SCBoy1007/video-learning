#!/bin/bash

# Brain Tumor Image Training - 4×A100-80G Configuration
# Based on Seg-Zero architecture with brain tumor image datasets

export CUDA_VISIBLE_DEVICES=0,1,2,3

set -x

# Debug settings - enable detailed logging for troubleshooting
export RAY_DEDUP_LOGS=0  # Show all Ray logs to debug distributed issues
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export VLLM_LOGGING_LEVEL=INFO  # Changed from WARNING to INFO for more details
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1  # Ensure Python output is not buffered

# WORKAROUND: Disable vLLM prefix caching to avoid multimodal cache corruption bug
# This prevents AssertionError crashes at step 117-118
# See: https://github.com/vllm-project/vllm/issues/20261
export VLLM_USE_MODELSCOPE=false
export VLLM_ENABLE_PREFIX_CACHING=0

MODEL_PATH=/root/Documents/video-learning/models/Qwen3-VL-8B-Thinking

RUN_NAME=$(basename "$0" .sh)

# Training datasets: 2 BraTS datasets (8 slices per sample)
# - BraTS GLI Main: T1C (1350 samples)
# - BraTS GLI Additional: T1C (273 samples)
# Total: ~1,623 multi-image training samples (each with 8 slices)
TRAIN_DATA="data/BraTS_GLI_Main_Image_280_MultiImage/T1C,\
data/BraTS_GLI_Additional_Image_280_MultiImage/T1C"

# Validation datasets: MSD dataset (8 slices per sample)
# - MSD Brain Tumor: T1Gd (484 samples)
# Total: ~484 multi-image validation samples (each with 8 slices)
VAL_DATA="data/MSD_BrainTumour_Image_280_MultiImage/T1Gd"

python3 -m verl.trainer.main \
    config=training_scripts/brain_tumor_image_4x80G.yaml \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${VAL_DATA} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.micro_batch_size_per_device_for_update=4 \
    worker.actor.micro_batch_size_per_device_for_experience=4 \
    worker.actor.kl_loss_coef=1.0e-2 \
    worker.actor.optim.lr=1.0e-5 \
    worker.actor.max_grad_norm=5.0 \
    worker.reward.compute_score=vision_reasoner \
    trainer.experiment_name=${RUN_NAME} \
    trainer.save_checkpoint_path=brain_tumor_workdir/${RUN_NAME}
