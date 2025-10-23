#!/bin/bash

# Brain Tumor Image Training - 4×A100-80G Configuration
# Based on Seg-Zero architecture with brain tumor image datasets

export CUDA_VISIBLE_DEVICES=0,1,2,3

set -x

# Reduce log verbosity
# Let VLLM auto-select attention backend (will use the best available)
# export VLLM_ATTENTION_BACKEND=FLASH_ATTN  # Disabled - has headdim restrictions
export RAY_DEDUP_LOGS=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export VLLM_LOGGING_LEVEL=WARNING
export TOKENIZERS_PARALLELISM=false

# WORKAROUND: Disable vLLM prefix caching to avoid multimodal cache corruption bug
# This prevents AssertionError crashes at step 117-118
export VLLM_USE_MODELSCOPE=false

MODEL_PATH=/root/Documents/video-learning/models/Qwen3-VL-8B-Thinking

RUN_NAME=$(basename "$0" .sh)

# Training datasets: 4 multi-image datasets (8 slices per sample)
# - BraTS GLI Main: T1C (1350 samples) + T2F (1350 samples)
# - MSD Brain Tumor: T1Gd (484 samples) + FLAIR (484 samples)
# Total: ~3,668 multi-image training samples (each with 8 slices)
TRAIN_DATA="data/BraTS_GLI_Main_Image_280_MultiImage/T1C,\
data/BraTS_GLI_Main_Image_280_MultiImage/T2F,\
data/MSD_BrainTumour_Image_280_MultiImage/T1Gd,\
data/MSD_BrainTumour_Image_280_MultiImage/FLAIR"

# Validation datasets: 2 GLI Additional multi-image datasets
# - BraTS GLI Additional: T1C (273 samples) + T2F (273 samples)
# Total: ~546 multi-image validation samples (each with 8 slices)
VAL_DATA="data/BraTS_GLI_Additional_Image_280_MultiImage/T1C,\
data/BraTS_GLI_Additional_Image_280_MultiImage/T2F"

python3 -m verl.trainer.main \
    config=training_scripts/brain_tumor_image_4x80G.yaml \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${VAL_DATA} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.kl_loss_coef=1.0e-2 \
    worker.actor.optim.lr=5.0e-6 \
    worker.actor.max_grad_norm=5.0 \
    worker.reward.compute_score=vision_reasoner \
    trainer.experiment_name=${RUN_NAME} \
    trainer.save_checkpoint_path=brain_tumor_workdir/${RUN_NAME}
