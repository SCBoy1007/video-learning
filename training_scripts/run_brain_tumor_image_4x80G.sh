#!/bin/bash

# Brain Tumor Image Training - 4×A100-80G Configuration
# Based on Seg-Zero architecture with brain tumor image datasets

export CUDA_VISIBLE_DEVICES=0,1,2,3

set -x

# Reduce log verbosity
export VLLM_ATTENTION_BACKEND=XFORMERS
export RAY_DEDUP_LOGS=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export VLLM_LOGGING_LEVEL=WARNING
export TOKENIZERS_PARALLELISM=false

MODEL_PATH=pretrained_models/Qwen2.5-VL-7B-Instruct

RUN_NAME=$(basename "$0" .sh)

# Training datasets: 5 image datasets
# - BraTS GLI Main: T1C (1350 samples) + T2F (1350 samples)
# - MSD Brain Tumor: T1Gd (484 samples) + FLAIR (484 samples)
# - MEN-RT: T1C (500 samples)
# Total: ~4,168 training samples
TRAIN_DATA="data/BraTS_GLI_Main_Image_280/T1C,\
data/BraTS_GLI_Main_Image_280/T2F,\
data/MSD_BrainTumour_Image_280/T1Gd,\
data/MSD_BrainTumour_Image_280/FLAIR,\
data/BraTS_MEN_RT_Image_280/T1C"

# Validation datasets: 2 GLI Additional image datasets
# - BraTS GLI Additional: T1C (273 samples) + T2F (273 samples)
# Total: ~546 validation samples
VAL_DATA="data/BraTS_GLI_Additional_Image_280/T1C,\
data/BraTS_GLI_Additional_Image_280/T2F"

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
