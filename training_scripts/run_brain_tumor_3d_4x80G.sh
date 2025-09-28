#!/bin/bash

# 4-card A100 80G configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3

set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

MODEL_PATH=pretrained_models/Qwen2.5-VL-7B-Instruct  # replace it with your local file path

RUN_NAME=$(basename "$0" .sh)

# Use two datasets for training, reserve Additional_video for validation
TRAIN_DATA="data/BraTS_GLI_TrainingData_video,data/MSD_T1_MRI_video"
VAL_DATA="data/BraTS_GLI_TrainingData_Additional_video"

python3 -m verl.trainer.main \
    config=training_scripts/brain_tumor_3d_4x80G.yaml \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${VAL_DATA} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.kl_loss_coef=1.0e-2 \
    worker.actor.optim.lr=1.0e-6 \
    worker.actor.micro_batch_size_per_device_for_update=8 \
    worker.actor.micro_batch_size_per_device_for_experience=8 \
    worker.rollout.enable_chunked_prefill=false \
    worker.rollout.n=4 \
    worker.rollout.tensor_parallel_size=2 \
    worker.rollout.gpu_memory_utilization=0.7 \
    worker.reward.compute_score=brain_tumor_3d \
    trainer.experiment_name=${RUN_NAME} \
    trainer.n_gpus_per_node=4 \
    trainer.total_episodes=15 \
    trainer.save_checkpoint_path=brain_tumor_workdir/${RUN_NAME}