#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

MODEL_PATH=pretrained_models/Qwen2.5-VL-7B-Instruct  # replace it with your local file path

RUN_NAME=$(basename "$0" .sh)

# 训练数据集：高质量模态 + 脑膜瘤数据
TRAIN_DATA="data/BraTS2024-BraTS-GLI-T1C/train,\
data/BraTS2024-BraTS-GLI-T2F/train,\
data/MSD-Task01-BrainTumour-T1Gd/train,\
data/MSD-Task01-BrainTumour-FLAIR/train,\
data/BraTS2024_MEN_RT_TrainingData_video/train"

# 验证集同样只用 BraTS 的 T1C/T2F
VAL_DATA="data/BraTS2024-BraTS-GLI-Additional-T1C/train,\
data/BraTS2024-BraTS-GLI-Additional-T2F/train"

python3 -m verl.trainer.main \
    config=training_scripts/brain_tumor_3d_7b.yaml \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${VAL_DATA} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.kl_loss_coef=1.0e-2 \
    worker.actor.optim.lr=1.0e-6 \
    worker.actor.micro_batch_size_per_device_for_update=16 \
    worker.actor.micro_batch_size_per_device_for_experience=16 \
    worker.rollout.enable_chunked_prefill=false \
    worker.rollout.n=8 \
    worker.reward.compute_score=brain_tumor_3d \
    trainer.experiment_name=${RUN_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.total_episodes=15 \
    trainer.save_checkpoint_path=brain_tumor_workdir/${RUN_NAME}