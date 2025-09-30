#!/bin/bash

# 4-card A100 80G configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3

set -x

# Reduce log verbosity
export VLLM_ATTENTION_BACKEND=XFORMERS
export RAY_DEDUP_LOGS=1                    # Deduplicate Ray logs across workers
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1 # Suppress transformers warnings
export VLLM_LOGGING_LEVEL=WARNING          # Only show vLLM warnings/errors
export TOKENIZERS_PARALLELISM=false        # Suppress tokenizer parallelism warnings

# Memory optimization
# expandable_segments incompatible with vLLM, using max_split_size_mb instead
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

MODEL_PATH=pretrained_models/Qwen2.5-VL-7B-Instruct  # Use local model path to avoid Docker cache issues

RUN_NAME=$(basename "$0" .sh)

# Use two datasets for training, reserve Additional_video for validation
TRAIN_DATA="data/BraTS_GLI_TrainingData_video/train,data/MSD_T1_MRI_video/train"
VAL_DATA="data/BraTS_GLI_TrainingData_Additional_video/train"

python3 -m verl.trainer.main \
    config=training_scripts/brain_tumor_3d_4x80G.yaml \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${VAL_DATA} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.kl_loss_coef=1.0e-3 \
    worker.actor.optim.lr=3.0e-5 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.rollout.enable_chunked_prefill=false \
    worker.rollout.n=8 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.gpu_memory_utilization=0.55 \
    worker.reward.compute_score=brain_tumor_3d \
    trainer.experiment_name=${RUN_NAME} \
    trainer.n_gpus_per_node=4 \
    trainer.total_episodes=15 \
    trainer.save_checkpoint_path=brain_tumor_workdir/${RUN_NAME}