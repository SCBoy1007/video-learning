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

# Memory optimization - use larger split size to balance fragmentation vs OOM
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

MODEL_PATH=pretrained_models/Qwen2.5-VL-7B-Instruct  # Use local model path to avoid Docker cache issues

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

# ============================================================================
# Parameter Override Section
# Only override parameters that:
#   1. Must be dynamic (data paths, experiment name)
#   2. Are actively being tuned (kl_coef, lr for experiments)
# Other parameters should match YAML to avoid confusion.
# ============================================================================
python3 -m verl.trainer.main \
    config=training_scripts/brain_tumor_3d_4x80G.yaml \
    `# Dynamic paths (must be in shell script)` \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${VAL_DATA} \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=${RUN_NAME} \
    trainer.save_checkpoint_path=brain_tumor_workdir/${RUN_NAME} \
    `# Experiment tuning params - THESE OVERRIDE brain_tumor_3d_4x80G.yaml` \
    worker.actor.kl_loss_coef=5.0e-2 \
    worker.actor.optim.lr=1.0e-5 \
    worker.actor.max_grad_norm=100.0 \
    worker.rollout.temperature=1.8
