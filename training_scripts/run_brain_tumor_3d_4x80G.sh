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

# 训练数据集：高质量模态 + 脑膜瘤数据（Hydra列表格式）
TRAIN_DATA="[data/BraTS2024-BraTS-GLI-T1C/train,data/BraTS2024-BraTS-GLI-T2F/train,data/MSD-Task01-BrainTumour-T1Gd/train,data/MSD-Task01-BrainTumour-FLAIR/train,data/BraTS2024_MEN_RT_TrainingData_video/train]"

# 验证集同样只用 BraTS 的 T1C/T2F（Hydra列表格式）
VAL_DATA="[data/BraTS2024-BraTS-GLI-Additional-T1C/train,data/BraTS2024-BraTS-GLI-Additional-T2F/train]"

# ============================================================================
# Parameter Override Section
# Only override parameters that:
#   1. Must be dynamic (data paths, experiment name)
#   2. Are actively being tuned (kl_coef, lr for experiments)
# Other parameters should match YAML to avoid confusion.
# ============================================================================
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=false \
    algorithm.use_kl_in_reward=false \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${VAL_DATA} \
    data.prompt_key=problem \
    data.max_prompt_length=28000 \
    data.max_response_length=800 \
    data.train_batch_size=16 \
    data.shuffle=true \
    data.filter_overlong_prompts=false \
    data.truncation=error \
    data.reward_fn_key=data_source \
    data.video_key=videos \
    +data.max_pixels=12845056 \
    +data.min_pixels=3136 \
    custom_reward_function.path=verl/utils/reward_score/brain_tumor_3d.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.use_fused_kernels=false \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.kl_loss_coef=8.0e-2 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.optim.lr=1.0e-5 \
    actor_rollout_ref.actor.optim.weight_decay=1.0e-2 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=false \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.free_cache_engine=true \
    actor_rollout_ref.rollout.enforce_eager=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    trainer.total_epochs=15 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=brain_tumor_3d_4x80G \
    trainer.experiment_name=${RUN_NAME} \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=230 \
    trainer.val_before_train=true \
    trainer.val_only=false \
    trainer.default_local_dir=brain_tumor_workdir/${RUN_NAME}
