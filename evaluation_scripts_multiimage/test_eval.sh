#!/bin/bash
# Quick test of evaluation script with base model on small subset

echo "=========================================================================="
echo "Quick Test: Evaluating base model on 10 samples"
echo "=========================================================================="

export CUDA_VISIBLE_DEVICES=0
export MODEL_PATH=models/Qwen3-VL-8B-Thinking

python evaluation_scripts/eval_brain_tumor.py \
    --model_path $MODEL_PATH \
    --data_path data/BraTS_GLI_Additional_Image_280/T1C \
    --output_path eval_results/quick_test \
    --batch_size 4 \
    --max_samples 10 \
    --image_size 280

echo ""
echo "Test complete! Check results at: eval_results/quick_test/evaluation_results.json"
