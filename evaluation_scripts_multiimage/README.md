# Multi-Image Evaluation Framework

This directory contains evaluation scripts for **8-slice MRI** brain tumor localization (multi-image mode), adapted from the original single-image evaluation framework.

## 📁 Directory Structure

```
evaluation_scripts/           # Original single-image evaluation (preserved)
evaluation_scripts_multiimage/ # New multi-image evaluation (8 slices)
```

## 🔍 Key Differences

| Feature | Single-Image (original) | Multi-Image (this folder) |
|---------|------------------------|---------------------------|
| **Images per sample** | 1 | 8 (configurable) |
| **Prompt** | Single <image> tag | 8x <image> tags with multi-slice instructions |
| **Use case** | Baseline comparison | Production/paper results |
| **Batch size** | 8 | 4 (due to memory) |

## 🚀 Quick Start

### Evaluate a checkpoint (e.g., step 400):

```bash
bash evaluation_scripts_multiimage/run_eval_multiimage.sh models/Qwen3-VL-8B-Thinking 400
```

### Evaluate base model:

```bash
bash evaluation_scripts_multiimage/run_eval_multiimage.sh models/Qwen3-VL-8B-Thinking
```

### Custom evaluation:

```bash
python evaluation_scripts_multiimage/eval_brain_tumor.py \
    --model_path models/Qwen3-VL-8B-Thinking \
    --data_path data/BraTS_GLI_Additional_Image_280/T1C \
    --output_path eval_results_multiimage/my_test \
    --num_images 8 \
    --batch_size 4 \
    --image_size 280
```

## 📊 Parameters

### `--num_images` (key parameter)
- **1**: Single-image mode (same as original evaluation_scripts)
- **8**: Multi-image mode (matches training with 8 slices)
- **4**: For testing with 4 slices
- **Default**: 8

### Other parameters:
- `--model_path`: Path to model or checkpoint
- `--data_path`: Dataset path (e.g., `data/BraTS_GLI_Additional_Image_280/T1C`)
- `--output_path`: Where to save results
- `--batch_size`: Batch size (default=4, reduce if OOM)
- `--image_size`: Image dimension (default=280)
- `--max_samples`: For quick testing (e.g., `--max_samples 10`)

## 📈 Example Output

```
==================================================
Brain Tumor Detection Evaluation (Multi-Image)
==================================================
Model: brain_tumor_workdir/.../global_step_400/actor
Dataset: data/BraTS_GLI_Additional_Image_280/T1C
Images per sample: 8 (multi-image mode)
Batch size: 4
==================================================

[1/4] Loading model...
[2/4] Loading dataset... (1234 samples)
[3/4] Preparing messages...
[4/4] Running inference...

Results:
- Average IoU: 0.523
- IoU@0.5: 67.8%
- IoU@0.7: 42.1%
```

## 🔬 For Paper Experiments

### Compare single-image vs multi-image:

```bash
# Single-image baseline
cd evaluation_scripts
bash run_eval_brain_tumor.sh models/Qwen3-VL-8B-Thinking

# Multi-image (8 slices)
cd evaluation_scripts_multiimage
bash run_eval_multiimage.sh models/Qwen3-VL-8B-Thinking 400
```

### Ablation study (2/4/8 slices):

```bash
for N in 2 4 8; do
    python evaluation_scripts_multiimage/eval_brain_tumor.py \
        --model_path models/Qwen3-VL-8B-Thinking \
        --data_path data/BraTS_GLI_Additional_Image_280/T1C \
        --output_path eval_results_multiimage/ablation_${N}slices \
        --num_images $N \
        --batch_size 4
done
```

## 📝 Notes

1. **Original scripts preserved**: The `evaluation_scripts/` folder remains unchanged for backward compatibility
2. **Same dataset format**: Both use the same data format from `generate_unified_image_dataset.py`
3. **Prompt alignment**: Multi-image prompts match the training setup in `verl/utils/rl_dataset.py`
4. **Memory usage**: Multi-image requires more memory, reduce batch_size if needed

## 🐛 Troubleshooting

### CUDA OOM:
```bash
# Reduce batch size
--batch_size 2

# Or reduce number of images
--num_images 4
```

### Dataset not found:
```bash
# Ensure dataset was generated with multi-image support
python generate_unified_image_dataset.py \
    --num_images 8 \
    --output_path data/BraTS_GLI_Additional_Image_280
```

### Wrong IoU scores:
- Check `--num_images` matches your training setup (should be 8)
- Verify dataset has "images" field (not just "image")
