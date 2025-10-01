# Video-Learning: 3D Medical Image Analysis with Vision-Language Models

This repository is a **research fork** of [Seg-Zero](https://github.com/dvlab-research/Seg-Zero) adapted for **3D brain tumor detection** using video-based vision-language models and reinforcement learning.

## Project Overview

**Video-Learning** applies the GRPO (Group Relative Policy Optimization) algorithm to train Qwen2.5-VL models for 3D medical imaging tasks. Instead of segmentation masks, the model predicts structured outputs including 3D bounding boxes, peak slices, and tumor volume ratios.

### Key Adaptations from Seg-Zero:

1. **Task**: Image segmentation → **3D brain tumor detection**
2. **Input**: 2D images → **Multi-frame medical videos** (4-modality MRI concatenation)
3. **Output**: Segmentation masks → **Structured JSON** with:
   - `bbox_3d`: 3D bounding box coordinates `[x1, y1, z1, x2, y2, z2]`
   - `peak_slice`: Peak tumor slice index
   - `tumor_ratio`: Tumor volume ratio
4. **Datasets**: RefCOCOg/ReasonSeg → **BraTS 2024 & MSD Brain Tumor MRI**
5. **Reward Function**: IoU-based segmentation rewards → **3D IoU + multi-metric rewards**

## Architecture

- **Base Model**: Qwen2.5-VL-7B-Instruct
- **Training Framework**: veRL (Volcano Engine Reinforcement Learning)
- **Algorithm**: GRPO with custom 3D medical imaging reward functions
- **Hardware**: Optimized for 4×A100-80GB GPUs (single-GPU inference, TP=1)

## Installation

```bash
git clone https://github.com/SCBoy1007/video-learning.git
cd video-learning
conda create -n vl python=3.10
conda activate vl
pip install torch==2.6.0 torchvision==0.21.0
pip install -e .
```

## Training

### Data Preparation

The project uses video representations of 3D MRI scans:
- **BraTS 2024 GLI Training Data**: 1,251 cases
- **MSD Brain Tumor Dataset (Task01)**: 484 cases
- **Video format**: 4-modality concatenation (T1C, T1N, T2F, T2W) as frames

Videos and annotations should be placed in:
```
data/
├── BraTS_GLI_TrainingData_video/
├── MSD_T1_MRI_video/
└── BraTS_GLI_TrainingData_Additional_video/  # Validation set
```

Each dataset includes:
- `.mp4` video files (normalized medical imaging frames)
- Arrow format annotations with ground truth 3D bounding boxes

### Model Training (4×A100-80GB)

```bash
# The model will automatically download from HuggingFace on first run
bash training_scripts/run_brain_tumor_3d_4x80G.sh
```

**Key Configuration** (optimized for 4-GPU setup):
- `tensor_parallel_size: 1` (single-GPU inference, no TP overhead)
- `rollout.n: 8` (8 GRPO samples per prompt)
- `gpu_memory_utilization: 0.6` (conservative memory usage)
- `micro_batch_size: 4` per device (global batch size: 16)

### Training Monitoring

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# View logs (if running in background)
tail -f train_4x80G.log
```

Logs are automatically sent to **Weights & Biases** (project: `brain_tumor_3d_4x80G`).

## Reward Function

The custom reward function evaluates predictions across multiple dimensions (max score: 8.5):

1. **Thinking Format Reward** (0.5): Strict `<think>...</think><answer>...</answer>` format (aligned with Seg-Zero)
2. **Video Keyword Reward** (0.5): Thinking must start with "This video shows" to reinforce video awareness
3. **Format Reward** (1.0): Valid JSON structure with required fields
4. **3D IoU Reward** (3.0): Bounding box overlap (doubled weight for core task)
   - IoU > 0.7: 3.0 points
   - IoU > 0.5: 2.0 points
   - IoU > 0.3: 1.0 points
5. **Peak Slice Reward** (1.5): Accuracy of peak tumor slice (increased weight)
   - Error ≤3: 1.5 points
   - Error ≤5: 1.05 points
   - Error ≤10: 0.45 points
6. **Tumor Ratio Reward** (1.5): Volume estimation accuracy (increased weight)
   - Error ≤10%: 1.5 points
   - Error ≤20%: 1.05 points
   - Error ≤30%: 0.45 points
7. **Non-repetition Reward** (0.5): Output quality check

**Note**: Completeness reward removed (was 100% redundant with Format reward).

See implementation: [`verl/utils/reward_score/brain_tumor_3d.py`](verl/utils/reward_score/brain_tumor_3d.py)

## Coordinate System

**Critical**: The training pipeline uses **video pixel coordinates**, not original NIfTI voxel coordinates.

- Videos are resized during generation (e.g., BraTS: 182×218×155 → 224×196×155)
- All bounding box annotations are **pre-converted** to video pixel space
- Conversion script: [`data/fix_dataset_coordinates.py`](data/fix_dataset_coordinates.py)

## Project Structure

```
video-learning/
├── verl/                          # Modified veRL framework
│   ├── utils/
│   │   ├── reward_score/
│   │   │   └── brain_tumor_3d.py # Custom 3D tumor reward function
│   │   └── rl_dataset.py          # Multi-dataset loading (fixed)
│   └── workers/                   # FSDP + vLLM workers
├── training_scripts/
│   ├── brain_tumor_3d_4x80G.yaml  # 4-GPU training config
│   └── run_brain_tumor_3d_4x80G.sh # Training launch script
├── data/                          # Dataset directory (not in git)
│   ├── fix_dataset_coordinates.py # Coordinate conversion tool
│   └── [video datasets...]
└── requirements.txt               # Python dependencies (SAM2 removed)
```

## Key Differences from Original Seg-Zero

| Aspect | Seg-Zero | Video-Learning |
|--------|----------|----------------|
| Task | 2D referring segmentation | 3D tumor detection |
| Input | Single images | Multi-frame videos (4 modalities) |
| Output | Segmentation masks (via SAM2) | Structured JSON predictions |
| Datasets | RefCOCOg, ReasonSeg | BraTS 2024, MSD Brain Tumor |
| Reward | Mask IoU | 3D bbox IoU + multi-metrics |
| Dependencies | Requires SAM2 | Pure vision-language (SAM2 removed) |
| Coordinate Space | Image pixels | Video frame coordinates |

## Hardware Requirements

**Recommended**:
- 4×A100-80GB GPUs
- 256GB+ RAM (for data loading)
- 500GB+ storage (for video datasets)

**Memory Profile**:
- Rollout (inference): ~28GB per GPU
- Actor (training): ~15GB per GPU (with parameter offloading)

## Git Workflow

This is a **research repository** with frequent updates. Typical workflow:

```bash
# Local development (Mac/Windows)
git add <files>
git commit -m "description"
git push origin master

# Server deployment (Linux)
cd ~/Downloads/video-learning
git pull
dos2unix training_scripts/*.sh  # Fix line endings if needed
bash training_scripts/run_brain_tumor_3d_4x80G.sh
```

## Recent Major Changes

- **2025-09-30**: Optimized 4-GPU config (TP=2 → TP=1, removed SAM2 dependency)
- **2025-09-29**: Fixed coordinate system bug (axis mapping correction)
- **2025-09-22**: Implemented 3D brain tumor detection reward function
- **2025-09-21**: Initial fork from Seg-Zero, multi-dataset support

See [commit history](https://github.com/SCBoy1007/video-learning/commits/master) for details.

## Troubleshooting

### Line Ending Issues (Windows/Mac → Linux)
```bash
dos2unix training_scripts/*.sh
```

### Model Download Issues
```bash
# If HuggingFace is slow, use mirror
export HF_ENDPOINT=https://hf-mirror.com
```

### Video Playback Issues on Mac
Mac QuickTime may show green screens for FFmpeg-encoded videos. This **does not affect training** (which uses OpenCV/FFmpeg decoding). Use VLC player to verify videos on Mac.

## Acknowledgements

This project is built upon:
- **[Seg-Zero](https://github.com/dvlab-research/Seg-Zero)** - Original GRPO-based vision reasoning framework
- **[veRL](https://github.com/volcengine/verl)** - Efficient reinforcement learning framework
- **[Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)** - Vision-language foundation model
- **[BraTS 2024](https://www.synapse.org/brats2024)** - Brain tumor segmentation challenge data
- **[Medical Segmentation Decathlon](http://medicaldecathlon.com/)** - MSD brain tumor dataset

## Citation

If you use this codebase, please cite the original Seg-Zero paper:

```bibtex
@article{liu2025segzero,
  title        = {Seg-Zero: Reasoning-Chain Guided Segmentation via Cognitive Reinforcement},
  author       = {Liu, Yuqi and Peng, Bohao and Zhong, Zhisheng and Yue, Zihao and Lu, Fanbin and Yu, Bei and Jia, Jiaya},
  journal      = {arXiv preprint arXiv:2503.06520},
  year         = {2025}
}
```

## License

This project inherits the license from the original Seg-Zero repository. Please refer to the upstream repository for license details.

---

**Note**: This is a research fork for educational purposes. The original Seg-Zero inference/evaluation scripts for referring segmentation are preserved but not used in the 3D medical imaging pipeline.