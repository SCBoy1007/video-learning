#!/bin/bash
# Storage check script for Video-Learning project
# Run this on the training server to check available space

set +x  # Disable command echo for cleaner output

echo "=========================================="
echo "Video-Learning Storage Check"
echo "Time: $(date)"
echo "=========================================="

echo ""
echo "1️⃣  Overall Disk Space"
echo "=========================================="
df -h | grep -E '(Filesystem|/dev/|overlay)'

echo ""
echo "2️⃣  Docker Storage Usage"
echo "=========================================="
if command -v docker &> /dev/null; then
    docker system df
else
    echo "Docker not found or not accessible"
fi

echo ""
echo "3️⃣  Project Directory Size"
echo "=========================================="
PROJECT_DIR="$HOME/Downloads/video-learning"
if [ -d "$PROJECT_DIR" ]; then
    echo "Total project size:"
    du -sh "$PROJECT_DIR"
    echo ""
    echo "Breakdown:"
    du -sh "$PROJECT_DIR"/* 2>/dev/null | sort -hr | head -10
else
    echo "Project directory not found at $PROJECT_DIR"
fi

echo ""
echo "4️⃣  Training Artifacts"
echo "=========================================="
WORKDIR="$PROJECT_DIR/brain_tumor_workdir"
if [ -d "$WORKDIR" ]; then
    echo "Workdir size: $(du -sh $WORKDIR | cut -f1)"
    echo "Checkpoint count:"
    find "$WORKDIR" -name "*.pt" -o -name "*.bin" | wc -l
    echo ""
    echo "Recent checkpoints:"
    find "$WORKDIR" -name "global_step*" -type d | sort -V | tail -5
else
    echo "No training workdir found yet"
fi

echo ""
echo "5️⃣  Pretrained Models"
echo "=========================================="
MODEL_DIR="$PROJECT_DIR/pretrained_models"
if [ -d "$MODEL_DIR" ]; then
    du -sh "$MODEL_DIR"
    ls -lh "$MODEL_DIR" 2>/dev/null
else
    echo "No pretrained models directory"
fi

echo ""
echo "6️⃣  Dataset Size"
echo "=========================================="
DATA_DIR="$PROJECT_DIR/data"
if [ -d "$DATA_DIR" ]; then
    echo "Total data size: $(du -sh $DATA_DIR | cut -f1)"
    echo ""
    echo "Breakdown by dataset:"
    du -sh "$DATA_DIR"/* 2>/dev/null | sort -hr
else
    echo "No data directory found"
fi

echo ""
echo "7️⃣  Space Recommendations"
echo "=========================================="

# Calculate available space
AVAIL_GB=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')

echo "Available space: ${AVAIL_GB}GB"
echo ""

if [ "$AVAIL_GB" -lt 100 ]; then
    echo "⚠️  WARNING: Less than 100GB available!"
    echo "Recommendations:"
    echo "  - Consider save_freq=50 (instead of 10)"
    echo "  - Enable remove_previous_ckpt=true"
    echo "  - Clean old experiments"
elif [ "$AVAIL_GB" -lt 500 ]; then
    echo "⚠️  MODERATE: 100-500GB available"
    echo "Recommendations:"
    echo "  - save_freq=20 is reasonable"
    echo "  - Consider remove_previous_ckpt=true after 50 steps"
else
    echo "✅ PLENTY: 500GB+ available"
    echo "Recommendations:"
    echo "  - save_freq=10 is fine"
    echo "  - Can keep remove_previous_ckpt=false for debugging"
fi

echo ""
echo "8️⃣  Cleanup Commands (if needed)"
echo "=========================================="
echo "# Remove old checkpoints (CAUTION!):"
echo "rm -rf $WORKDIR/run_brain_tumor_3d_4x80G/global_step_*"
echo ""
echo "# Clean Docker cache:"
echo "docker system prune -a --volumes"
echo ""
echo "# Remove old WandB logs:"
echo "rm -rf $PROJECT_DIR/wandb/run-*"

echo ""
echo "=========================================="
echo "Storage check complete!"
echo "=========================================="
