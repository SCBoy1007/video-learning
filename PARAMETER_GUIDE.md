# Parameter Configuration Guide

This document explains the parameter hierarchy in the Video-Learning project to avoid confusion from multiple configuration sources.

## Parameter Priority (High → Low)

```
Shell Script (.sh) > YAML Config (.yaml) > Code Defaults
```

**Rule**: If a parameter appears in multiple places, the **rightmost source** in the chain above takes precedence.

---

## Parameter Organization Strategy

**Updated Strategy** (as of latest commit):
- **YAML now contains the actual values** used in training
- **Shell script only overrides**:
  1. Dynamic parameters (data paths, experiment name) - must be in shell
  2. Tuning parameters (kl_coef, lr) - kept in shell for quick experiments

| Parameter Type | Location | Reason |
|----------------|----------|--------|
| `data.train_files` | Shell only | Dynamic, depends on experiment |
| `trainer.experiment_name` | Shell only | Dynamic, depends on script name |
| `worker.actor.kl_loss_coef` | YAML: 0.08<br>Shell: can override | Experiment tuning parameter |
| `worker.actor.optim.lr` | YAML: 1e-5<br>Shell: can override | Experiment tuning parameter |
| `worker.rollout.n` | YAML only (8) | Stable, no need to override |
| `worker.rollout.gpu_memory_utilization` | YAML only (0.55) | Stable, no need to override |

**Key Improvement**: YAML and Shell are now **mostly consistent**. Shell overrides are minimal and intentional.

---

## Parameter Sources Breakdown

### 1. YAML File (`brain_tumor_3d_4x80G.yaml`)

**Purpose**: Default configuration baseline

**Key sections**:
```yaml
algorithm:
  kl_coef: 0.0  # Reward-level KL (disabled, use kl_loss_coef instead)

worker.actor:
  kl_loss_coef: 0.01   # ⚠️ OVERRIDDEN → 0.08
  lr: 3e-5              # ⚠️ OVERRIDDEN → 1e-5

worker.rollout:
  n: 8                  # GRPO samples per prompt
  temperature: 1.0      # Sampling temperature
```

### 2. Shell Script (`run_brain_tumor_3d_4x80G.sh`)

**Purpose**: Runtime overrides for experiment-specific settings

**Key overrides**:
```bash
worker.actor.kl_loss_coef=8.0e-2    # 0.01 → 0.08
worker.actor.optim.lr=1.0e-5         # 3e-5 → 1e-5
```

**Why use shell overrides?**
- Quick parameter tuning without editing YAML
- Easy to compare different experiments (just duplicate .sh file)
- Command-line arguments are self-documenting in logs

### 3. Code Defaults (`verl/utils/reward_score/brain_tumor_3d.py`)

**Purpose**: Reward function weights

```python
REWARD_WEIGHTS = {
    'thinking_format': 0.5,
    'video_keyword': 0.5,
    'format': 0.5,
    'iou': 1.0,           # Core task (reduced to stabilize training)
    'peak_slice': 1.0,    # Reduced to stabilize training
    'tumor_ratio': 1.0,   # Reduced to stabilize training
}
```

**Note**: These are NOT overridable by YAML/Shell. Must modify code directly.

---

## Quick Reference: Where to Modify Parameters

| Want to change... | Edit this file | Notes |
|-------------------|----------------|-------|
| **KL coefficient** | `.sh` script | Also update YAML comment |
| **Learning rate** | `.sh` script | Also update YAML comment |
| **Reward weights** | `brain_tumor_3d.py` | Requires code change + commit |
| **Batch size** | `.sh` script (or YAML if permanent) | |
| **Rollout samples (n)** | `.sh` script | |
| **Training episodes** | `.sh` script | |
| **Dataset paths** | `.sh` script | |
| **Prompt template** | `rl_dataset.py` | Requires code change + commit |

---

## Common Mistakes to Avoid

### ❌ Mistake 1: Only checking YAML
```bash
# You see in YAML:
lr: 3e-5

# But actual training uses:
lr: 1e-5  (from shell script)
```

**Solution**: Always check both YAML and Shell script!

### ❌ Mistake 2: Modifying YAML when shell overrides exist
```bash
# You change YAML:
kl_loss_coef: 0.05

# But shell still overrides:
worker.actor.kl_loss_coef=8.0e-2  # This wins!
```

**Solution**: Either modify shell script, or remove the override line from shell.

### ❌ Mistake 3: Forgetting to commit code changes
```python
# You change reward weights in brain_tumor_3d.py locally
REWARD_WEIGHTS = {'iou': 10.0}

# But forgot to git commit/push
# Training on server still uses old weights!
```

**Solution**: Always commit + push code changes, then `git pull` on server.

---

## Best Practice Workflow

### When tuning hyperparameters:

1. **Quick experiments** (KL, lr, batch size):
   - Modify `.sh` script
   - Push to git
   - Run on server

2. **Permanent changes** (architecture, default configs):
   - Modify YAML
   - Remove override from `.sh` (or update to match)
   - Push to git

3. **Reward function changes**:
   - Modify `brain_tumor_3d.py`
   - Commit with clear message
   - Push to git
   - Pull on server

### Parameter change checklist:

- [ ] Updated `.sh` script (if runtime override needed)
- [ ] Updated YAML comments (if shell overrides YAML)
- [ ] Committed code changes (if modified .py files)
- [ ] Pushed to git
- [ ] Pulled on server before training

---

## Current Training Configuration (as of latest commit)

```yaml
# Actual values used (after all overrides)
algorithm.kl_coef: 0.0
worker.actor.kl_loss_coef: 5.0e-3
worker.actor.optim.lr: 1.0e-6
worker.actor.max_grad_norm: 200.0
worker.actor.micro_batch_size: 2
worker.rollout.n: 8
worker.rollout.enable_chunked_prefill: false

# Reward weights (from brain_tumor_3d.py)
thinking_format: 0.5
video_keyword: 0.5
format: 0.5
iou: 1.0
peak_slice: 1.0
tumor_ratio: 1.0
# Total: 4.5 points
```

---

## Debugging Tips

### How to verify what parameters are actually used?

1. **Check WandB logs**: All hyperparameters are logged at start
2. **Look at console output**: First few lines show full config
3. **Read the shell script**: Final source of truth for overrides

### Parameter not taking effect?

1. Did you check the shell script? (most common issue)
2. Did you git pull on server after push?
3. Did you restart training after config change?
4. Is there a typo in parameter name? (e.g., `kl_coef` vs `kl_loss_coef`)

---

## Related Files

- `training_scripts/brain_tumor_3d_4x80G.yaml` - Base configuration
- `training_scripts/run_brain_tumor_3d_4x80G.sh` - Runtime overrides
- `verl/utils/reward_score/brain_tumor_3d.py` - Reward weights
- `verl/utils/rl_dataset.py` - Prompt template
- `verl/trainer/config.py` - Config schema definitions

---

**Last Updated**: 2025-10-02 (after reward weight rebalancing to stabilize training)
