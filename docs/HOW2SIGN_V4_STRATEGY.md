# How2Sign V4 Training Strategy

## Smaller Model to Combat Overfitting

### 📊 Problem Analysis

**Previous Training Attempts:**

| Version | Val Loss | BLEU-4 | Parameters | Issue                            |
| ------- | -------- | ------ | ---------- | -------------------------------- |
| V1      | 4.12     | 0.008  | 10.8M      | Overfitting, aggressive LR decay |
| V2      | 4.77     | N/A    | 7.5M       | LR collapsed to 1e-8 by epoch 7  |
| V3      | 4.77     | N/A    | 7.5M       | Train/val gap 0.81, overfitting  |

**Root Cause Identified:**

- ✅ Dataset converges (train loss decreases)
- ✅ Scheduler works (CosineAnnealingLR smooth decay)
- ❌ **Model too large for dataset size** (29k samples)
- ❌ Val loss stops improving after ~8 epochs
- ❌ Train loss keeps decreasing → massive overfitting

### 🎯 V4 Strategy: Reduce Model Capacity

**Key Changes:**

```bash
# Architecture
d_model: 512 → 256 (50% reduction)
encoder_layers: 3 → 2
decoder_layers: 3 → 2
dim_feedforward: 2048 → 1024 (50% reduction)
Parameters: 7.5M → ~2M (73% reduction!)

# Training
batch_size: 32 → 64 (more updates per epoch)
dropout: 0.2 → 0.3 (stronger regularization)
epochs: 70 → 30 (with early stopping)
early_stopping: 10 epochs patience

# Scheduler
CosineAnnealingLR: 1e-4 → 1e-6 (smooth decay)
```

### 💡 Why This Should Work

**Theoretical Justification:**

1. **Smaller capacity** = less ability to memorize training data
2. **Higher dropout** = forces model to learn robust features
3. **Larger batches** = more stable gradients, less variance
4. **Early stopping** = prevents training after optimal point
5. **Smooth LR decay** = consistent optimization throughout

**Expected Behavior:**

- Train/val gap < 0.3 (was 0.81 in V3)
- Val loss continues improving beyond epoch 8
- BLEU-4 > 0.02-0.05 (vs 0.008 in V1)
- Model generalizes instead of memorizing

### ⏱️ Time Estimates

**V4 Quick Test:** 10 minutes (3 epochs)

- Verify smaller model works
- Check memory usage with batch_size=64
- Confirm val loss improving

**V4 Full Training:** 4-5 hours (up to 30 epochs)

- Likely stops early (~15-20 epochs)
- Much faster than V2/V3 (13 hours)

### 🚀 Execution Plan

**Step 1: Quick Test** (optional but recommended)

```bash
./run_how2sign_training_v4_quick.sh
```

- Duration: ~10 minutes
- Check: Val loss improving, no crashes

**Step 2: Full Training**

```bash
./run_how2sign_training_v4.sh
```

- Duration: 4-5 hours
- Auto-stops if no improvement for 10 epochs

**Step 3: Evaluation**

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
python src/sign_to_text/evaluate_how2sign.py \
    --checkpoint models/sign_to_text/how2sign_v4/best_checkpoint.pt \
    --split val
```

### 📈 Success Criteria

**Minimum Success:**

- Val loss < 4.5 (better than V2/V3)
- Train/val gap < 0.5
- BLEU-4 > 0.015 (2x better than V1)

**Good Success:**

- Val loss < 4.0
- Train/val gap < 0.3
- BLEU-4 > 0.03-0.05
- Vocabulary coverage > 2%

**Excellent Success:**

- Val loss < 3.8
- Train/val gap < 0.2
- BLEU-4 > 0.08-0.10
- Coherent predictions (no repetitions)

### 🔄 If V4 Still Overfits

**Plan B Options:**

1. **Even smaller model:**

   - d_model=128, layers 2+2
   - ~500k parameters

2. **Data augmentation:**

   - Temporal jittering
   - Spatial noise on landmarks
   - Speed perturbation

3. **Reduced dataset:**

   - Train on 10k samples first
   - Fine-tune on full 29k

4. **Pre-trained embeddings:**
   - Use pre-trained sign language features
   - Freeze encoder, train only decoder

### 📝 For Thesis

**Narrative for Methodology:**

> "Through systematic analysis of training dynamics, we identified model capacity mismatch as the primary bottleneck. Training curves revealed consistent convergence on training data but validation loss plateauing after 8 epochs, indicating overfitting rather than optimization issues. We addressed this by reducing model parameters from 7.5M to 2M (73% reduction) while increasing regularization through higher dropout (0.3) and larger batch sizes (64). This architectural optimization, guided by empirical evidence rather than blind hyperparameter search, demonstrates the importance of matching model capacity to dataset scale in neural machine translation tasks."

**Key Points:**

- ✅ Evidence-based optimization (not trial-and-error)
- ✅ Clear diagnosis (overfitting, not underfitting)
- ✅ Principled solution (capacity reduction)
- ✅ Iterative improvement (V1→V2→V3→V4)

### 🎓 Learning Outcomes

**What worked:**

- ✅ Systematic debugging (analyzing training curves)
- ✅ Understanding failure modes (overfitting vs optimization)
- ✅ CosineAnnealingLR (smooth, predictable)
- ✅ Early stopping (prevents wasted computation)

**What didn't work:**

- ❌ Large models on small datasets
- ❌ ReduceLROnPlateau (too aggressive)
- ❌ Training to 70 epochs (unnecessary)

**Thesis Value:**

- Demonstrates engineering methodology
- Shows understanding of deep learning principles
- Provides clear improvement trajectory
- Offers insights beyond just final metrics
