#!/bin/bash
#
# How2Sign Training V4 - Smaller Model to Reduce Overfitting
# ===========================================================
#
# Strategy: Reduce model capacity to match dataset size
# - d_model: 512 → 256 (50% reduction)
# - layers: 3+3 → 2+2 (33% reduction)
# - params: 7.5M → ~2M (73% reduction!)
# - batch: 32 → 64 (more updates, less variance)
# - dropout: 0.2 → 0.3 (stronger regularization)
# - epochs: 70 → 30 (with early stopping)
#
# Previous attempts analysis:
# V1: val_loss 4.12, BLEU-4 0.008, overfitting (10.8M params)
# V2: val_loss 4.77, aggressive LR decay (7.5M params)
# V3: val_loss 4.77, train/val gap 0.81, overfitting (7.5M params)
# V4: Smaller model → less overfitting expected!
#

set -e

echo "========================================================================"
echo "🚀 HOW2SIGN TRAINING V4 - SMALLER MODEL"
echo "========================================================================"

# Configuration - SMALLER MODEL
EPOCHS=30
BATCH_SIZE=64          # was 32 → more updates per epoch
LR=0.0001
D_MODEL=256            # was 512 → 50% smaller!
NUM_ENCODER_LAYERS=2   # was 3 → lighter encoder
NUM_DECODER_LAYERS=2   # was 3 → lighter decoder
NHEAD=8                # keep 8 (d_model/nhead = 32)
DIM_FEEDFORWARD=1024   # was 2048 → 50% smaller
DROPOUT=0.3            # was 0.2 → stronger regularization
WEIGHT_DECAY=0.0001
LABEL_SMOOTHING=0.1
EARLY_STOPPING_PATIENCE=10  # Stop if no improvement for 10 epochs

echo ""
echo "📋 Configuration (V4 - Smaller Model):"
echo "   Epochs: $EPOCHS (reduced from 70)"
echo "   Batch Size: $BATCH_SIZE (increased from 32)"
echo "   Learning Rate: $LR → 1e-6 (cosine decay)"
echo "   Early Stopping: $EARLY_STOPPING_PATIENCE epochs patience"
echo ""
echo "🔧 Model Architecture:"
echo "   d_model: $D_MODEL (was 512, -50%)"
echo "   Encoder layers: $NUM_ENCODER_LAYERS (was 3)"
echo "   Decoder layers: $NUM_DECODER_LAYERS (was 3)"
echo "   FFN dimension: $DIM_FEEDFORWARD (was 2048, -50%)"
echo "   Dropout: $DROPOUT (was 0.2, +50%)"
echo ""
echo "📊 Expected parameters: ~2M (was 7.5M, -73%!)"
echo ""
echo "💡 Why this should work:"
echo "   ✓ Smaller model = less capacity = less overfitting"
echo "   ✓ Larger batch = more stable gradients"
echo "   ✓ Higher dropout = better generalization"
echo "   ✓ Cosine scheduler = smooth LR decay"
echo ""
echo "📈 Previous results:"
echo "   V1: val_loss 4.12, BLEU-4 0.008 (10.8M params, overfitting)"
echo "   V2: val_loss 4.77 (7.5M params, LR collapse)"
echo "   V3: val_loss 4.77 (7.5M params, train/val gap 0.81)"
echo ""
echo "🎯 Target for V4:"
echo "   val_loss < 4.0"
echo "   train/val gap < 0.3"
echo "   BLEU-4 > 0.02-0.05"
echo ""

# Create logs dir
mkdir -p logs

# Set MPS fallback for Apple Silicon
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Activate venv
source .venv/bin/activate

# Launch training
echo "🎯 Launching training..."
echo "⏱️  Estimated time: 4-5 hours (vs 13 hours for V2/V3)"
echo ""

python src/sign_to_text/train_how2sign.py \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --d_model $D_MODEL \
  --num_encoder_layers $NUM_ENCODER_LAYERS \
  --num_decoder_layers $NUM_DECODER_LAYERS \
  --nhead $NHEAD \
  --dim_feedforward $DIM_FEEDFORWARD \
  --dropout $DROPOUT \
  --weight_decay $WEIGHT_DECAY \
  --label_smoothing $LABEL_SMOOTHING \
  --early_stopping_patience $EARLY_STOPPING_PATIENCE \
  --output_dir models/sign_to_text/how2sign_v4 \
  --num_workers 0 \
  2>&1 | tee logs/how2sign_training_v4_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "================================================================================"
echo "✅ TRAINING COMPLETE!"
echo "================================================================================"
echo ""
echo "📊 Final Results:"
echo "   Model saved to: models/sign_to_text/how2sign_v4/"
echo ""
echo "🔍 Next: Evaluate the model"
echo "   python src/sign_to_text/evaluate_how2sign.py \\"
echo "     --checkpoint models/sign_to_text/how2sign_v4/best_checkpoint.pt \\"
echo "     --split val"
echo ""
echo "📈 Compare all versions:"
echo "   V1: val_loss 4.12, BLEU-4 0.008 (10.8M params)"
echo "   V2: val_loss 4.77 (7.5M params)"
echo "   V3: val_loss 4.77 (7.5M params)"
echo "   V4: [check results] (~2M params)"
echo ""
echo "💡 If V4 still overfits, next options:"
echo "   - Add data augmentation"
echo "   - Use pre-trained embeddings"
echo "   - Reduce dataset to 10k samples for faster iteration"
echo ""

