#!/bin/bash
#
# How2Sign Training V3 - CosineAnnealingLR Scheduler
# ===================================================
#
# Fix dello scheduler: CosineAnnealingLR invece di ReduceLROnPlateau
# - ReduceLROnPlateau troppo aggressivo (factor=0.5, patience=3)
# - Cosine: decay dolce e prevedibile da LR iniziale a eta_min
# - Niente crolli improvvisi del learning rate
#

set -e

echo "========================================================================"
echo "🚀 HOW2SIGN TRAINING V3 - COSINE ANNEALING SCHEDULER"
echo "========================================================================"

# Configuration
EPOCHS=70
BATCH_SIZE=32
LR=0.0001
D_MODEL=512
NUM_ENCODER_LAYERS=3
NUM_DECODER_LAYERS=3
NHEAD=8
DIM_FEEDFORWARD=2048
DROPOUT=0.2
WEIGHT_DECAY=0.0001
LABEL_SMOOTHING=0.1

echo ""
echo "📋 Configuration:"
echo "   Epochs: $EPOCHS"
echo "   Batch Size: $BATCH_SIZE"
echo "   Learning Rate: $LR → 1e-6 (cosine decay)"
echo "   Model: d_model=$D_MODEL, layers=$NUM_ENCODER_LAYERS+$NUM_DECODER_LAYERS"
echo "   Regularization: dropout=$DROPOUT, wd=$WEIGHT_DECAY"
echo ""
echo "🔧 Key Fix:"
echo "   ✓ CosineAnnealingLR instead of ReduceLROnPlateau"
echo "   ✓ Smooth decay: 1e-4 → 1e-6 over 70 epochs"
echo "   ✓ No sudden LR drops (was: 1e-4 → 1e-8 in 7 epochs!)"
echo ""
echo "📊 Previous issues:"
echo "   ❌ V1: val_loss 4.12, aggressive LR decay"
echo "   ❌ V2: val_loss 4.77, LR collapsed to 1e-8 by epoch 7"
echo ""

# Create logs dir
mkdir -p logs

# Set MPS fallback for Apple Silicon
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Activate venv
source .venv/bin/activate

# Launch training
echo "🎯 Launching training..."
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
  --output_dir models/sign_to_text/how2sign_v3 \
  --num_workers 0 \
  2>&1 | tee logs/how2sign_training_v3_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "================================================================================"
echo "✅ TRAINING COMPLETE!"
echo "================================================================================"
echo ""
echo "📊 Final Results:"
echo "   Model saved to: models/sign_to_text/how2sign_v3/"
echo ""
echo "🔍 Next: Evaluate the model"
echo "   python src/sign_to_text/evaluate_how2sign.py \\"
echo "     --checkpoint models/sign_to_text/how2sign_v3/best_checkpoint.pt \\"
echo "     --split val"
echo ""
echo "📈 Compare with previous versions:"
echo "   V1: val_loss 4.12, BLEU-4 0.008"
echo "   V2: val_loss 4.77 (worse!)"
echo "   V3: [check after evaluation]"
echo ""

