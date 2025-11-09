#!/bin/bash
#
# How2Sign Training V2 - Optimized Hyperparameters
# =================================================
#
# Hyperparameters ottimizzati basati su analisi training precedente:
# - LR più basso (1e-4 invece di 3.7e-4)
# - Batch size maggiore (32 invece di 16)
# - Architettura più leggera (3+3 layers invece di 4+4)
# - Dropout moderato (0.2)
# - Scheduler più gentile (CosineAnnealing)
#

set -e

echo "========================================================================"
echo "🚀 HOW2SIGN TRAINING V2 - OPTIMIZED HYPERPARAMETERS"
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
echo "   Learning Rate: $LR"
echo "   Model: d_model=$D_MODEL, layers=$NUM_ENCODER_LAYERS+$NUM_DECODER_LAYERS"
echo "   Regularization: dropout=$DROPOUT, wd=$WEIGHT_DECAY"
echo ""
echo "💡 Improvements vs previous:"
echo "   ✓ Lower LR (1e-4 vs 3.7e-4) → more stable"
echo "   ✓ Larger batch (32 vs 16) → smoother gradients"
echo "   ✓ Fewer layers (3+3 vs 4+4) → less overfitting"
echo "   ✓ Moderate dropout (0.2) → better generalization"
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
  --output_dir models/sign_to_text/how2sign_v2 \
  --num_workers 0 \
  2>&1 | tee logs/how2sign_training_v2_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "✅ Training complete!"
echo "   Model saved to: models/sign_to_text/how2sign_v2/"
echo ""
echo "🔍 Next: Evaluate the model"
echo "   python src/sign_to_text/evaluate_how2sign.py \\"
echo "     --checkpoint models/sign_to_text/how2sign_v2/best_checkpoint.pt \\"
echo "     --split val"
echo ""
