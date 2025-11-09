#!/bin/bash
#
# How2Sign Training V3 - Quick Test (5 epochs)
# =============================================
#
# Test veloce per verificare che CosineAnnealingLR funzioni correttamente.
# Durata: ~15 minuti
#

set -e

echo "========================================================================"
echo "🧪 HOW2SIGN TRAINING V3 - QUICK TEST (5 epochs)"
echo "========================================================================"

# Configuration
EPOCHS=5
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
echo "📋 Quick Test Configuration:"
echo "   Epochs: $EPOCHS (only 5 for quick test)"
echo "   Batch Size: $BATCH_SIZE"
echo "   Learning Rate: $LR → 1e-6 (cosine decay)"
echo "   Scheduler: CosineAnnealingLR"
echo ""
echo "⏱️  Expected duration: ~15 minutes"
echo ""

# Create logs dir
mkdir -p logs

# Set MPS fallback for Apple Silicon
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Activate venv
source .venv/bin/activate

# Launch training
echo "🎯 Launching quick test..."
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
  --output_dir models/sign_to_text/how2sign_v3_quick \
  --num_workers 0

echo ""
echo "================================================================================"
echo "✅ QUICK TEST COMPLETE!"
echo "================================================================================"
echo ""
echo "📊 Check if LR decays smoothly:"
echo "   cat models/sign_to_text/how2sign_v3_quick/history.json | python3 -c \"import sys, json; h=json.load(sys.stdin); [print(f'Epoch {i+1}: LR={lr:.2e}') for i, lr in enumerate(h['lr'])]\""
echo ""
echo "🔍 If LR decays smoothly (no sudden drops), proceed with full training:"
echo "   ./run_how2sign_training_v3.sh"
echo ""

