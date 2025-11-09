#!/bin/bash
#
# How2Sign Training V4 - Quick Test (3 epochs)
# =============================================
#
# Quick test to verify smaller model works before full training
# Duration: ~10 minutes
#

set -e

echo "========================================================================"
echo "🧪 HOW2SIGN TRAINING V4 - QUICK TEST (3 epochs)"
echo "========================================================================"

# Configuration - SMALLER MODEL
EPOCHS=3
BATCH_SIZE=64
LR=0.0001
D_MODEL=256
NUM_ENCODER_LAYERS=2
NUM_DECODER_LAYERS=2
NHEAD=8
DIM_FEEDFORWARD=1024
DROPOUT=0.3
WEIGHT_DECAY=0.0001
LABEL_SMOOTHING=0.1

echo ""
echo "📋 Quick Test - Smaller Model:"
echo "   d_model: $D_MODEL (was 512)"
echo "   layers: $NUM_ENCODER_LAYERS + $NUM_DECODER_LAYERS (was 3+3)"
echo "   batch: $BATCH_SIZE (was 32)"
echo "   dropout: $DROPOUT (was 0.2)"
echo ""
echo "⏱️  Expected: ~10 minutes"
echo ""

# Create logs dir
mkdir -p logs

# Set MPS fallback
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
  --output_dir models/sign_to_text/how2sign_v4_quick \
  --num_workers 0

echo ""
echo "================================================================================"
echo "✅ QUICK TEST COMPLETE!"
echo "================================================================================"
echo ""
echo "📊 Check results:"
echo "   cat models/sign_to_text/how2sign_v4_quick/history.json | python3 -c \"import sys, json; h=json.load(sys.stdin); print(f'Val Loss: {h[\\\"val_loss\\\"][-1]:.4f}'); print(f'Train/Val gap: {h[\\\"train_loss\\\"][-1] - h[\\\"val_loss\\\"][-1]:.4f}')\""
echo ""
echo "🎯 Target signs:"
echo "   ✓ Val loss improving (not plateauing)"
echo "   ✓ Train/val gap < 0.5"
echo "   ✓ No memory errors with batch_size=64"
echo ""
echo "🚀 If looks good, launch full training:"
echo "   ./run_how2sign_training_v4.sh"
echo ""

