#!/bin/bash
#
# Launch How2Sign Training
# ========================
#
# Training completo Sign-to-Text su How2Sign (31k samples)
#

set -e

echo "========================================================================"
echo "🚀 HOW2SIGN TRAINING LAUNCHER"
echo "========================================================================"

# Config
EPOCHS=30
BATCH_SIZE=16
LR=1e-4
D_MODEL=512
NUM_ENCODER_LAYERS=4
NUM_DECODER_LAYERS=4
NHEAD=8
DROPOUT=0.1
MAX_FRAMES=200
MAX_CAPTION_LEN=50

echo ""
echo "📋 Configuration:"
echo "   Epochs: $EPOCHS"
echo "   Batch size: $BATCH_SIZE"
echo "   Learning rate: $LR"
echo "   Model dim: $D_MODEL"
echo "   Encoder layers: $NUM_ENCODER_LAYERS"
echo "   Decoder layers: $NUM_DECODER_LAYERS"
echo "   Attention heads: $NHEAD"
echo "   Max frames: $MAX_FRAMES"
echo "   Max caption length: $MAX_CAPTION_LEN"
echo ""

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
  --dropout $DROPOUT \
  --max_frames $MAX_FRAMES \
  --max_caption_len $MAX_CAPTION_LEN \
  --num_workers 4 \
  --grad_clip 1.0 \
  --label_smoothing 0.1 \
  --weight_decay 1e-4 \
  2>&1 | tee logs/how2sign_training_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "✅ Training complete!"
echo ""
