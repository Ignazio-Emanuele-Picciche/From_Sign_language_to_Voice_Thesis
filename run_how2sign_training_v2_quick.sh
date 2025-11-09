#!/bin/bash
#
# How2Sign Training V2 - QUICK TEST (5 epochs)
# =============================================
#
# Test veloce per verificare che i nuovi hyperparameters funzionino
#

set -e

echo "========================================================================"
echo "🧪 HOW2SIGN TRAINING V2 - QUICK TEST"
echo "========================================================================"

export PYTORCH_ENABLE_MPS_FALLBACK=1
source .venv/bin/activate

python src/sign_to_text/train_how2sign.py \
  --epochs 5 \
  --batch_size 32 \
  --lr 0.0001 \
  --d_model 512 \
  --num_encoder_layers 3 \
  --num_decoder_layers 3 \
  --nhead 8 \
  --dim_feedforward 2048 \
  --dropout 0.2 \
  --weight_decay 0.0001 \
  --label_smoothing 0.1 \
  --output_dir models/sign_to_text/how2sign_v2_test \
  --num_workers 0

echo ""
echo "✅ Test complete! Check if loss is decreasing properly."
echo ""
