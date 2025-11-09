#!/bin/bash
#
# How2Sign Hyperparameter Tuning
# ===============================
#
# Quick tuning: 10 trials × 3 epochs (~1-2 ore)
#

set -e

echo "========================================================================"
echo "🔍 HOW2SIGN HYPERPARAMETER TUNING - QUICK"
echo "========================================================================"

# Config
N_TRIALS=10
EPOCHS=3
USE_SUBSET="--use_subset"  # Usa subset per velocità

echo ""
echo "📋 Configuration:"
echo "   Trials: $N_TRIALS"
echo "   Epochs per trial: $EPOCHS"
echo "   Dataset: Subset (5k samples train, 500 val)"
echo "   Estimated time: 1-2 hours"
echo ""

# Create logs dir
mkdir -p logs

# Activate venv
source .venv/bin/activate

# Set MPS fallback for Apple Silicon
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Launch tuning
echo "🎯 Launching tuning..."
echo ""

python src/sign_to_text/tune_how2sign.py \
  --n_trials $N_TRIALS \
  --epochs $EPOCHS \
  $USE_SUBSET \
  --study_name "how2sign_tuning_quick" \
  --max_frames 150 \
  2>&1 | tee logs/how2sign_tuning_quick_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "✅ Tuning complete!"
echo "   Best params: results/how2sign_tuning/best_hyperparameters.json"
echo ""
echo "🚀 Next: Launch training with best params"
echo "   python src/sign_to_text/train_how2sign_from_tuning.py --epochs 50"
echo ""
