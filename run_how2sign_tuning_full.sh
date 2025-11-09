#!/bin/bash
#
# How2Sign Hyperparameter Tuning - Full
# ======================================
#
# Full tuning: 30 trials × 5 epochs (~4-6 ore)
#

set -e

echo "========================================================================"
echo "🔍 HOW2SIGN HYPERPARAMETER TUNING - FULL"
echo "========================================================================"

# Config
N_TRIALS=40
EPOCHS=7

echo ""
echo "📋 Configuration:"
echo "   Trials: $N_TRIALS"
echo "   Epochs per trial: $EPOCHS"
echo "   Dataset: Full (29.6k samples train, 1.7k val)"
echo "   Estimated time: 4-6 hours"
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
  --study_name "how2sign_tuning_full" \
  --max_frames 150 \
  --use_mlflow \
  2>&1 | tee logs/how2sign_tuning_full_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "✅ Tuning complete!"
echo "   Best params: results/how2sign_tuning/best_hyperparameters.json"
echo "   MLflow UI: mlflow ui"
echo ""
echo "🚀 Next: Launch training with best params"
echo "   python src/sign_to_text/train_how2sign_from_tuning.py --epochs 50"
echo ""
