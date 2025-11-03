#!/bin/bash

# Sign-to-Text Hyperparameter Tuning Script
# ==========================================
#
# Quick tuning runs con configurazioni predefinite

set -e

echo "=================================="
echo "🔧 Sign-to-Text Hyperparameter Tuning"
echo "=================================="
echo ""

# Default: 30 trials, 5 epochs, optimize BLEU
echo "Running tuning with:"
echo "  - 30 trials"
echo "  - 5 epochs per trial"
echo "  - Optimize: BLEU"
echo "  - Full training set"
echo ""

.venv/bin/python src/sign_to_text/tune.py \
    --n_trials 30 \
    --epochs 5 \
    --optimize bleu \
    --subset_fraction 1.0 \
    --study_name sign_to_text_tuning

echo ""
echo "✅ Tuning completato!"
echo "📊 Controlla i risultati in:"
echo "   - results/best_hyperparameters.json"
echo "   - results/param_importances.html"
echo "   - MLflow UI: mlflow ui (http://localhost:5000)"
echo ""
