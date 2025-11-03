#!/bin/bash

# Quick test del tuning (2 trials, 2 epochs, subset 20%)
# Per verificare che tutto funzioni prima di lanciare tuning completo

echo "🧪 Testing Optuna tuning (2 trials, 2 epochs, 20% subset)"
echo ""

.venv/bin/python src/sign_to_text/tune.py \
    --n_trials 2 \
    --epochs 2 \
    --subset_fraction 0.2 \
    --optimize bleu \
    --study_name test_tuning

echo ""
echo "✅ Test completato! Se funziona, lancia il tuning completo:"
echo "   ./run_tuning.sh"
