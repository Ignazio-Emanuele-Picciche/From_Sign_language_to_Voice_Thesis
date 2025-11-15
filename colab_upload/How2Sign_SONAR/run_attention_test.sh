#!/bin/bash
# Quick 2-epoch test to validate attention bridge fix

echo "=========================================="
echo "ATTENTION BRIDGE FIX - QUICK TEST"
echo "=========================================="
echo ""
echo "Testing with 2 epochs to check if mode collapse is resolved"
echo ""

# First run validation test
echo "Step 1: Validating model architecture..."
python test_attention_fix.py

if [ $? -ne 0 ]; then
    echo "❌ Architecture validation failed!"
    exit 1
fi

echo ""
echo "Step 2: Running 2-epoch training..."
echo ""

python train_sonar_with_t5.py \
    --epochs 2 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --warmup_steps 100 \
    --save_dir checkpoints/sonar_t5_attention_test \
    --log_file logs/train_attention_test.log

echo ""
echo "=========================================="
echo "TRAINING COMPLETE"
echo "=========================================="
echo ""
echo "Check the output above for:"
echo "  1. Sample predictions should be DIFFERENT (not all the same)"
echo "  2. BLEU score should be >2% (ideally 5-10%)"
echo "  3. Loss should be reasonable (1.5-2.5)"
echo ""
echo "If outputs are still identical → need more architectural changes"
echo "If outputs are diverse → proceed with full 20-epoch training"
