#!/bin/bash
# Complete training command for SONAR+T5 with attention bridge
# Run this on Google Colab after uploading all files

echo "=========================================="
echo "SONAR + T5 Training with Attention Bridge"
echo "=========================================="
echo ""

# Quick 2-epoch test first (RECOMMENDED!)
echo "Step 1: Quick 2-epoch test to check if mode collapse is fixed"
echo ""

python train_sonar_with_t5.py \
    --sonar_checkpoint checkpoints/sonar_encoder_finetuned/best_encoder.pt \
    --train_features features/train \
    --train_manifest manifests/train.tsv \
    --val_features features/val \
    --val_manifest manifests/val.tsv \
    --output_dir checkpoints/sonar_t5_attention_test \
    --t5_model t5-small \
    --freeze_encoder \
    --epochs 2 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --warmup_steps 100 \
    --device cuda

echo ""
echo "=========================================="
echo "CHECK THE RESULTS ABOVE!"
echo "=========================================="
echo ""
echo "If the 5 sample predictions are DIFFERENT → proceed with full training"
echo "If they are all IDENTICAL → mode collapse persists"
echo ""
echo "Press Enter to continue with full 20-epoch training, or Ctrl+C to stop"
read

# Full 20-epoch training (only if test passes)
echo ""
echo "Step 2: Full 20-epoch training"
echo ""

python train_sonar_with_t5.py \
    --sonar_checkpoint checkpoints/sonar_encoder_finetuned/best_encoder.pt \
    --train_features features/train \
    --train_manifest manifests/train.tsv \
    --val_features features/val \
    --val_manifest manifests/val.tsv \
    --output_dir checkpoints/sonar_t5_attention_final \
    --t5_model t5-small \
    --freeze_encoder \
    --epochs 20 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --warmup_steps 500 \
    --device cuda

echo ""
echo "=========================================="
echo "TRAINING COMPLETE!"
echo "=========================================="
echo ""
echo "Best model saved to: checkpoints/sonar_t5_attention_final/best_model.pt"
