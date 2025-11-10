#!/bin/bash
################################################################################
# Prepare All How2Sign Splits for SSVP-SLT
################################################################################
# Prepara automaticamente train, val e test splits
################################################################################

set -e

echo "================================================================================"
echo "🎬 Preparing All How2Sign Splits for SSVP-SLT"
echo "================================================================================"
echo ""

# Configuration
VIDEO_DIR="../../data/raw/how2sign/videos"
OUTPUT_DIR="../../data/how2sign_ssvp"
SPLITS_DIR="../../results/how2sign_splits"

# Check if splits exist
if [ ! -d "$SPLITS_DIR" ]; then
    echo "❌ Splits directory not found: $SPLITS_DIR"
    echo "   Please run How2Sign dataset preparation first."
    exit 1
fi

# Prepare train split
echo "1️⃣  Preparing TRAIN split..."
python ../prepare_how2sign_for_ssvp.py \
    --input_csv "$SPLITS_DIR/train_split.csv" \
    --video_dir "$VIDEO_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --split train

echo ""

# Prepare val split
echo "2️⃣  Preparing VAL split..."
python ../prepare_how2sign_for_ssvp.py \
    --input_csv "$SPLITS_DIR/val_split.csv" \
    --video_dir "$VIDEO_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --split val

echo ""

# Prepare test split (if exists)
if [ -f "$SPLITS_DIR/test_split.csv" ]; then
    echo "3️⃣  Preparing TEST split..."
    python ../prepare_how2sign_for_ssvp.py \
        --input_csv "$SPLITS_DIR/test_split.csv" \
        --video_dir "$VIDEO_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --split test
else
    echo "⚠️  Test split not found, skipping."
fi

echo ""
echo "================================================================================"
echo "✅ All Splits Prepared!"
echo "================================================================================"
echo ""
echo "📍 Output directory: $OUTPUT_DIR"
echo ""
echo "📊 Verify structure:"
echo "   ls -lh $OUTPUT_DIR/clips/"
echo "   ls -lh $OUTPUT_DIR/manifest/"
echo ""
echo "🚀 Next: Download pretrained model"
echo "   python ../download_pretrained.py --model base"
