#!/bin/bash
#
# Run How2Sign Model Evaluation
# =============================

set -e

echo "🔍 How2Sign Model Evaluation"
echo "=============================="
echo ""

# Default paths
CHECKPOINT="${1:-models/sign_to_text/how2sign/best_checkpoint.pt}"
SPLIT="${2:-val}"
OUTPUT_DIR="results/how2sign_evaluation"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Error: Checkpoint not found: $CHECKPOINT"
    echo ""
    echo "Usage: $0 [checkpoint_path] [split]"
    echo "  checkpoint_path: Path to .pt file (default: models/sign_to_text/how2sign/best_checkpoint.pt)"
    echo "  split:           train|val|test (default: val)"
    exit 1
fi

# Create output dir
mkdir -p "$OUTPUT_DIR"

echo "📁 Checkpoint: $CHECKPOINT"
echo "📊 Split:      $SPLIT"
echo "💾 Output:     $OUTPUT_DIR"
echo ""

# Run evaluation
python src/sign_to_text/evaluate_how2sign.py \
    --checkpoint "$CHECKPOINT" \
    --split "$SPLIT" \
    --batch_size 8 \
    --num_examples 20 \
    --save_examples "$OUTPUT_DIR/examples_${SPLIT}.txt" \
    --save_metrics "$OUTPUT_DIR/metrics_${SPLIT}.json" \
    --num_workers 0

echo ""
echo "✅ Evaluation complete!"
echo "📊 Metrics: $OUTPUT_DIR/metrics_${SPLIT}.json"
echo "💬 Examples: $OUTPUT_DIR/examples_${SPLIT}.txt"
echo ""
