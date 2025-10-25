#!/bin/bash

# Script per analizzare gli audio generati dalle golden labels

echo "=========================================="
echo "ANALISI AUDIO GOLDEN LABELS"
echo "=========================================="
echo ""

# Directory degli audio
AUDIO_DIR="results/tts_audio"
OUTPUT_DIR="results/tts_analysis"

# Controlla se esistono audio
if [ ! -d "$AUDIO_DIR" ] || [ -z "$(ls -A $AUDIO_DIR 2>/dev/null)" ]; then
    echo "⚠️  Nessun audio trovato in $AUDIO_DIR"
    echo "Genera prima gli audio con:"
    echo "  python src/models/two_classes/vivit/test_golden_labels_vivit.py --generate_tts"
    exit 1
fi

# Conta gli audio
NUM_AUDIO=$(find "$AUDIO_DIR" -name "*.mp3" | wc -l)
echo "📊 Trovati $NUM_AUDIO file audio"
echo ""

# Esegui analisi
echo "Esecuzione analisi..."
.venv/bin/python src/analysis/run_analysis.py \
    --audio_dir "$AUDIO_DIR" \
    --output_dir "$OUTPUT_DIR"

# Check risultati
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ ANALISI COMPLETATA"
    echo "=========================================="
    echo ""
    echo "Risultati salvati in: $OUTPUT_DIR"
    echo ""
    echo "File generati:"
    echo "  📊 CSV:    $OUTPUT_DIR/audio_analysis_results.csv"
    echo "  📈 Grafici: $OUTPUT_DIR/emotion_comparison_plots.png"
    echo "  📄 Report:  $OUTPUT_DIR/statistical_report.txt"
    echo ""
    echo "Apri i grafici con:"
    echo "  open $OUTPUT_DIR/emotion_comparison_plots.png"
else
    echo ""
    echo "⚠️  Errore durante l'analisi"
    exit 1
fi
