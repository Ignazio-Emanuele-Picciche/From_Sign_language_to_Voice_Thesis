# 🚀 Quick Commands - TTS + Audio Explainability

> **📖 Documentazione Completa**: [`docs/tts_complete_workflow.md`](docs/tts_complete_workflow.md)  
> **Spiegazione dettagliata** di tutto il sistema, glossario, risultati, e troubleshooting.

---

## Test Veloce (con TTS)

```bash
python src/models/two_classes/vivit/test_golden_labels_vivit.py \
  --model_uri mlartifacts/697363764579443849/models/m-de73e05128734690a016c37e5610eeb2/artifacts \
  --batch_size 1 \
  --save_results \
  --generate_tts
```

## Test Standard (solo classificazione)

```bash
python src/models/two_classes/vivit/test_golden_labels_vivit.py \
  --model_uri mlartifacts/697363764579443849/models/m-de73e05128734690a016c37e5610eeb2/artifacts \
  --batch_size 1 \
  --save_results
```

## File Generati

**Risultati classificazione**:

- `results/vivit_golden_labels_test_results_2_classes.csv`
- `results/vivit_golden_labels_metrics_summary_2_classes.csv`

**Risultati TTS** (con `--generate_tts`):

- `results/tts_audio/generated/*.mp3` - Audio files
- `results/tts_audio/baseline/baseline_neutral.mp3` - Baseline
- `results/vivit_tts_audio_analysis_2_classes.csv` - Report validazione

## Struttura Moduli

```
src/
├── tts/
│   ├── emotion_mapper.py       # Emozione → Prosody params
│   ├── text_templates.py       # Template testo TTS
│   └── tts_generator.py        # Generazione audio
│
└── explainability/audio/
    ├── acoustic_analyzer.py    # Estrazione features
    └── prosody_validator.py    # Validazione prosody
```

## Metriche Chiave

- **Pitch Accuracy**: % correttezza modulazione pitch
- **Rate Accuracy**: % correttezza velocità eloquio
- **Volume Accuracy**: % correttezza volume
- **Overall Accuracy**: Media delle 3 metriche

## Threshold

- **Overall Accuracy > 70%** → Prosody applicata correttamente ✅
- **Overall Accuracy < 70%** → Prosody non applicata correttamente ❌
