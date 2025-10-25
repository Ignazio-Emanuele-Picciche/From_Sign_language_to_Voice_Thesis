# TTS Audio Explainability - Workflow Visuale

## 🔄 ARCHITETTURA DEL SISTEMA

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FASE 1: INFERENZA VIDEO                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  📹 Video Sign Language (ASLLRP)                                    │
│         │                                                            │
│         ├─ Frame Extraction (8 FPS)                                 │
│         │                                                            │
│         ├─ ViViT Image Processor                                    │
│         │    └─ Resize, Normalize, Tensor conversion                │
│         │                                                            │
│         ├─ ViViT Model (Video Vision Transformer)                   │
│         │    ├─ Spatial attention (per-frame)                       │
│         │    ├─ Temporal attention (across frames)                  │
│         │    └─ Classification head                                 │
│         │                                                            │
│         └─ OUTPUT:                                                  │
│              ├─ Emotion: "Positive" | "Negative"                    │
│              ├─ Confidence: 0.0 - 1.0 (or 0-100%)                   │
│              └─ Logits: [logit_positive, logit_negative]            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  FASE 2: GENERAZIONE AUDIO TTS                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  📊 Emotion + Confidence                                            │
│         │                                                            │
│         ├─ [src/tts/emotion_mapper.py]                             │
│         │    Emotion-to-Prosody Mapping                             │
│         │    ┌──────────────────────────────────────┐               │
│         │    │ Positive: rate=+15%, pitch=+8%, vol=+5%│             │
│         │    │ Negative: rate=-12%, pitch=-6%, vol=-3%│             │
│         │    │ Scaled by confidence                  │             │
│         │    └──────────────────────────────────────┘               │
│         │         ↓                                                 │
│         │    Prosody Params: {rate, pitch, volume}                  │
│         │                                                            │
│         ├─ [src/tts/text_templates.py]                             │
│         │    Text Preparation                                        │
│         │    ┌──────────────────────────────────────┐               │
│         │    │ 1. Get caption from dataset          │               │
│         │    │ 2. Clean special characters          │               │
│         │    │    - Remove quotes, backticks        │               │
│         │    │    - Normalize apostrophes           │               │
│         │    │ 3. Return clean text                 │               │
│         │    └──────────────────────────────────────┘               │
│         │         ↓                                                 │
│         │    Clean Text: "I was happy about the news"               │
│         │                                                            │
│         └─ [src/tts/tts_generator.py]                              │
│              TTS Generation                                          │
│              ┌────────────────────────────────────┐                 │
│              │ Edge-TTS (Microsoft Neural Voices) │                 │
│              │                                     │                 │
│              │ 1. Convert params:                 │                 │
│              │    - pitch: % → Hz (+8% → +12Hz)  │                 │
│              │    - rate: float → int (+14.2% → +14%)│              │
│              │    - volume: float → int           │                 │
│              │                                     │                 │
│              │ 2. Synthesize:                     │                 │
│              │    edge_tts.Communicate(           │                 │
│              │        text=text,                  │                 │
│              │        voice="en-US-AriaNeural",   │                 │
│              │        rate="+14%",                │                 │
│              │        pitch="+12Hz",              │                 │
│              │        volume="+4%"                │                 │
│              │    )                               │                 │
│              │                                     │                 │
│              │ 3. Save to file                    │                 │
│              └────────────────────────────────────┘                 │
│                   ↓                                                 │
│              🔊 Audio File (.mp3)                                   │
│                 results/tts_audio/generated/                         │
│                 {video_name}_{emotion}.mp3                          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│           FASE 3: AUDIO EXPLAINABILITY & VALIDATION                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  🔊 Generated Audio + 📊 Target Prosody                             │
│         │                                                            │
│         ├─ [src/explainability/audio/acoustic_analyzer.py]         │
│         │    Feature Extraction                                      │
│         │    ┌─────────────────────────────────────┐                │
│         │    │ Praat-Parselmouth:                  │                │
│         │    │  ├─ Pitch (F0): mean, std, range   │                │
│         │    │  ├─ Jitter: pitch variability      │                │
│         │    │  ├─ Shimmer: amplitude variability │                │
│         │    │  └─ HNR: harmonic-to-noise ratio   │                │
│         │    │                                     │                │
│         │    │ Librosa:                            │                │
│         │    │  ├─ Energy: RMS, mean, std, max    │                │
│         │    │  ├─ Rate: onset detection*         │                │
│         │    │  └─ Duration: total length         │                │
│         │    └─────────────────────────────────────┘                │
│         │         ↓                                                 │
│         │    Features: {pitch_hz, energy_db, rate_syll_sec}         │
│         │                                                            │
│         ├─ [src/explainability/audio/prosody_validator.py]         │
│         │    Validation                                              │
│         │    ┌─────────────────────────────────────┐                │
│         │    │ 1. Compare generated vs baseline   │                │
│         │    │ 2. Calculate delta percentages     │                │
│         │    │ 3. Compare with target params      │                │
│         │    │ 4. Compute accuracy metrics        │                │
│         │    └─────────────────────────────────────┘                │
│         │         ↓                                                 │
│         │    Validation Report:                                     │
│         │      {pitch_accuracy, rate_accuracy*, volume_accuracy}    │
│         │                                                            │
│         └─ [src/analysis/run_analysis.py]                          │
│              Statistical Analysis                                    │
│              ┌─────────────────────────────────────┐                │
│              │ 1. Collect all features (n=200)    │                │
│              │                                     │                │
│              │ 2. Group by emotion:                │                │
│              │    - Positive: n=160                │                │
│              │    - Negative: n=40                 │                │
│              │                                     │                │
│              │ 3. Descriptive statistics:          │                │
│              │    - Mean, Std, Min, Max            │                │
│              │                                     │                │
│              │ 4. Statistical tests:               │                │
│              │    - Shapiro-Wilk (normality)      │                │
│              │    - Independent t-test            │                │
│              │    - Cohen's d (effect size)       │                │
│              │                                     │                │
│              │ 5. Visualizations:                  │                │
│              │    - Box plots                      │                │
│              │    - Swarm plots                    │                │
│              │    - Statistical summary            │                │
│              └─────────────────────────────────────┘                │
│                   ↓                                                 │
│              📊 Results:                                            │
│                 - audio_analysis_results.csv                         │
│                 - emotion_comparison_plots.png                       │
│                 - statistical_report.txt                             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
                    ✅ VALIDAZIONE COMPLETA

              Pitch: p<0.001*** (Significant!)
              Cohen's d = 0.637 (Medium effect)

              → Sistema funziona correttamente ✓

* Note: Speaking rate non funziona su audio TTS (limitazione tecnica)
```

---

## 📁 FILE MAPPING

```
test_golden_labels_vivit.py  ──┐
                               ├─→ [FASE 1] Video → Emotion
video_dataset.py              ─┘

emotion_mapper.py            ──┐
text_templates.py            ──┼─→ [FASE 2] Emotion → Audio
tts_generator.py             ──┘

acoustic_analyzer.py         ──┐
prosody_validator.py         ──┼─→ [FASE 3] Audio → Validation
audio_comparison.py          ──┤
statistical_tests.py         ──┤
run_analysis.py              ──┘

analyze_golden_labels_audio.sh → [SHORTCUT] Run all analysis
```

---

## 🔢 DATA FLOW

```
Input:
  Video: 256351.mp4 (sign language)
  Golden Label: "Positive"
  Caption: "I was like, "Oh, wow, that`s fine.""

↓ FASE 1: ViViT Inference

Output:
  Predicted: "Positive"
  Confidence: 0.923 (92.3%)

↓ FASE 2: TTS Generation

Intermediate:
  Prosody: {rate: "+13.8%", pitch: "+7.4%", volume: "+4.6%"}
  Clean Text: "I was like, Oh, wow, that's fine."

Output:
  Audio: 256351_positive.mp3
  Duration: 2.8 seconds
  Size: ~220 KB

↓ FASE 3: Analysis

Features Extracted:
  Pitch: 221.5 Hz
  Energy: -28.3 dB
  Rate: 0.0 syll/sec*

Validation:
  Target pitch: +8% → Measured: +7.4% → Accuracy: 92.5%

Statistical Analysis (across all n=200):
  Positive pitch: 219.7 ± 8.9 Hz
  Negative pitch: 214.0 ± 8.7 Hz
  Difference: +2.6% (p<0.001***)

Final Output:
  ✅ Sistema valido: Differenze significative rilevate
```

---

## 🎯 DECISION TREE: "Quando Usare Quale Parte?"

```
Vuoi...

├─ Classificare nuovi video?
│  └─→ Usa: test_golden_labels_vivit.py (senza --generate_tts)
│
├─ Generare audio da video?
│  └─→ Usa: test_golden_labels_vivit.py --generate_tts
│
├─ Generare audio standalone (senza video)?
│  └─→ Usa: src/tts/tts_generator.py direttamente
│      Example:
│        from src.tts.tts_generator import generate_emotional_audio
│        generate_emotional_audio("Positive", 0.95, "test", "output/", caption="Hello")
│
├─ Analizzare audio già generati?
│  └─→ Usa: ./analyze_golden_labels_audio.sh
│      O: python src/analysis/run_analysis.py --audio_dir <path>
│
├─ Test veloce (pochi sample)?
│  └─→ Genera 8 audio manualmente + run_analysis.py
│
├─ Modificare parametri prosodici?
│  └─→ Edita: src/tts/emotion_mapper.py (PROSODY_MAPPING)
│
├─ Cambiare voce TTS?
│  └─→ Edita: src/tts/tts_generator.py (DEFAULT_VOICE)
│      Opzioni: en-US-AriaNeural, en-US-GuyNeural, etc.
│
└─ Aggiungere nuove features acustiche?
   └─→ Edita: src/explainability/audio/acoustic_analyzer.py
```

---

## 🔧 TROUBLESHOOTING VISUALE

```
Problema: "Audio dice 'quote', 'slash', 'backtick'"
  ↓
Causa: Caratteri speciali nel caption
  ↓
Dove guardare: src/tts/text_templates.py
  ↓
Funzione: clean_text_for_tts()
  ↓
Fix: ✅ Già implementato
  ├─ Rimuove: " ' ` / \ | [ ] { } < > _ * # @ &
  └─ Normalizza spazi e apostrofi

────────────────────────────────────────────

Problema: "Speaking rate sempre 0.0"
  ↓
Causa: Onset detection (librosa) non funziona su TTS
  ↓
Dove guardare: src/explainability/audio/acoustic_analyzer.py
  ↓
Funzione: extract_rate_features()
  ↓
Fix: ❌ Limitazione tecnica
  └─ Soluzione: Usa solo pitch ed energy

────────────────────────────────────────────

Problema: "Edge-TTS error: Invalid pitch '+7%'"
  ↓
Causa: Pitch deve essere in Hz, non percentuale
  ↓
Dove guardare: src/tts/tts_generator.py
  ↓
Funzione: convert_pitch_to_hz()
  ↓
Fix: ✅ Già implementato
  └─ Converte: +8% → +12Hz (baseline 150Hz)

────────────────────────────────────────────

Problema: "p-value non significativo"
  ↓
Possibili cause:
  ├─ Sample size troppo piccolo (n<30)
  ├─ Alta variabilità intra-gruppo
  ├─ Effect size realmente piccolo
  └─ Dataset sbilanciato
  ↓
Soluzioni:
  ├─ Aumenta n (genera più audio)
  ├─ Aumenta parametri prosodici (+25% invece di +15%)
  ├─ Bilancia dataset (50% Positive, 50% Negative)
  └─ Riporta come limitazione in tesi
```

---

## 📊 PERFORMANCE METRICS FLOW

```
Generated Audio (n=200)
        │
        ├─ Feature Extraction
        │    ├─ Pitch: 5-10 sec/file
        │    ├─ Energy: 1-2 sec/file
        │    └─ Rate: 2-3 sec/file*
        │
        ├─ Statistical Tests
        │    ├─ Shapiro-Wilk: <1 sec
        │    ├─ t-test: <1 sec
        │    └─ Cohen's d: <1 sec
        │
        └─ Visualization
             ├─ Box plots: 5-10 sec
             └─ Save PNG: 1-2 sec

Total Time: ~20-30 min for 200 files

* Rate detection attempts but returns 0
```

---

## 🎓 THESIS INTEGRATION FLOWCHART

```
Thesis Chapter/Section
        │
        ├─ Introduction
        │    └─ Mention: Multimodal emotion transfer (sign → speech)
        │
        ├─ Related Work
        │    ├─ Sign language emotion recognition
        │    ├─ TTS with emotion
        │    └─ Audio explainability (novel contribution)
        │
        ├─ Methodology
        │    ├─ 3.1: ViViT for emotion classification
        │    ├─ 3.2: Emotion-to-Prosody mapping
        │    ├─ 3.3: TTS generation (Edge-TTS)
        │    └─ 3.4: Audio explainability framework ← NEW SECTION
        │         ├─ Acoustic feature extraction
        │         ├─ Statistical validation
        │         └─ Implementation details
        │
        ├─ Results
        │    ├─ 4.1: ViViT classification results
        │    ├─ 4.2: TTS generation results (200 audio files)
        │    └─ 4.3: Audio explainability results ← NEW SECTION
        │         ├─ TABLE: Descriptive statistics
        │         ├─ FIGURE: Box plots comparison
        │         ├─ TABLE: Statistical test results
        │         └─ TEXT: Interpretation
        │
        ├─ Discussion
        │    ├─ Validation successful (p<0.001 for pitch)
        │    ├─ Limitations (speaking rate, dataset imbalance)
        │    └─ Practical implications
        │
        └─ Conclusion & Future Work
             ├─ Novel contribution validated
             ├─ Suggestions: neural TTS, more emotions, balanced dataset
             └─ Applications: accessibility, assistive tech
```

---

**Ultimo aggiornamento**: 23 Ottobre 2025  
**Autore**: Ignazio Emanuele Picciche
