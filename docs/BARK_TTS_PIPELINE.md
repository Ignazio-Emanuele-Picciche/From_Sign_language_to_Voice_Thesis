# 🎵 Pipeline ViViT → Bark TTS - Documentazione Completa

## 📋 Panoramica

Sistema completo per generare audio emotivo da video di sign language:

```
Video Sign Language → ViViT Model → Predizione Emozione → Bark TTS → Audio Emotivo
```

## 🏗️ Architettura

### Struttura File System

```
src/tts/
├── bark/                         # 🆕 Bark TTS Engine
│   ├── __init__.py
│   ├── emotion_mapper.py         # Mapping emozioni → speaker prompts
│   ├── tts_generator.py          # Generazione audio con Bark
│   └── README.md
├── __init__.py                   # Edge-TTS (legacy)
├── emotion_mapper.py             # Edge-TTS mapping
├── tts_generator.py              # Edge-TTS generator
└── text_templates.py             # Template di testo condivisi
```

## 🎯 Mapping Emozioni → Bark

### Configurazione Speaker

| Emozione     | Speaker           | Voce                    | Tag Emotivo | Temperature |
| ------------ | ----------------- | ----------------------- | ----------- | ----------- |
| **Positive** | `v2/en_speaker_6` | Femminile energica      | `[laughs]`  | 0.7         |
| **Negative** | `v2/en_speaker_3` | Maschile calma          | `[sighs]`   | 0.6         |
| **Neutral**  | `v2/en_speaker_9` | Narratore professionale | _(none)_    | 0.5         |

### Speaker Alternativi

Ogni emozione ha 3 speaker disponibili per varietà:

```python
ALTERNATIVE_SPEAKERS = {
    "Positive": ["v2/en_speaker_6", "v2/en_speaker_5", "v2/en_speaker_7"],
    "Negative": ["v2/en_speaker_3", "v2/en_speaker_1", "v2/en_speaker_4"],
    "Neutral": ["v2/en_speaker_9", "v2/en_speaker_0", "v2/en_speaker_2"],
}
```

## 🚀 Utilizzo

### 1. Test Standalone

```bash
# Test emotion mapper
python src/tts/bark/emotion_mapper.py

# Test generazione audio
python test_tts_bark.py
```

### 2. Pipeline Completa con ViViT

```bash
python src/models/two_classes/vivit/test_golden_labels_vivit.py \
  --model_uri mlartifacts/697363764579443849/models/m-de73e05128734690a016c37e5610eeb2/artifacts \
  --batch_size 1 \
  --save_results \
  --generate_tts
```

**Output:**

- Video analizzati: 200 campioni
- Audio generati: `results/tts_audio/bark_generated/*.wav`
- CSV risultati: `results/vivit_tts_bark_2_classes.csv`

### 3. Uso Programmatico

```python
from src.tts.bark.tts_generator import generate_emotional_audio, preload_bark_models
from src.tts.bark.emotion_mapper import map_emotion_to_bark_prompt

# Step 1: Pre-carica modelli (raccomandato)
preload_bark_models()  # Carica ~10GB in RAM

# Step 2: Ottieni predizione da ViViT
emotion = "Positive"      # Output di ViViT
confidence = 0.92         # Confidenza predizione
caption = "Hello world!"  # Testo del sign language

# Step 3: Genera audio emotivo
audio_path = generate_emotional_audio(
    emotion=emotion,
    confidence=confidence,
    video_name="video_001",
    output_dir="results/audio",
    caption=caption,
    use_emotional_tags=True,  # Usa [laughs], [sighs]
    alternative_speaker=0,     # 0-2 per varietà
    preload=True,              # Usa modelli pre-caricati
)

print(f"Audio generato: {audio_path}")
# Output: results/audio/video_001_positive.wav
```

## 📊 Esempio Output Pipeline

### Caso 1: Emozione Positiva

```
[1/200] Processing: video_001.mp4
  Emotion: Positive (confidence: 92.3%)
  Caption: "Thank you so much for your help!"
  Bark Speaker: v2/en_speaker_6
  Emotional tag: [laughs]
  ✅ Audio generato: video_001_positive.wav
```

**Audio generato:**

- Speaker: Voce femminile energica
- Tag: `[laughs] Thank you so much for your help!`
- Durata: ~3-5 secondi
- Formato: WAV 24kHz

### Caso 2: Emozione Negativa

```
[2/200] Processing: video_002.mp4
  Emotion: Negative (confidence: 87.5%)
  Caption: "I'm sorry, I can't help you with that."
  Bark Speaker: v2/en_speaker_3
  Emotional tag: [sighs]
  ✅ Audio generato: video_002_negative.wav
```

**Audio generato:**

- Speaker: Voce maschile calma
- Tag: `[sighs] I'm sorry, I can't help you with that.`
- Durata: ~4-6 secondi
- Formato: WAV 24kHz

## 📈 CSV Output

File: `results/vivit_tts_bark_2_classes.csv`

| video_name    | emotion  | confidence | bark_speaker    | emotional_tag | temperature | audio_path                 | caption      |
| ------------- | -------- | ---------- | --------------- | ------------- | ----------- | -------------------------- | ------------ |
| video_001.mp4 | Positive | 0.923      | v2/en_speaker_6 | [laughs]      | 0.7         | .../video_001_positive.wav | Thank you... |
| video_002.mp4 | Negative | 0.875      | v2/en_speaker_3 | [sighs]       | 0.6         | .../video_002_negative.wav | I'm sorry... |

## ⚙️ Parametri Configurabili

### `generate_emotional_audio()`

```python
audio_path = generate_emotional_audio(
    emotion="Positive",               # 'Positive', 'Negative', 'Neutral'
    confidence=0.92,                  # 0.0-1.0
    video_name="video_001",           # Nome identificativo
    output_dir="results/audio",       # Directory output
    caption="Hello world!",           # Testo da pronunciare
    use_emotional_tags=True,          # Usa [laughs], [sighs]
    alternative_speaker=0,            # 0-2 per speaker alternativi
    preload=True,                     # Usa modelli pre-caricati
)
```

### Tag Emotivi Bark

Oltre a `[laughs]` e `[sighs]`, Bark supporta:

- `[gasps]` - Sorpresa/shock
- `[clears throat]` - Esitazione
- `[music]` - Musica di sottofondo
- `[MAN]` / `[WOMAN]` - Forza il genere

## 🔧 Troubleshooting

### Problema: "Bark non installato"

```bash
pip install git+https://github.com/suno-ai/bark.git
```

### Problema: "Out of Memory"

Bark richiede ~10GB RAM. Opzioni:

1. Non pre-caricare i modelli (più lento ma usa meno RAM)

```python
generate_emotional_audio(..., preload=False)
```

2. Generare audio uno alla volta invece che in batch

### Problema: Generazione troppo lenta

1. Pre-carica i modelli una sola volta:

```python
preload_bark_models()  # Una volta all'inizio

# Poi genera molti audio velocemente
for video in videos:
    generate_emotional_audio(..., preload=True)
```

2. Usa GPU se disponibile (automatico con PyTorch)

## 📊 Metriche di Performance

### Bark vs Edge-TTS

| Metrica              | Bark              | Edge-TTS                 |
| -------------------- | ----------------- | ------------------------ |
| Qualità audio        | ⭐⭐⭐⭐⭐        | ⭐⭐⭐                   |
| Espressività emotiva | ⭐⭐⭐⭐⭐        | ⭐⭐⭐                   |
| Velocità generazione | ⭐⭐ (10-20s)     | ⭐⭐⭐⭐⭐ (1-2s)        |
| Uso RAM              | ⭐ (10GB)         | ⭐⭐⭐⭐⭐ (100MB)       |
| Dimensione file      | ⭐⭐ (WAV grande) | ⭐⭐⭐⭐ (MP3 compresso) |
| Open source          | ✅                | ❌                       |

**Raccomandazione:** Usa **Bark** per qualità superiore, **Edge-TTS** per velocità.

## 🎓 Riferimenti

- [Bark GitHub](https://github.com/suno-ai/bark)
- [Bark Paper](https://arxiv.org/abs/2301.03728)
- Documentazione progetto: `src/tts/bark/README.md`
