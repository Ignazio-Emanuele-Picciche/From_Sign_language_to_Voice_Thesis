# TTS Emotivo - Bark vs Edge-TTS

Questo progetto supporta due engine TTS per la generazione di audio emotivo:

## 📁 Struttura

```
src/tts/
├── __init__.py              # Edge-TTS (originale)
├── emotion_mapper.py        # Edge-TTS mapping
├── tts_generator.py         # Edge-TTS generator
├── text_templates.py        # Condiviso tra entrambi
└── bark/                    # 🆕 Bark TTS (Suno AI)
    ├── __init__.py
    ├── emotion_mapper.py        # Bark speaker prompts
    ├── emotion_tag_optimizer.py # 🆕 Posizionamento intelligente tag
    ├── tts_generator.py         # Bark audio generator
    └── pytorch_patch.py         # Fix compatibilità PyTorch 2.9+
```

## 🎯 Bark TTS (Raccomandato)

**Vantaggi:**

- ✅ Audio molto più espressivo e naturale
- ✅ Tag emotivi speciali: `[laughs]`, `[sighs]`, `[gasps]`
- ✅ Speaker con caratteristiche emotive diverse
- ✅ Modulazione naturale di pitch, rate, volume
- ✅ Open source (Suno AI)

**Svantaggi:**

- ❌ Richiede ~10GB RAM per caricare i modelli
- ❌ Generazione più lenta (~10-20s per clip)
- ❌ Output in formato WAV (file più grandi)

### Installazione Bark

```bash
pip install git+https://github.com/suno-ai/bark.git
```

### Uso in Pipeline ViViT

```python
from src.tts.bark.tts_generator import generate_emotional_audio, preload_bark_models
from src.tts.bark.emotion_mapper import map_emotion_to_bark_prompt

# Pre-carica modelli (opzionale ma consigliato)
preload_bark_models()

# Genera audio emotivo
audio_path = generate_emotional_audio(
    emotion="Positive",               # Emozione predetta da ViViT
    confidence=0.92,                  # Confidenza predizione
    video_name="video_001",           # Nome video
    output_dir="results/audio",       # Directory output
    caption="Hello world!",           # Testo del sign language
    use_emotional_tags=True,          # Usa [laughs], [sighs]
    optimize_tag_placement=True,      # 🆕 Posizionamento intelligente
    preload=True,                     # Usa modelli pre-caricati
)
```

### 🆕 Ottimizzazione Posizionamento Tag

Il modulo `emotion_tag_optimizer.py` posiziona i tag emotivi in modo **strategico** invece di metterli sempre all'inizio:

**Strategia POSITIVE (`[laughs]`):**

- Testo corto (<40 char): Tag all'inizio (spontaneo)
- Testo medio (40-100): Tag dopo prima pausa naturale (virgola, punto)
- Testo lungo (>100): Tag all'inizio + a metà frase

**Strategia NEGATIVE (`[sighs]`):**

- Testo corto: Tag all'inizio
- Testo medio: Tag a metà frase (più drammatico)
- Testo lungo: Tag all'inizio + verso la fine (75% del testo)

**Strategia NEUTRAL:**

- Testo corto/medio: Nessun tag (mantieni neutralità)
- Testo lungo (>80): Tag solo all'inizio se necessario

**Esempio:**

```python
from src.tts.bark.emotion_tag_optimizer import optimize_emotional_text

# Input
text = "I am so happy to see you today, and I hope you're doing well!"

# Output con ottimizzazione
# "I am so happy to see you today, [laughs] and I hope you're doing well!"
# Tag posizionato DOPO la virgola invece che all'inizio
```

### 🆕 Tag Emotivi Multipli

Il sistema supporta **12+ tag emotivi** invece dei soli [laughs] e [sighs]:

**Tag POSITIVE (4 varianti):**

- `[laughs]` - Risata genuina ✅ Default alta confidenza
- `[chuckles]` - Risata contenuta, professionale ✅ Default media confidenza
- `[giggles]` - Risata leggera, giocosa
- `[laughter]` - Risata (variante)

**Tag NEGATIVE (4 varianti):**

- `[sighs]` - Sospiro di tristezza ✅ Default
- `[gasps]` - Shock/sorpresa negativa
- `[sad]` - Voce triste
- `[clears throat]` - Disagio/esitazione ✅ Bassa confidenza

**Tag NEUTRAL (2 varianti):**

- _(nessuno)_ - Mantieni neutralità ✅ Default
- `[clears throat]` - Solo se necessario

**Uso Automatico (Raccomandato):**

```python
# Sistema sceglie tag automaticamente basato su confidenza
audio = generate_emotional_audio(
    emotion="Positive",
    confidence=0.92,  # Alta → usa [laughs]
    video_name="video_001",
    output_dir="results/audio",
    caption="I am so happy!",
    confidence_based_tags=True  # ✅ Auto-selezione
)
```

**Uso Manuale (Tag Specifico):**

```python
# Forza uso di [chuckles] invece di [laughs]
audio = generate_emotional_audio(
    emotion="Positive",
    confidence=0.92,
    video_name="video_001",
    output_dir="results/audio",
    caption="I am happy.",
    alternative_tag=1,  # 0=[laughs], 1=[chuckles], 2=[giggles]
    confidence_based_tags=False
)
```

**Strategia Confidence-Based:**

| Confidenza | Positive     | Negative          | Neutral           |
| ---------- | ------------ | ----------------- | ----------------- |
| >90%       | `[laughs]`   | `[sighs]`         | _(nessuno)_       |
| 70-90%     | `[chuckles]` | `[sighs]`         | _(nessuno)_       |
| <70%       | _(nessuno)_  | `[clears throat]` | `[clears throat]` |

````

### Mapping Emozioni → Bark Speakers

| Emozione | Speaker           | Caratteristiche       | Tag Emotivo |
| -------- | ----------------- | --------------------- | ----------- |
| Positive | `v2/en_speaker_6` | Energico, allegro     | `[laughs]`  |
| Negative | `v2/en_speaker_3` | Calmo, riflessivo     | `[sighs]`   |
| Neutral  | `v2/en_speaker_9` | Neutro, professionale | _(none)_    |

## 🔄 Edge-TTS (Fallback)

**Vantaggi:**

- ✅ Veloce (~1-2s per clip)
- ✅ Leggero (pochi MB)
- ✅ Output in formato MP3

**Svantaggi:**

- ❌ Audio meno espressivo
- ❌ Richiede SSML per emozioni
- ❌ Dipende da servizi Microsoft

### Installazione Edge-TTS

```bash
pip install edge-tts
````

### Uso

```python
from src.tts.tts_generator import generate_emotional_audio

audio_path = generate_emotional_audio(
    emotion="Positive",
    confidence=0.92,
    video_name="video_001",
    output_dir="results/audio",
    caption="Hello world!",
)
```

## 🚀 Test Rapido

### Test Bark

```bash
python test_tts_bark.py
```

### Test Edge-TTS

```bash
python test_tts_ssml.py
```

## 📊 Pipeline Completa: ViViT → TTS

```bash
# Con Bark (raccomandato per qualità)
python src/models/two_classes/vivit/test_golden_labels_vivit.py \
  --model_uri mlartifacts/.../artifacts \
  --batch_size 1 \
  --save_results \
  --generate_tts

# Output:
# - Video analizzato → Emozione predetta (Positive/Negative)
# - Audio emotivo generato con Bark
# - CSV con risultati: results/vivit_tts_bark_2_classes.csv
```

## 🎨 Personalizzazione

### Usare Speaker Alternativi (Bark)

```python
# Speaker alternativo per Positive (0=default, 1-2=alternative)
audio_path = generate_emotional_audio(
    emotion="Positive",
    video_name="test",
    output_dir="output",
    alternative_speaker=1,  # Usa speaker alternativo
)
```

### Disabilitare Tag Emotivi (Bark)

```python
audio_path = generate_emotional_audio(
    emotion="Positive",
    video_name="test",
    output_dir="output",
    use_emotional_tags=False,  # No [laughs], [sighs]
)
```

## 📝 Note

- **Bark è raccomandato** per la qualità audio superiore
- **Edge-TTS** rimane disponibile come fallback veloce
- I modelli Bark vengono scaricati automaticamente la prima volta (~5GB)
- Usa `preload_bark_models()` per velocizzare la generazione batch

## 🔗 Riferimenti

- [Bark GitHub](https://github.com/suno-ai/bark)
- [Edge-TTS GitHub](https://github.com/rany2/edge-tts)
