# 🎵 Bark TTS Generator - Pipeline Emotiva per Sign Language

## 📋 Panoramica

Sistema completo per la generazione di audio emotivo da video di linguaggio dei segni utilizzando **Bark TTS**, un modello transformer-based di Suno AI che supporta generazione vocale altamente espressiva e naturale.

### Pipeline Completa

```
Video Sign Language → ViViT Model → Predizione Emozione → Bark TTS → Audio Emotivo WAV
```

### Caratteristiche Principali

- ✅ **Audio Altamente Espressivo**: Bark genera voci naturali con prosodia emotiva
- ✅ **Tag Emotivi**: Supporto per tag speciali come `[laughs]`, `[sighs]`, `[gasps]`
- ✅ **Speaker Multipli**: 10 voci diverse per lingua con caratteristiche uniche
- ✅ **Controllo Fine**: Temperature personalizzabili per ogni emozione
- ✅ **Posizionamento Intelligente**: Ottimizzazione automatica del posizionamento dei tag emotivi
- ✅ **Open Source**: Modello completamente open source e modificabile

---

## 🏗️ Architettura del Sistema

### Struttura Moduli

```
src/tts/bark/
├── __init__.py                    # Package initialization
├── tts_generator.py              # 🎯 Core: Generazione audio Bark
├── emotion_mapper.py             # 🗺️ Mapping emozioni → speaker + tag
├── emotion_tag_optimizer.py      # 🧠 Ottimizzazione posizionamento tag
├── pytorch_patch.py              # 🔧 Fix compatibilità PyTorch 2.9+
└── README.md                     # 📖 Questa documentazione
```

### Componenti Principali

#### 1. **tts_generator.py** - Generazione Audio

```python
from src.tts.bark.tts_generator import generate_emotional_audio

audio_path = generate_emotional_audio(
    emotion="Positive",
    confidence=0.92,
    video_name="video_001",
    output_dir="results/audio",
    caption="Thank you so much!",
    use_emotional_tags=True,
    optimize_tag_placement=True
)
```

**Funzioni principali:**

- `generate_emotional_audio()` - Genera audio emotivo completo
- `generate_baseline_audio()` - Genera audio neutrale per baseline
- `preload_bark_models()` - Pre-carica modelli in RAM per velocità

#### 2. **emotion_mapper.py** - Mapping Emozioni

```python
from src.tts.bark.emotion_mapper import (
    map_emotion_to_bark_prompt,
    get_bark_speaker,
    get_emotional_tag
)

# Ottieni configurazione per un'emozione
config = map_emotion_to_bark_prompt("Positive")
# {'history_prompt': 'v2/en_speaker_6', 'text_prefix': '[laughs]', ...}

# Ottieni speaker (con alternative)
speaker = get_bark_speaker("Positive", alternative=0)

# Ottieni tag emotivo basato su confidenza
tag = get_emotional_tag("Positive", confidence=0.92)
```

**Mappature disponibili:**

- 3 emozioni supportate: `Positive`, `Negative`, `Neutral`
- 3 speaker alternativi per emozione (varietà vocale)
- Tag emotivi multipli con selezione intelligente
- Temperature ottimizzate per emozione

#### 3. **emotion_tag_optimizer.py** - Ottimizzazione Tag

```python
from src.tts.bark.emotion_tag_optimizer import optimize_emotional_text

# Posiziona tag in modo intelligente
optimized_text = optimize_emotional_text(
    text="This is a longer sentence with natural breaks and pauses.",
    emotion="Positive",
    use_tags=True
)
# Output: "[laughs] This is a longer sentence with [laughs] natural breaks..."
```

**Strategie di ottimizzazione:**

- **Positive**: Tag all'inizio + dopo pause naturali (spontaneità)
- **Negative**: Tag a metà frase per drammaticità
- **Neutral**: Tag solo se necessario (testo lungo)
- Rilevamento automatico di pause naturali (punteggiatura, congiunzioni)

---

## 🎯 Mapping Emozioni → Configurazione Bark

### Tabella Configurazione

| Emozione     | Speaker           | Voce Descrizione          | Tag Default | Temp | Alternative Tags                        |
| ------------ | ----------------- | ------------------------- | ----------- | ---- | --------------------------------------- |
| **Positive** | `v2/en_speaker_6` | Femminile energica        | `[laughs]`  | 0.7  | `[chuckles]`, `[giggles]`, `[laughter]` |
| **Negative** | `v2/en_speaker_3` | Maschile calma/riflessiva | `[sighs]`   | 0.6  | `[gasps]`, `[sad]`, `[clears throat]`   |
| **Neutral**  | `v2/en_speaker_9` | Narratore professionale   | _(nessuno)_ | 0.5  | `[clears throat]`, `[breath]`           |

### Speaker Alternativi

Ogni emozione ha 3 varianti di speaker per maggiore varietà:

```python
ALTERNATIVE_SPEAKERS = {
    "Positive": [
        "v2/en_speaker_6",  # Default: Energico
        "v2/en_speaker_5",  # Alternativa 1: Allegro
        "v2/en_speaker_7",  # Alternativa 2: Vivace
    ],
    "Negative": [
        "v2/en_speaker_3",  # Default: Calmo
        "v2/en_speaker_1",  # Alternativa 1: Riflessivo
        "v2/en_speaker_4",  # Alternativa 2: Serio
    ],
    "Neutral": [
        "v2/en_speaker_9",  # Default: Neutro professionale
        "v2/en_speaker_0",  # Alternativa 1: Narratore
        "v2/en_speaker_2",  # Alternativa 2: Standard
    ]
}
```

### Tag Emotivi Basati su Confidenza

Il sistema seleziona automaticamente il tag emotivo ottimale basandosi sulla confidenza della predizione:

```python
# Positive
confidence > 90% → "[laughs]"        # Risata piena
confidence 70-90% → "[chuckles]"     # Risata contenuta
confidence < 70% → ""                # Nessun tag

# Negative
confidence > 90% → "[sighs]"         # Sospiro marcato
confidence 70-90% → "[sighs]"        # Sospiro normale
confidence < 70% → "[clears throat]" # Più neutro

# Neutral
confidence > 70% → ""                # Sempre neutro
confidence < 70% → "[clears throat]" # Solo per testi lunghi
```

---

## 🚀 Utilizzo

### Installazione

```bash
# Installa Bark TTS
pip install git+https://github.com/suno-ai/bark.git

# Installa dipendenze (se necessario)
pip install numpy scipy torch
```

### Test Rapido

```bash
# Test del modulo emotion_mapper
python src/tts/bark/emotion_mapper.py

# Test del modulo emotion_tag_optimizer
python src/tts/bark/emotion_tag_optimizer.py

# Test generazione audio
python src/tts/bark/tts_generator.py

# Test completo con script dedicato
python test_tts_bark.py
```

### Uso Base

```python
from src.tts.bark.tts_generator import generate_emotional_audio

# Genera audio emotivo
audio_path = generate_emotional_audio(
    emotion="Positive",                # Emozione predetta da ViViT
    confidence=0.92,                   # Confidenza predizione (0.0-1.0)
    video_name="video_001",            # Nome identificativo video
    output_dir="results/tts_audio",    # Directory output
    caption="Hello world!",            # Testo da pronunciare
    use_emotional_tags=True,           # Usa tag emotivi
    preload=False                      # Pre-carica modelli (prima volta)
)

print(f"✅ Audio generato: {audio_path}")
# Output: results/tts_audio/video_001_positive.wav
```

### Uso Avanzato con Ottimizzazioni

```python
from src.tts.bark.tts_generator import generate_emotional_audio, preload_bark_models

# 1. Pre-carica modelli una volta (velocizza generazioni successive)
print("Caricamento modelli Bark...")
preload_bark_models()  # Carica ~10GB in RAM

# 2. Genera molti audio velocemente
for video in video_list:
    audio_path = generate_emotional_audio(
        emotion=video.emotion,
        confidence=video.confidence,
        video_name=video.name,
        output_dir="results/audio",
        caption=video.caption,
        use_emotional_tags=True,
        alternative_speaker=0,              # 0-2 per varietà vocale
        alternative_tag=0,                  # 0-3 per tag alternativi
        confidence_based_tags=True,         # Tag basato su confidenza
        optimize_tag_placement=True,        # Posizionamento intelligente
        preload=True                        # Usa modelli già in RAM
    )
    print(f"✅ {video.name}: {audio_path}")
```

### Pipeline Completa con ViViT

```bash
# Esegui inferenza ViViT + generazione TTS Bark
python src/models/two_classes/vivit/test_golden_labels_vivit.py \
  --model_uri mlartifacts/.../models/... \
  --batch_size 1 \
  --save_results \
  --generate_tts \
  --use_bark
```

**Output generato:**

- Video analizzati: 200 campioni
- Audio files: `results/tts_audio/bark_generated/*.wav`
- CSV risultati: `results/vivit_tts_bark_2_classes.csv`
- Log completi: predizioni + configurazione Bark

---

## 📊 Esempi Output

### Caso 1: Emozione Positiva (Alta Confidenza)

```
[1/200] Processing: video_001.mp4
  Emotion: Positive (confidence: 92.3%)
  Caption: "Thank you so much for your help!"
  Bark Speaker: v2/en_speaker_6 (energetic female)
  Emotional tag: [laughs] (high confidence)
  Temperature: 0.7
  Optimized text: "[laughs] Thank you so much for your help!"
  ✅ Audio generato: video_001_positive.wav (3.2s, 24kHz WAV)
```

**Caratteristiche audio:**

- Voce femminile energica e allegra
- Risata naturale all'inizio
- Tono positivo e caloroso
- Durata: ~3-5 secondi

### Caso 2: Emozione Negativa (Media Confidenza)

```
[2/200] Processing: video_002.mp4
  Emotion: Negative (confidence: 78.1%)
  Caption: "I'm sorry, I can't help you with that."
  Bark Speaker: v2/en_speaker_3 (calm male)
  Emotional tag: [sighs] (medium confidence)
  Temperature: 0.6
  Optimized text: "[sighs] I'm sorry, I can't help you with that."
  ✅ Audio generato: video_002_negative.wav (4.5s, 24kHz WAV)
```

**Caratteristiche audio:**

- Voce maschile calma e riflessiva
- Sospiro empatico all'inizio
- Tono dispiaciuto ma controllato
- Durata: ~4-6 secondi

### Caso 3: Emozione Neutral (Bassa Confidenza)

```
[3/200] Processing: video_003.mp4
  Emotion: Neutral (confidence: 68.5%)
  Caption: "The meeting is scheduled for tomorrow."
  Bark Speaker: v2/en_speaker_9 (neutral narrator)
  Emotional tag: (none) (neutral, no tags)
  Temperature: 0.5
  Optimized text: "The meeting is scheduled for tomorrow."
  ✅ Audio generato: video_003_neutral.wav (3.0s, 24kHz WAV)
```

**Caratteristiche audio:**

- Voce neutra e professionale
- Nessun tag emotivo (mantiene neutralità)
- Tono chiaro e informativo
- Durata: ~3-4 secondi

---

## 📈 Output CSV

File generato: `results/vivit_tts_bark_2_classes.csv`

```csv
video_name,emotion,confidence,bark_speaker,emotional_tag,temperature,optimize_tag,audio_path,caption,duration
video_001.mp4,Positive,0.923,v2/en_speaker_6,[laughs],0.7,True,results/.../video_001_positive.wav,Thank you...,3.2
video_002.mp4,Negative,0.781,v2/en_speaker_3,[sighs],0.6,True,results/.../video_002_negative.wav,I'm sorry...,4.5
video_003.mp4,Neutral,0.685,v2/en_speaker_9,,0.5,False,results/.../video_003_neutral.wav,The meeting...,3.0
```

**Colonne:**

- `video_name`: Nome del video processato
- `emotion`: Emozione predetta da ViViT
- `confidence`: Confidenza predizione (0.0-1.0)
- `bark_speaker`: Speaker ID utilizzato
- `emotional_tag`: Tag emotivo inserito nel testo
- `temperature`: Valore temperature usato per generazione
- `optimize_tag`: Se ottimizzazione tag abilitata
- `audio_path`: Path file audio generato
- `caption`: Testo pronunciato
- `duration`: Durata audio in secondi

---

## ⚙️ Parametri Configurabili

### `generate_emotional_audio()`

```python
def generate_emotional_audio(
    emotion: str,                      # 'Positive', 'Negative', 'Neutral'
    confidence: float,                 # 0.0-1.0 (o 0-100, normalizza auto)
    video_name: str,                   # Nome identificativo video
    output_dir: str,                   # Directory output (creata se non esiste)
    caption: str = None,               # Testo da pronunciare (opzionale)
    use_emotional_tags: bool = True,   # Usa tag emotivi ([laughs], [sighs])
    alternative_speaker: int = 0,      # Speaker alternativo (0-2)
    alternative_tag: int = 0,          # Tag alternativo (0-3)
    confidence_based_tags: bool = True,# Seleziona tag basato su confidenza
    preload: bool = False,             # Usa modelli pre-caricati
    optimize_tag_placement: bool = True # Posizionamento intelligente tag
) -> str:
    """
    Returns: Path completo file audio generato (.wav)
    """
```

**Esempi combinazioni parametri:**

```python
# Default: Tag basato su confidenza, ottimizzazione attiva
generate_emotional_audio("Positive", 0.92, "vid001", "output/")

# Speaker alternativo per varietà
generate_emotional_audio("Positive", 0.92, "vid001", "output/", alternative_speaker=1)

# Tag alternativo specifico (chuckles invece di laughs)
generate_emotional_audio("Positive", 0.92, "vid001", "output/", alternative_tag=1)

# Disabilita ottimizzazione (tag solo all'inizio)
generate_emotional_audio("Positive", 0.92, "vid001", "output/", optimize_tag_placement=False)

# Disabilita completamente tag emotivi
generate_emotional_audio("Positive", 0.92, "vid001", "output/", use_emotional_tags=False)
```

---

## 🔧 Troubleshooting

### ❌ "Bark non installato"

**Errore:**

```
RuntimeError: Bark non è installato! Installa con: pip install git+https://github.com/suno-ai/bark.git
```

**Soluzione:**

```bash
pip install git+https://github.com/suno-ai/bark.git
```

### ❌ "Out of Memory" / "CUDA out of memory"

**Problema:** Bark richiede ~10GB RAM (o VRAM per GPU).

**Soluzioni:**

1. **Non pre-caricare modelli** (usa meno memoria ma generazione più lenta):

```python
generate_emotional_audio(..., preload=False)
```

2. **Genera audio uno alla volta** invece che in batch:

```python
for video in videos:
    audio = generate_emotional_audio(...)  # Uno alla volta
    # Processa audio...
```

3. **Usa CPU invece di GPU** (se CUDA out of memory):

```python
import torch
torch.cuda.is_available = lambda: False  # Forza CPU
```

### ❌ "FutureWarning: weights_only=False"

**Problema:** PyTorch 2.9+ ha cambiato il default di `torch.load()`.

**Soluzione:** Automatica! Il file `pytorch_patch.py` si applica automaticamente all'import:

```python
from src.tts.bark.tts_generator import generate_emotional_audio  # Patch auto-applicato
```

### ⏱️ Generazione Troppo Lenta

**Problema:** Bark genera audio in 10-20 secondi per file.

**Soluzioni:**

1. **Pre-carica modelli una volta** all'inizio:

```python
from src.tts.bark.tts_generator import preload_bark_models

preload_bark_models()  # Una volta all'inizio del programma

# Poi genera molti audio velocemente
for video in videos:
    generate_emotional_audio(..., preload=True)
```

2. **Usa GPU se disponibile** (automatico con PyTorch CUDA):

```python
import torch
print(f"GPU disponibile: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

3. **Considera Edge-TTS per velocità** (se qualità non critica):

```python
# Bark: alta qualità, 10-20s per audio
# Edge-TTS: qualità media, 1-2s per audio
```

### 🔊 Audio Distorto o di Bassa Qualità

**Cause possibili:**

1. Temperature troppo alta → Ridurre a 0.5-0.7
2. Testo troppo lungo → Dividere in frasi più corte
3. Tag emotivi eccessivi → Usare 1-2 tag max per frase

**Soluzioni:**

```python
# Temperature più conservativa
EMOTION_BARK_MAPPING["Positive"]["temperature"] = 0.6  # invece di 0.7

# Limita lunghezza testo
if len(text) > 200:
    text = text[:200]  # Tronca

# Disabilita tag multipli
optimize_tag_placement=False  # Solo tag all'inizio
```

---

## 📊 Metriche di Performance

### Bark vs Edge-TTS

| Metrica                  | Bark TTS                | Edge-TTS                   |
| ------------------------ | ----------------------- | -------------------------- |
| **Qualità audio**        | ⭐⭐⭐⭐⭐ (eccellente) | ⭐⭐⭐ (buona)             |
| **Espressività emotiva** | ⭐⭐⭐⭐⭐ (naturale)   | ⭐⭐⭐ (sintetica)         |
| **Velocità generazione** | ⭐⭐ (10-20s/audio)     | ⭐⭐⭐⭐⭐ (1-2s/audio)    |
| **Uso RAM**              | ⭐ (10GB)               | ⭐⭐⭐⭐⭐ (100MB)         |
| **Uso GPU**              | ⭐⭐ (opzionale)        | ⭐⭐⭐⭐⭐ (non richiesta) |
| **Dimensione file**      | ⭐⭐ (WAV grande)       | ⭐⭐⭐⭐ (MP3 piccolo)     |
| **Open source**          | ✅ Sì                   | ❌ No (Microsoft)          |
| **Controllo prosodia**   | ⭐⭐⭐⭐⭐ (massimo)    | ⭐⭐⭐ (limitato)          |
| **Lingue supportate**    | ⭐⭐⭐ (principali)     | ⭐⭐⭐⭐⭐ (40+)           |

**Raccomandazione:**

- Usa **Bark** per: ricerca, qualità audio superiore, massima espressività
- Usa **Edge-TTS** per: produzione, velocità, risorse limitate

### Benchmark su How2Sign Dataset

Test su 200 video del dataset How2Sign:

```
Sistema Operativo: macOS (Apple Silicon M1/M2)
RAM: 16GB
GPU: Apple M1/M2 Neural Engine

Statistiche Generazione:
- Tempo medio per audio: 12.5s
- Tempo totale (200 audio): ~42 minuti
- RAM utilizzata: 8-10GB durante generazione
- Dimensione media file: 150-200KB (WAV 24kHz)

Con Pre-loading:
- Primo audio: 25s (caricamento modelli)
- Audio successivi: 8-10s (modelli già in RAM)
- Risparmio tempo: ~40%
```

---

## 🎓 Tag Emotivi Supportati

### Tag Verificati (Alta Affidabilità)

#### Risate e Gioia

- `[laughs]` - Risata genuina ✅ **Default Positive**
- `[chuckles]` - Risata contenuta
- `[giggles]` - Risata leggera
- `[laughter]` - Risata (variante)

#### Tristezza e Frustrazione

- `[sighs]` - Sospiro ✅ **Default Negative**
- `[gasps]` - Rantolo/shock
- `[sad]` - Voce triste

#### Suoni Vocali

- `[clears throat]` - Schiarimento voce ✅ **Default Neutral**
- `[coughs]` - Colpo di tosse
- `[sniffs]` - Sniffata
- `[breath]` - Respiro

#### Musica e Altri

- `[music]` - Melodia/musica
- `[singing]` - Canto
- `...` - Pausa (usando puntini)

### Tag Sperimentali (Affidabilità Variabile)

- `[whispers]` - Sussurro
- `[shouts]` - Urlo
- `[excited]` - Eccitazione
- `[angry]` - Rabbia

---

## 📁 File Output

### Formato Audio

- **Formato**: WAV (non compresso)
- **Sample rate**: 24,000 Hz (24kHz)
- **Bit depth**: 16-bit
- **Canali**: Mono
- **Dimensione media**: 150-200KB per audio (3-5s)

### Naming Convention

```
{video_name}_{emotion}.wav

Esempi:
- video_001_positive.wav
- video_002_negative.wav
- video_003_neutral.wav
```

### Struttura Directory Output

```
results/tts_audio/
├── bark_generated/
│   ├── video_001_positive.wav
│   ├── video_002_negative.wav
│   ├── video_003_neutral.wav
│   └── ...
├── baseline/
│   └── baseline_neutral.wav
└── vivit_tts_bark_2_classes.csv
```

---

## 🧪 Test e Validazione

### Script di Test Disponibili

```bash
# Test componenti individuali
python src/tts/bark/emotion_mapper.py
python src/tts/bark/emotion_tag_optimizer.py
python src/tts/bark/tts_generator.py

# Test completo Bark
python test_tts_bark.py

# Test con comparazione Edge-TTS vs Bark
python test_bark_optimization_comparison.py

# Test tag multipli
python test_bark_multiple_tags.py

# Test emotion tag optimizer
python test_emotion_tag_optimizer.py
```

### Validazione Output

```python
# Verifica file audio generato
import wave

with wave.open("output/video_001_positive.wav", 'rb') as wf:
    print(f"Channels: {wf.getnchannels()}")
    print(f"Sample width: {wf.getsampwidth()}")
    print(f"Framerate: {wf.getframerate()}")
    print(f"Frames: {wf.getnframes()}")
    print(f"Duration: {wf.getnframes() / wf.getframerate():.2f}s")

# Output atteso:
# Channels: 1 (mono)
# Sample width: 2 (16-bit)
# Framerate: 24000 (24kHz)
# Duration: 3-5s
```

---

## 🔬 Componenti Tecnici Dettagliati

### PyTorch Patch

Il file `pytorch_patch.py` risolve problemi di compatibilità con PyTorch 2.9+:

```python
# Automaticamente applicato all'import
from src.tts.bark import tts_generator  # Patch attivo

# Fix: torch.load() con weights_only=False per Bark
# Necessario perché Bark usa checkpoint pre-PyTorch 2.6
```

### Gestione Memoria

```python
# Bark usa ~10GB RAM quando modelli pre-caricati
MODELS_PRELOADED = False

def preload_bark_models():
    global MODELS_PRELOADED
    if not MODELS_PRELOADED:
        from bark import preload_models
        preload_models()  # Carica in RAM
        MODELS_PRELOADED = True
```

### Generazione Audio Core

```python
from bark import generate_audio, SAMPLE_RATE

audio_array = generate_audio(
    text,                          # Testo + tag emotivi
    history_prompt="v2/en_speaker_6",  # Speaker ID
    text_temp=0.7,                 # Temperature per testo
    waveform_temp=0.7              # Temperature per audio
)

# audio_array: numpy array di float32
# SAMPLE_RATE: 24000 Hz
```

---

## 📚 Riferimenti e Risorse

### Documentazione Ufficiale

- [Bark GitHub Repository](https://github.com/suno-ai/bark)
- [Bark Paper - Text-to-Audio Generation](https://arxiv.org/abs/2301.03728)
- [Suno AI Blog](https://suno.ai)

### Documentazione Progetto

- `docs/BARK_TTS_PIPELINE.md` - Pipeline completa
- `docs/BARK_EMOTIONAL_TAGS.md` - Lista completa tag emotivi
- `docs/EMOTION_TAG_OPTIMIZATION.md` - Strategie ottimizzazione
- `docs/tts_complete_workflow.md` - Workflow TTS completo

### Paper e Ricerca

- Bark TTS: Transformer-based audio generation
- Emotional Speech Synthesis: Prosody modulation
- Sign Language to Speech: Multimodal translation

---

## 🚀 Sviluppi Futuri

### Features Pianificate

- [ ] Multi-speaker mixing (più voci in un audio)
- [ ] Tag emotivi custom (definiti dall'utente)
- [ ] Fine-tuning su dataset specifico
- [ ] Supporto lingue diverse dall'inglese
- [ ] Batch generation parallela (GPU multi-thread)
- [ ] Compressione output (WAV → MP3/OGG)

### Miglioramenti in Corso

- [ ] Riduzione uso memoria (<5GB)
- [ ] Velocità generazione (<5s/audio)
- [ ] Qualità tag emotivi (A/B testing)
- [ ] Integrazione con valutazione automatica (MOS score)

---

## 📞 Supporto e Contributi

### Issues Comuni

Consulta la sezione **Troubleshooting** per problemi comuni e soluzioni.

### Bug Report

Per segnalare bug, apri una issue con:

- Descrizione errore
- Stack trace completo
- Versioni librerie (`pip list | grep -E "bark|torch"`)
- Sistema operativo e RAM

### Contributi

Contributi benvenuti! Aree di interesse:

- Ottimizzazione velocità
- Nuovi tag emotivi
- Supporto lingue
- Testing e validazione

---

## 📝 Changelog

### v1.0.0 (Attuale)

- ✅ Implementazione completa pipeline Bark TTS
- ✅ Mapping emozioni → speaker + tag
- ✅ Ottimizzazione posizionamento tag emotivi
- ✅ Supporto tag multipli e alternativi
- ✅ Selezione tag basata su confidenza
- ✅ Pre-loading modelli per velocità
- ✅ Fix compatibilità PyTorch 2.9+
- ✅ Integrazione con ViViT pipeline
- ✅ CSV output con metadati completi

---

## ✅ Conclusioni

Bark TTS rappresenta lo **stato dell'arte** per la generazione di audio emotivo nel progetto EmoSign. Grazie alla sua capacità di produrre voci altamente naturali ed espressive, è la scelta ideale per la ricerca e per applicazioni dove la qualità audio è prioritaria.

**Punti di forza:**

- Audio realistico con prosodia emotiva naturale
- Tag emotivi per controllo fine dell'espressività
- Ottimizzazione intelligente del posizionamento tag
- Completamente open source e personalizzabile

**Trade-off da considerare:**

- Richiede risorse hardware significative (10GB RAM)
- Generazione più lenta rispetto a TTS cloud (10-20s vs 1-2s)
- File audio WAV non compressi (dimensioni maggiori)

Per la maggior parte dei casi d'uso in ricerca e sviluppo, **Bark è la soluzione raccomandata**.

---

**Autore**: EmoSign Thesis Project  
**Versione**: 1.0.0  
**Ultimo aggiornamento**: Novembre 2025  
**Licenza**: MIT (Bark) + Academic Use (EmoSign)
