# 🎤 Sistema TTS Emotivo + Audio Explainability

Sistema integrato per generare audio emotivo basato su predizioni ViViT e validare quantitativamente i parametri prosodici applicati.

---

## 📦 Componenti

### 1. **TTS Generation** (`src/tts/`)

- `emotion_mapper.py`: Mappa emozioni → parametri prosodici (pitch, rate, volume)
- `text_templates.py`: Template di testo per sintetizzazione
- `tts_generator.py`: Genera audio con edge-tts

### 2. **Audio Explainability** (`src/explainability/audio/`)

- `acoustic_analyzer.py`: Estrae features acustiche (pitch, rate, energy)
- `prosody_validator.py`: Valida parametri target vs measured

---

## 🚀 Quick Start

### Installazione Dipendenze

```bash
pip install edge-tts praat-parselmouth librosa soundfile
```

### Uso con Test ViViT

**Comando base** (solo classificazione):

```bash
python src/models/two_classes/vivit/test_golden_labels_vivit.py \
  --model_uri mlartifacts/697363764579443849/models/m-de73e05128734690a016c37e5610eeb2/artifacts \
  --batch_size 1 \
  --save_results
```

**Comando con TTS + Audio Explainability** (NUOVO):

```bash
python src/models/two_classes/vivit/test_golden_labels_vivit.py \
  --model_uri mlartifacts/697363764579443849/models/m-de73e05128734690a016c37e5610eeb2/artifacts \
  --batch_size 1 \
  --save_results \
  --generate_tts
```

---

## 📊 Output

### File Generati

**1. Audio Files** (`results/tts_audio/generated/`)

```
video_001_positive.mp3
video_002_negative.mp3
...
```

**2. Baseline Audio** (`results/tts_audio/baseline/`)

```
baseline_neutral.mp3  # Audio neutrale per confronti
```

**3. CSV Report** (`results/vivit_tts_audio_analysis_2_classes.csv`)

```csv
video_name,emotion,confidence,audio_path,pitch_accuracy,rate_accuracy,volume_accuracy,overall_accuracy,...
video_001,Positive,0.92,results/.../video_001_positive.mp3,0.85,0.78,0.91,0.85,...
```

### Console Output

```
======================================================================
TTS GENERATION SUMMARY
======================================================================
Total audio files generated: 50

PROSODY VALIDATION:
  Pitch modulation accuracy:  85.2%
  Rate modulation accuracy:   78.6%
  Volume modulation accuracy: 91.3%
  Overall accuracy:           85.0%

EMOTION BREAKDOWN:

Positive (25 samples):
  Avg pitch delta: +7.8% (target: +8.0%)
  Avg rate delta:  +14.2% (target: +15.0%)
  Avg accuracy:    84.5%

Negative (25 samples):
  Avg pitch delta: -5.9% (target: -6.0%)
  Avg rate delta:  -11.3% (target: -12.0%)
  Avg accuracy:    85.5%
```

---

## 🔧 Configurazione Parametri Prosodici

### File: `src/tts/emotion_mapper.py`

```python
PROSODY_MAPPING = {
    'Positive': {
        'rate': '+15%',    # Velocità di eloquio +15%
        'pitch': '+8%',    # Pitch (altezza voce) +8%
        'volume': '+5%'    # Volume +5%
    },
    'Negative': {
        'rate': '-12%',    # Velocità -12%
        'pitch': '-6%',    # Pitch -6%
        'volume': '-3%'    # Volume -3%
    }
}
```

**Personalizzazione**: Modifica questi valori per cambiare l'intensità della modulazione emotiva.

---

## 📈 Come Funziona l'Audio Explainability

### 1. Generazione Audio

```
Video → ViViT → Emotion (Positive) → Prosody Params (+15% rate, +8% pitch, +5% volume)
                     ↓
              edge-tts genera audio.mp3
```

### 2. Analisi Acustica

```
audio.mp3 → AcousticAnalyzer → Features {
    mean_pitch_hz: 210.4 Hz,
    speaking_rate_syll_sec: 5.2,
    mean_energy_db: -18.3 dB
}
```

### 3. Validazione

```
Baseline features: {pitch: 195 Hz, rate: 4.5 syll/sec, ...}
Generated features: {pitch: 210 Hz, rate: 5.2 syll/sec, ...}

Measured delta pitch: +7.7%
Target delta pitch:   +8.0%
Accuracy:             96.3% ✅
```

---

## 🎯 Metriche di Validazione

### Pitch (Frequenza Fondamentale)

- **Target**: Aumento/diminuzione % richiesto
- **Measured**: Aumento/diminuzione % effettivo
- **Accuracy**: `1 - |measured - target| / |target|`

### Rate (Velocità di Eloquio)

- **Misurato in**: sillabe/secondo
- **Stima**: Onset detection con librosa

### Volume (Energia)

- **Misurato in**: dB (decibel)
- **Calcolo**: RMS energy convertito in dB

### Overall Accuracy

- **Formula**: `(pitch_acc + rate_acc + volume_acc) / 3`
- **Threshold**: > 70% considerato "applicato correttamente"

---

## 📚 Uso Standalone dei Moduli

### Test TTS Generator

```python
from src.tts.tts_generator import generate_emotional_audio, generate_baseline_audio

# Genera baseline
generate_baseline_audio('baseline.mp3')

# Genera audio emotivo
generate_emotional_audio(
    emotion='Positive',
    confidence=0.92,
    video_name='test_video',
    output_dir='output/'
)
```

### Test Acoustic Analyzer

```python
from src.explainability.audio.acoustic_analyzer import AcousticAnalyzer

analyzer = AcousticAnalyzer('audio.mp3')
features = analyzer.get_all_features()
print(features)
# {'mean_pitch_hz': 210.4, 'speaking_rate_syll_sec': 5.2, ...}
```

### Test Prosody Validator

```python
from src.explainability.audio.prosody_validator import validate_prosody, print_validation_report

target = {'rate': '+15%', 'pitch': '+8%', 'volume': '+5%'}
report = validate_prosody('positive.mp3', 'baseline.mp3', target)
print_validation_report(report)
```

---

## 🔊 Voci TTS Disponibili

### Raccomandate (edge-tts)

- `en-US-AriaNeural` - Femminile USA (default, molto naturale)
- `en-US-GuyNeural` - Maschile USA
- `en-GB-SoniaNeural` - Femminile UK
- `en-GB-RyanNeural` - Maschile UK

### Cambiare Voce

Modifica in `src/tts/tts_generator.py`:

```python
DEFAULT_VOICE = "en-US-GuyNeural"  # Cambia qui
```

---

## 🐛 Troubleshooting

### Errore: "No module named 'parselmouth'"

```bash
pip install praat-parselmouth
```

### Errore: "No module named 'edge_tts'"

```bash
pip install edge-tts
```

### Audio generato troppo corto/vuoto

- Verifica che il testo non sia troppo breve
- Controlla connessione internet (edge-tts usa server Microsoft)

### Accuracy bassa (<50%)

- **È normale**: edge-tts non applica SSML al 100%
- **Soluzione**: Aumenta i parametri target (es: +20% invece di +15%)
- **Alternativa**: Usa TTS diverso (Coqui TTS, Azure Neural TTS premium)

---

## 📖 References

**Parametri Prosodici**:

- Pitch (F0): Frequenza fondamentale della voce (Hz)
- Rate: Velocità di eloquio (sillabe/secondo)
- Volume: Intensità/energia della voce (dB)

**Librerie Usate**:

- `edge-tts`: Microsoft Edge TTS (gratuito)
- `praat-parselmouth`: Analisi fonetica (pitch, jitter, shimmer)
- `librosa`: Analisi audio (spectral features, onset detection)

**SSML (Speech Synthesis Markup Language)**:

- Standard W3C per controllo sintesi vocale
- Supporta `<prosody>` tag per modulazione

---

## ✅ Next Steps

1. ✅ **Test su subset golden labels** (~10 video)
2. ⏳ **Esegui su tutti i golden labels** (50+ video)
3. ⏳ **Analizza risultati aggregati**
4. ⏳ **Genera visualizzazioni** (plots per tesi)
5. ⏳ **Scrivi sezione tesi** con risultati

---

**Sistema pronto all'uso! Esegui con `--generate_tts` per attivare TTS + Audio Explainability.**
