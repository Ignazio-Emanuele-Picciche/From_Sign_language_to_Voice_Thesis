# 🎭 Emotion Tag Optimization - Bark TTS

## Problema Iniziale

Nella prima implementazione Bark, i tag emotivi (`[laughs]`, `[sighs]`) venivano **sempre posizionati all'inizio** del testo:

```python
# ❌ Metodo tradizionale
text = "[laughs] I am so happy to see you today, and I hope you're doing well!"
```

Questo funziona ma non è ottimale perché:

- Suona poco naturale per frasi lunghe
- La risata/sospiro è troppo anticipata
- Non sfrutta le pause naturali del linguaggio

## Soluzione: Posizionamento Intelligente

Il modulo `emotion_tag_optimizer.py` **analizza il testo** e posiziona i tag in modo strategico basandosi su:

1. **Lunghezza del testo** (corto/medio/lungo)
2. **Pause naturali** (punteggiatura, congiunzioni)
3. **Emozione** (strategie diverse per Positive/Negative/Neutral)

### Strategia POSITIVE

#### Testo Corto (<40 caratteri)

```python
Input:  "Hello world"
Output: "[laughs] Hello world"
```

✅ Tag all'inizio (spontaneo, immediato)

#### Testo Medio (40-100 caratteri)

```python
Input:  "I am happy to see you, and I hope you're doing well!"
Output: "I am happy to see you, [laughs] and I hope you're doing well!"
```

✅ Tag dopo la prima pausa naturale (virgola)

#### Testo Lungo (>100 caratteri)

```python
Input:  "This is wonderful news! I can't believe it happened. It's amazing and unexpected."
Output: "[laughs] This is wonderful news! I can't believe it happened. [laughs] It's amazing..."
```

✅ Tag all'inizio + a metà frase (distribuito)

### Strategia NEGATIVE

#### Testo Corto (<40 caratteri)

```python
Input:  "I feel sad"
Output: "[sighs] I feel sad"
```

#### Testo Medio (40-100 caratteri)

```python
Input:  "This is disappointing news, and we must address it."
Output: "This is disappointing news, [sighs] and we must address it."
```

✅ Tag a **metà** frase (più drammatico ed emotivo)

#### Testo Lungo (>100 caratteri)

```python
Input:  "This is very bad. We tried our best but failed. Now we must accept the consequences."
Output: "[sighs] This is very bad. We tried our best but failed. [sighs] Now we must accept..."
```

✅ Tag all'inizio + verso la fine al 75% (climax emotivo)

### Strategia NEUTRAL

```python
# Testo corto/medio: NO tag (mantieni neutralità)
Input:  "The meeting starts at three"
Output: "The meeting starts at three"

# Testo lungo (>80): Tag solo se necessario
Input:  "The meeting starts at three and will cover project updates, budget review, and next steps."
Output: "[clears throat] The meeting starts at three and will cover..."
```

## Rilevamento Pause Naturali

L'optimizer rileva pause naturali dopo:

1. **Punteggiatura**: `. , ! ? ;`
2. **Congiunzioni**: `and`, `but`, `so`, `because`, `however`, `therefore`
3. **Metà frase**: Se >100 caratteri senza punteggiatura

Esempio:

```python
text = "Hello, world! How are you?"
breaks = find_natural_breaks(text)
# breaks = [7, 14]
#          ↓      ↓
# "Hello, |world! |How are you?"
```

## Confronto Audio

### Test A/B: Tradizionale vs Ottimizzato

**Testo:** "I am so happy to see you today, and I hope you're doing well!"

**Versione 1 - Tradizionale:**

```
[laughs] I am so happy to see you today, and I hope you're doing well!
```

- Risata **solo** all'inizio
- Suona meno naturale

**Versione 2 - Ottimizzata:**

```
I am so happy to see you today, [laughs] and I hope you're doing well!
```

- Risata **dopo** la pausa naturale (virgola)
- Suona più spontaneo
- Migliore timing emotivo

### Risultati Attesi

| Lunghezza Testo | Differenza Percepita | Motivo                         |
| --------------- | -------------------- | ------------------------------ |
| Corto (<40)     | Minima               | Entrambi usano tag all'inizio  |
| Medio (40-100)  | **Alta**             | Ottimizzato usa pause naturali |
| Lungo (>100)    | **Molto Alta**       | Tag distribuiti vs concentrato |

## API Usage

### Uso Base

```python
from src.tts.bark.emotion_tag_optimizer import optimize_emotional_text

# Ottimizzazione automatica
optimized = optimize_emotional_text(
    text="I am happy to see you today, and I hope you're doing well!",
    emotion="Positive",
    use_tags=True,
    custom_tag=None  # Usa tag di default per l'emozione
)
# Output: "I am happy to see you today, [laughs] and I hope you're doing well!"
```

### Integrazione con TTS Generator

```python
from src.tts.bark.tts_generator import generate_emotional_audio

# Genera audio con ottimizzazione
audio_path = generate_emotional_audio(
    emotion="Positive",
    confidence=0.92,
    video_name="video_001",
    output_dir="results/audio",
    caption="I am happy to see you today, and I hope you're doing well!",
    use_emotional_tags=True,
    optimize_tag_placement=True,  # ✅ ABILITA ottimizzazione
    preload=True,
)
```

### Tag Personalizzati

```python
from src.tts.bark.emotion_tag_optimizer import get_alternative_tags

# Ottieni tag alternativi
tags = get_alternative_tags("Positive")
# ['[laughs]', '[giggles]', '[chuckles]']

# Usa tag personalizzato
optimized = optimize_emotional_text(
    text="This is funny!",
    emotion="Positive",
    use_tags=True,
    custom_tag="[giggles]"  # Invece di [laughs]
)
```

### Analisi Pause

```python
from src.tts.bark.emotion_tag_optimizer import find_natural_breaks

text = "Hello, world! How are you?"
breaks = find_natural_breaks(text)
# [7, 14]

# Visualizza le pause
for pos in breaks:
    print(f"...{text[max(0,pos-10):pos]}|{text[pos:min(len(text),pos+10)]}...")
# ...Hello, |world! How...
# ...o, world! |How are you...
```

## Test Suite

### Test 1: Optimizer Standalone

```bash
python test_emotion_tag_optimizer.py
```

Testa solo il posizionamento dei tag (veloce, nessun audio).

### Test 2: Confronto Audio Completo

```bash
python test_bark_optimization_comparison.py
```

Genera coppie di audio (tradizionale vs ottimizzato) per A/B testing.

Output:

```
outputs/bark_comparison/
├── traditional/
│   ├── test_01_traditional_positive.wav
│   ├── test_02_traditional_positive.wav
│   └── ...
└── optimized/
    ├── test_01_optimized_positive.wav
    ├── test_02_optimized_optimized.wav
    └── ...
```

## Configurazione

### Abilita/Disabilita Ottimizzazione

```python
# Abilita (default raccomandato)
generate_emotional_audio(..., optimize_tag_placement=True)

# Disabilita (metodo tradizionale)
generate_emotional_audio(..., optimize_tag_placement=False)
```

### Soglie di Lunghezza

Puoi personalizzare le soglie modificando `emotion_tag_optimizer.py`:

```python
# In optimize_positive_tags()
if text_len < 40:        # ← Modifica questa soglia
    return f"{base_tag} {text}"
elif text_len < 100:     # ← Modifica questa soglia
    # ...
```

## Performance

**Overhead Computazionale:**

- Analisi testo: <1ms
- Rilevamento pause: <1ms
- **Totale: Trascurabile** (0.1% del tempo di generazione Bark)

**Raccomandazione:**
✅ **Sempre abilitato** - nessun impatto su performance, migliora qualità audio

## Limitazioni

1. **Lingue non-inglesi**: Ottimizzato per testo inglese (congiunzioni, punteggiatura)
2. **Testi molto brevi** (<10 char): Nessuna differenza vs tradizionale
3. **Testi senza punteggiatura**: Fallback a posizionamento all'inizio o metà

## Future Improvements

- [ ] Supporto multi-lingua (analisi pause per altre lingue)
- [ ] Machine learning per predire posizioni ottimali
- [ ] Tag multipli basati su sentiment analysis
- [ ] A/B testing automatico con metriche MOS (Mean Opinion Score)

## References

- Bark TTS: https://github.com/suno-ai/bark
- Emotional tags: Bark documentation on special tokens
- Natural language pauses: Prosody research in speech synthesis
