# 🎭 Bark Emotional Tags - Lista Completa

## Tag Emotivi Supportati da Bark

Bark supporta una vasta gamma di tag emotivi che possono essere inseriti nel testo per controllare l'espressività dell'audio.

### ✅ Tag Verificati (funzionano bene)

#### Risate e Gioia

- `[laughs]` - Risata normale
- `[laughter]` - Risata (variante)
- `[chuckles]` - Risatina
- `[giggles]` - Risatina leggera (più femminile)

#### Tristezza e Frustrazione

- `[sighs]` - Sospiro
- `[gasps]` - Rantolo di sorpresa/shock
- `[sad]` - Voce triste (modificatore tono)

#### Suoni Vocali

- `[clears throat]` - Schiarimento voce
- `[coughs]` - Colpo di tosse
- `[sniffs]` - Sniffata
- `[yawns]` - Sbadiglio

#### Musica e Suoni

- `[music]` - Può generare melodie/musica
- `[singing]` - Canto
- `[humming]` - Canticchiare

#### Pause e Respirazione

- `[breath]` - Respiro
- `...` - Pausa (usando puntini)

### ⚠️ Tag Sperimentali (risultati variabili)

- `[whispers]` - Sussurro
- `[shouts]` - Urlo
- `[excited]` - Eccitazione
- `[angry]` - Rabbia
- `[scared]` - Paura

## 🎯 Raccomandazioni per EmoSign

### Per Emozione POSITIVE

**Tag Primari (più affidabili):**

1. `[laughs]` - Risata genuina ✅ **Default attuale**
2. `[chuckles]` - Risata più contenuta
3. `[giggles]` - Risata leggera, allegra

**Tag Secondari:** 4. `[excited]` - Per alta confidenza (>90%) 5. `[music]` - Per momenti molto positivi 6. `...` (pause) - Per enfatizzare anticipazione positiva

**Combinazioni:**

```python
# Alta confidenza (>90%)
"[laughs] This is amazing! [giggles] Really great!"

# Media confidenza (70-90%)
"[chuckles] This is nice."

# Bassa confidenza (50-70%)
"This is... [laughs] okay."
```

### Per Emozione NEGATIVE

**Tag Primari:**

1. `[sighs]` - Sospiro triste ✅ **Default attuale**
2. `[gasps]` - Shock/sorpresa negativa
3. `[sad]` - Voce triste

**Tag Secondari:** 4. `[clears throat]` - Imbarazzo, disagio 5. `...` (pause) - Esitazione, pensiero triste 6. `[breath]` - Respiro affaticato/preoccupato

**Combinazioni:**

```python
# Alta confidenza negativa (>90%)
"[sighs] This is terrible... [gasps] really bad."

# Media confidenza (70-90%)
"[sighs] This is disappointing."

# Bassa confidenza (50-70%)
"This is... [clears throat] not great."
```

### Per Emozione NEUTRAL

**Tag Primari:**

1. `[clears throat]` - Professionale ✅ **Default attuale**
2. Nessun tag - Mantieni neutralità

**Tag Secondari (solo se necessario):** 3. `...` - Pausa riflessiva 4. `[breath]` - Respiro neutro

## 🔬 Mappatura Avanzata Emozioni → Tag

### Strategia Multi-Tag Basata su Confidenza

```python
CONFIDENCE_BASED_TAGS = {
    "Positive": {
        "high": ["[laughs]", "[giggles]"],      # >90%
        "medium": ["[chuckles]", "[laughs]"],   # 70-90%
        "low": ["[chuckles]"],                  # 50-70%
    },
    "Negative": {
        "high": ["[sighs]", "[gasps]"],         # >90%
        "medium": ["[sighs]"],                  # 70-90%
        "low": ["[clears throat]"],             # 50-70% (più neutro)
    },
    "Neutral": {
        "high": [""],                           # Nessun tag
        "medium": [""],
        "low": ["[clears throat]"],             # Solo se molto lungo
    }
}
```

## 💡 Idee per Implementazione

### Variante 1: Tag Multipli per Varietà

Invece di usare sempre `[laughs]`, rotazione casuale:

```python
import random

POSITIVE_TAGS = ["[laughs]", "[chuckles]", "[giggles]"]
tag = random.choice(POSITIVE_TAGS)
```

### Variante 2: Tag Basati su Lunghezza + Confidenza

```python
def get_optimal_tag(emotion, confidence, text_length):
    if emotion == "Positive":
        if confidence > 0.9:
            return "[laughs]" if text_length > 50 else "[giggles]"
        elif confidence > 0.7:
            return "[chuckles]"
        else:
            return ""  # Bassa confidenza = nessun tag
    # ...
```

### Variante 3: Tag Multipli Posizionati

Per testi lunghi, usa tag diversi in posizioni diverse:

```python
# Inizio: [laughs]
# Metà: [giggles]
# Fine: [chuckles]
"[laughs] This is great! [giggles] Really amazing. [chuckles] Love it!"
```

## 🧪 Test Consigliati

### Test 1: Confronto Tag Singoli

Genera audio con tag diversi per stessa emozione:

```python
tags_to_test = ["[laughs]", "[chuckles]", "[giggles]"]
for tag in tags_to_test:
    generate_audio(f"{tag} Hello world")
```

### Test 2: Combinazioni Multi-Tag

```python
single_tag = "[laughs] This is great!"
multi_tag = "[laughs] This is great! [giggles] Really!"
```

### Test 3: Tag Basati su Confidenza

```python
high_conf = "[laughs] [giggles] Amazing!"  # 95% confidence
low_conf = "This is nice."                  # 55% confidence
```

## 📊 Effectiveness Matrix (Stime)

| Tag               | Positive   | Negative   | Neutral    | Naturalezza | Affidabilità |
| ----------------- | ---------- | ---------- | ---------- | ----------- | ------------ |
| `[laughs]`        | ⭐⭐⭐⭐⭐ | ❌         | ❌         | ⭐⭐⭐⭐    | ⭐⭐⭐⭐⭐   |
| `[chuckles]`      | ⭐⭐⭐⭐   | ❌         | ⭐         | ⭐⭐⭐⭐⭐  | ⭐⭐⭐⭐     |
| `[giggles]`       | ⭐⭐⭐⭐⭐ | ❌         | ❌         | ⭐⭐⭐      | ⭐⭐⭐⭐     |
| `[sighs]`         | ❌         | ⭐⭐⭐⭐⭐ | ⭐         | ⭐⭐⭐⭐⭐  | ⭐⭐⭐⭐⭐   |
| `[gasps]`         | ⭐         | ⭐⭐⭐⭐   | ❌         | ⭐⭐⭐      | ⭐⭐⭐       |
| `[sad]`           | ❌         | ⭐⭐⭐     | ❌         | ⭐⭐        | ⭐⭐         |
| `[clears throat]` | ❌         | ⭐         | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐  | ⭐⭐⭐⭐⭐   |
| `[music]`         | ⭐⭐⭐     | ❌         | ❌         | ⭐⭐        | ⭐⭐         |

## 🚀 Prossimi Passi

1. **Test Audio Comparativo**: Genera audio con tag diversi
2. **Ascolto Qualitativo**: Valuta quale suona meglio
3. **Implementa Varianti**: Aggiungi logica per tag multipli
4. **Metriche**: Raccogli feedback su naturalezza/espressività

## 📚 References

- Bark GitHub Issues: Tag emotivi discussi dalla community
- Bark Examples: Repository con esempi di tag
- TTS Research: Prosody e emotional speech synthesis
