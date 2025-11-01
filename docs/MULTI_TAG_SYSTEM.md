# 🎭 Tag Emotivi Multipli - Sistema Completo

## Domanda Iniziale

> "ma possiamo usare solo laugh e sighs?"

**Risposta:** NO! Bark supporta **molti altri tag emotivi** e abbiamo implementato un sistema completo per usarli.

## ✅ Tag Emotivi Implementati

### POSITIVE (4 varianti)

1. **`[laughs]`** - Risata genuina, forte ✅ **Default alta confidenza**
2. **`[chuckles]`** - Risata contenuta, professionale ✅ **Default media confidenza**
3. **`[giggles]`** - Risata leggera, giocosa
4. **`[laughter]`** - Risata (variante alternativa)

### NEGATIVE (4 varianti)

1. **`[sighs]`** - Sospiro di tristezza ✅ **Default**
2. **`[gasps]`** - Shock/sorpresa negativa
3. **`[sad]`** - Voce triste
4. **`[clears throat]`** - Disagio/esitazione ✅ **Bassa confidenza**

### NEUTRAL (2 varianti)

1. **(nessun tag)** - Mantieni neutralità ✅ **Default**
2. **`[clears throat]`** - Solo se necessario (testi lunghi)

## 🚀 Nuove Funzionalità

### 1. Tag Basati su Confidenza

Il sistema sceglie **automaticamente** il tag migliore basandosi sulla confidenza della predizione:

```python
# Alta confidenza (>90%) → Tag forte
generate_emotional_audio(emotion="Positive", confidence=0.95)
# Usa: [laughs]

# Media confidenza (70-90%) → Tag moderato
generate_emotional_audio(emotion="Positive", confidence=0.82)
# Usa: [chuckles]

# Bassa confidenza (<70%) → Nessun tag
generate_emotional_audio(emotion="Positive", confidence=0.62)
# Usa: (nessun tag)
```

**Strategia:**

| Confidenza | Positive     | Negative          | Neutral           |
| ---------- | ------------ | ----------------- | ----------------- |
| >90%       | `[laughs]`   | `[sighs]`         | _(nessuno)_       |
| 70-90%     | `[chuckles]` | `[sighs]`         | _(nessuno)_       |
| <70%       | _(nessuno)_  | `[clears throat]` | `[clears throat]` |

### 2. Tag Alternativi Manuali

Puoi scegliere **manualmente** quale tag usare:

```python
# Usa [chuckles] invece di [laughs]
generate_emotional_audio(
    emotion="Positive",
    confidence=0.90,
    alternative_tag=1,  # 0=[laughs], 1=[chuckles], 2=[giggles], 3=[laughter]
    confidence_based_tags=False  # Disabilita auto-selezione
)

# Usa [gasps] invece di [sighs]
generate_emotional_audio(
    emotion="Negative",
    alternative_tag=1,  # 0=[sighs], 1=[gasps], 2=[sad], 3=[clears throat]
    confidence_based_tags=False
)
```

### 3. Interrogazione Tag Disponibili

```python
from src.tts.bark.emotion_mapper import get_alternative_emotional_tags

# Ottieni tutti i tag per un'emozione
positive_tags = get_alternative_emotional_tags("Positive")
# ['[laughs]', '[chuckles]', '[giggles]', '[laughter]']

negative_tags = get_alternative_emotional_tags("Negative")
# ['[sighs]', '[gasps]', '[sad]', '[clears throat]']
```

## 📊 Matrice Configurazione

### Configurazione Completa per Emozione

```python
EMOTIONAL_TAGS = {
    "Positive": {
        "primary": "[laughs]",                    # Tag di default
        "alternatives": [                         # Tag alternativi
            "[chuckles]",
            "[giggles]",
            "[laughter]"
        ],
        "high_confidence": "[laughs]",            # >90%
        "medium_confidence": "[chuckles]",        # 70-90%
        "low_confidence": "",                     # <70% (nessuno)
    },
    "Negative": {
        "primary": "[sighs]",
        "alternatives": [
            "[gasps]",
            "[sad]",
            "[clears throat]"
        ],
        "high_confidence": "[sighs]",
        "medium_confidence": "[sighs]",
        "low_confidence": "[clears throat]",
    },
    "Neutral": {
        "primary": "",                            # Nessun tag
        "alternatives": [
            "[clears throat]",
            "[breath]"
        ],
        "high_confidence": "",
        "medium_confidence": "",
        "low_confidence": "[clears throat]",
    }
}
```

## 🧪 Come Testare

### Test 1: Verifica Tag Disponibili (veloce)

```bash
python test_emotional_tags.py
```

Output: Lista di tutti i tag + configurazione (1 secondo)

### Test 2: Genera Audio con Tag Diversi

```bash
python test_bark_multiple_tags.py
```

Output: Audio multipli in `outputs/bark_tags_comparison/`

### Test 3: Confronto A/B Manuale

Ascolta gli audio generati e confronta:

- `positive_tag_0_positive.wav` - [laughs]
- `positive_tag_1_positive.wav` - [chuckles]
- `positive_tag_2_positive.wav` - [giggles]
- `positive_tag_3_positive.wav` - [laughter]

## 💡 Esempi Pratici

### Esempio 1: Usa Default (Automatico)

```python
# Sistema sceglie automaticamente tag basato su confidenza
audio = generate_emotional_audio(
    emotion="Positive",
    confidence=0.92,  # Alta → usa [laughs]
    video_name="video_001",
    output_dir="results/audio",
    caption="I am so happy!"
)
```

### Esempio 2: Forza Tag Specifico

```python
# Forza uso di [giggles] per suono più leggero
audio = generate_emotional_audio(
    emotion="Positive",
    confidence=0.92,
    video_name="video_001",
    output_dir="results/audio",
    caption="I am so happy!",
    alternative_tag=2,  # [giggles]
    confidence_based_tags=False
)
```

### Esempio 3: Confidence-Based per Batch

```python
from src.tts.bark import preload_bark_models, generate_emotional_audio

# Pre-carica modelli
preload_bark_models()

# Genera audio per molti video (tag automatico per confidenza)
for video in videos:
    audio = generate_emotional_audio(
        emotion=video.emotion,
        confidence=video.confidence,  # ← Adatta tag automaticamente!
        video_name=video.name,
        output_dir="results/batch",
        caption=video.caption,
        confidence_based_tags=True,  # ✅ Abilita auto-selezione
        preload=True
    )
```

## 📈 Quando Usare Quale Tag?

### POSITIVE

| Tag          | Quando Usare               | Confidenza | Esempio Caption        |
| ------------ | -------------------------- | ---------- | ---------------------- |
| `[laughs]`   | Alta emozione positiva     | >90%       | "I won the lottery!"   |
| `[chuckles]` | Emozione positiva moderata | 70-90%     | "This is pretty good." |
| `[giggles]`  | Emozione giocosa/leggera   | Custom     | "That's funny!"        |
| `[laughter]` | Variante di laughs         | Custom     | "Amazing news!"        |

### NEGATIVE

| Tag               | Quando Usare             | Confidenza | Esempio Caption          |
| ----------------- | ------------------------ | ---------- | ------------------------ |
| `[sighs]`         | Tristezza/frustrazione   | >70%       | "This is disappointing." |
| `[gasps]`         | Shock negativo           | Custom     | "Oh no!"                 |
| `[sad]`           | Tristezza profonda       | Custom     | "I feel so sad."         |
| `[clears throat]` | Bassa confidenza/disagio | <70%       | "This is... not great."  |

## 🔧 API Reference

### Nuove Funzioni

```python
from src.tts.bark.emotion_mapper import (
    get_emotional_tag,
    get_alternative_emotional_tags
)

# Ottieni tag per confidenza
tag = get_emotional_tag("Positive", confidence=0.92)
# → "[laughs]" (alta confidenza)

# Ottieni tag alternativo
tag = get_emotional_tag("Positive", alternative=1)
# → "[chuckles]"

# Ottieni tutti i tag disponibili
tags = get_alternative_emotional_tags("Positive")
# → ['[laughs]', '[chuckles]', '[giggles]', '[laughter]']
```

### Parametri TTS Generator

```python
generate_emotional_audio(
    emotion: str,
    confidence: float,
    video_name: str,
    output_dir: str,
    caption: str = None,

    # 🆕 Nuovi parametri tag
    alternative_tag: int = 0,           # 0=default, 1+=alternatives
    confidence_based_tags: bool = True, # Auto-selezione basata su confidence

    # Parametri esistenti
    use_emotional_tags: bool = True,
    optimize_tag_placement: bool = True,
    alternative_speaker: int = 0,
    preload: bool = False
)
```

## 🎯 Vantaggi Sistema Multi-Tag

✅ **Varietà:** 4 tag diversi per Positive, 4 per Negative
✅ **Adattivo:** Tag cambia automaticamente con la confidenza
✅ **Controllo:** Puoi forzare tag specifici quando serve
✅ **Ottimizzato:** Tag posizionati intelligentemente nel testo
✅ **Espressività:** Più sfumature emotive nell'audio generato

## 📚 File Creati/Modificati

### Nuovi File

- ✅ `docs/BARK_EMOTIONAL_TAGS.md` - Lista completa tag + guida
- ✅ `test_emotional_tags.py` - Test tag disponibili
- ✅ `test_bark_multiple_tags.py` - Genera audio con tag diversi
- ✅ `docs/MULTI_TAG_SYSTEM.md` - Questo documento

### File Modificati

- ✅ `src/tts/bark/emotion_mapper.py` - Aggiunti `EMOTIONAL_TAGS`, `get_emotional_tag()`, `get_alternative_emotional_tags()`
- ✅ `src/tts/bark/tts_generator.py` - Aggiunti parametri `alternative_tag`, `confidence_based_tags`
- ✅ `src/tts/bark/__init__.py` - Export nuove funzioni

## 🎓 Conclusioni

Il sistema ora supporta **12+ tag emotivi diversi** invece dei soli 2 iniziali ([laughs], [sighs]).

Hai 3 modalità di utilizzo:

1. **Automatico** (raccomandato): Tag scelto da confidenza
2. **Manuale**: Forza tag specifico
3. **Ibrido**: Combina entrambi

Per la pipeline ViViT, usa modalità **automatica** per adattare l'espressività alla sicurezza della predizione! 🚀

---

**Implementato:** 29 Ottobre 2025
**Status:** ✅ Production Ready
**Testing:** Pronto per generazione audio
