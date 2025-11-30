# 🗣️ Multimodal Emotional TTS Generation (Bark)

> **Modulo di Sintesi Vocale Espressiva per il progetto EmoSign.** > Questo componente trasforma le predizioni multimodali (Video + Testo) in audio parlato, modulando tono, prosodia ed espressioni non verbali (risate, sospiri) in base all'emozione rilevata e al livello di certezza del modello.

---

## 🏗️ Architettura del Sistema

Il sistema non è un semplice Text-to-Speech (TTS), ma una pipeline **Context-Aware** che adatta la voce al contesto emotivo del video originale.

### Flusso dei Dati

1.  **Input:** Predizioni del Meta-Learner (Classi: _Positive, Negative, Neutral_ + Score di Confidenza).
2.  **Data Enrichment (`merge_captions.py`):** Recupera il testo originale (caption) dal dataset di test.
3.  **Emotion Mapping (`emotion_mapper.py`):** Traduce i dati numerici in parametri acustici (Speaker ID, Temperatura).
4.  **Text Optimization (`emotion_tag_optimizer.py`):** Inserisce tag emotivi (es. `[laughs]`) nel testo in posizioni sintatticamente corrette.
5.  **Synthesis (`tts_generator.py`):** Genera l'audio waveform utilizzando il modello generativo **Suno Bark**.

---

## 📂 Struttura del Modulo

Il codice si trova in `src/tts/bark/` ed è organizzato nei seguenti script:

| Script                         | Funzione Principale                                                                                                                  |
| :----------------------------- | :----------------------------------------------------------------------------------------------------------------------------------- |
| **`tts_generator.py`**         | **Motore Core.** Gestisce il caricamento del modello Bark, l'allocazione VRAM (ottimizzato per A100) e il loop di generazione batch. |
| **`emotion_mapper.py`**        | **Cervello Emotivo.** Definisce le regole di mappatura (es. _Alta Confidenza Positiva_ = Speaker 6 + `[laughter]`).                  |
| **`emotion_tag_optimizer.py`** | **Linguista.** Analizza la frase per inserire i tag emotivi senza rompere il flusso del discorso (es. alle virgole).                 |
| **`text_templates.py`**        | **Fallback.** Fornisce frasi sintetiche ("I am feeling...") se la caption originale è mancante.                                      |
| **`analyze_tts_results.py`**   | **Validazione.** Analizza gli audio generati (durata, copertura) e seleziona campioni per l'ascolto qualitativo.                     |
| **`analyze_prosody.py`**       | **Analisi Scientifica.** Estrae Pitch (F0) e Spettrogrammi per dimostrare oggettivamente la varianza emotiva.                        |

---

## 🧠 Logica di Espressività "Adaptive Confidence"

Il sistema non applica le emozioni in modo binario, ma scala l'intensità in base alla **confidenza** del classificatore (Meta-Learner).

### 1. Selezione Speaker (Identity Consistency)

- Lo speaker non è scelto a caso, ma tramite un **hash deterministico** sul nome del video.
- **Risultato:** Lo stesso video avrà sempre la stessa voce, ma il dataset nel complesso avrà una grande varietà di voci.

### 2. Soglie Emotive Asimmetriche

Abbiamo calibrato le soglie analizzando la distribuzione delle probabilità del modello (che tende a essere molto sicuro sui positivi e cauto sui negativi).

| Emozione     | Livello Confidenza      | Azione TTS           | Tag Esempio                 |
| :----------- | :---------------------- | :------------------- | :-------------------------- |
| **POSITIVE** | **Alta (> 0.92)**       | Massima Espressività | `[laughter]` (Risata piena) |
|              | **Media (0.75 - 0.92)** | Allegria Standard    | `[laughs]` (Sorriso)        |
|              | **Bassa (< 0.75)**      | Incertezza / Cautela | `[clears throat]`           |
| **NEGATIVE** | **Alta (> 0.65)**       | Drammaticità         | `[sighs]` (Sospiro)         |
|              | **Media (0.55 - 0.65)** | Sorpresa Negativa    | `[gasps]`                   |
|              | **Bassa (< 0.55)**      | Esitazione           | `...` (Pausa)               |
| **NEUTRAL**  | **Qualsiasi**           | Tono Piatto          | Nessun tag                  |

> **Nota:** Per la bassa confidenza, usiamo interiezioni come `uhm...` o pause, simulando un parlato naturale ma incerto, evitando di "allucinare" emozioni forti quando il modello non è sicuro.

---

## 🚀 Istruzioni per l'Esecuzione

### 1. Preparazione Dati

Prima di generare l'audio, assicurati di aver unito le predizioni con le caption corrette:

```bash
python src/utils/merge_captions.py
```
