# 🎯 Workflow SONAR: Colab + Mac - Spiegato Chiaramente

**Domanda**: Perché usare Colab se poi faccio fine-tuning sul Mac?

**Risposta**: Perché **feature extraction** e **fine-tuning** sono due fasi separate con requisiti diversi!

---

## 🔍 Le Due Fasi del Processo

### FASE 1: Feature Extraction (Google Colab)

**Cosa fa**: Estrae rappresentazioni visive dai video usando SONAR SignHiera

**Input**:

- Video How2Sign (.mp4)
- 6229 video totali
- ~50 GB di dati

**Processing**:

```python
# Per ogni video:
video_frames = load_video("video.mp4")        # (T, 3, 224, 224)
features = SignHiera_model(video_frames)      # (T, 256)
save_features("video.npy", features)          # Salva su disco
```

**Output**:

- File `.npy` per ogni video
- Shape: `(num_frames, 256)` per video
- 6229 file totali
- ~3 GB di dati

**Requisiti**:

- ❌ Mac M-series: NON supportato (no CUDA, incompatibilità SSVP-SLT)
- ✅ Google Colab: Linux + CUDA T4 (ambiente ufficialmente supportato)

**Tempo**: 8-11 ore

---

### FASE 2: Fine-Tuning (Mac Locale)

**Cosa fa**: Allena SONAR Encoder a tradurre feature → testo inglese

**Input**:

- Feature `.npy` (già estratte su Colab)
- ~3 GB di dati

**Processing**:

```python
# Per ogni sample:
features = load_features("video.npy")         # (T, 256)
embeddings = SONAR_encoder(features)          # (T, 1024)
translation = decoder(embeddings)             # "I love programming"
loss = compute_loss(translation, ground_truth)
optimizer.step()                               # Aggiorna pesi
```

**Output**:

- Modello SONAR fine-tunato per How2Sign
- Checkpoints durante training

**Requisiti**:

- ✅ Mac M-series: Perfettamente compatibile!
- Feature già estratte (nessun problema di dipendenze video)
- Solo PyTorch + transformers (funzionano su Mac)

**Tempo**: 1-2 giorni training

**Performance attesa**: BLEU-4 30-35%

---

## 💡 Perché Questa Separazione?

### Problema: Mac M-series NON supporta SSVP-SLT

**SSVP-SLT richiede**:

- Linux OS (REQUIRED in INSTALL.md)
- CUDA >= 11.7 (REQUIRED)
- torch 2.2.0 compilato per CUDA
- torchvision 0.17.0 compilato per CUDA

**Mac M-series ha**:

- macOS (non Linux)
- Apple Silicon GPU (non CUDA)
- torch compilato per Metal/MPS
- Incompatibilità a livello di piattaforma

**Risultato**: Impossibile eseguire SSVP-SLT su Mac

### Soluzione: Divide et Impera

**Parte pesante (feature extraction)**:

- Richiede Linux + CUDA → Eseguila su Colab
- Output: Feature leggere (3 GB)

**Parte leggera (fine-tuning)**:

- Non richiede dipendenze specifiche → Eseguila su Mac
- Input: Feature già estratte

---

## 📊 Confronto Diretto

### Opzione A: Tutto su Colab

| Fase               | Colab               | Note                       |
| ------------------ | ------------------- | -------------------------- |
| Feature extraction | ✅ 8-11 ore         | Funziona bene              |
| Upload video       | ⚠️ 2-5 ore          | 50 GB da caricare          |
| Fine-tuning        | ⚠️ 1-2 giorni       | Runtime limit (12h free)   |
| Costo              | GRATIS (con limiti) | Può disconnettere          |
| Controllo          | ❌ Limitato         | Devi tenere tab aperto     |
| **TOTALE**         | **~3-4 giorni**     | Con possibili interruzioni |

### Opzione B: Colab + Mac (CONSIGLIATA)

| Fase               | Dove            | Tempo               | Note                 |
| ------------------ | --------------- | ------------------- | -------------------- |
| Feature extraction | Colab           | 8-11 ore            | Funziona bene        |
| Upload video       | Drive           | 2-5 ore             | Una volta sola       |
| Download feature   | Mac             | 30 min              | Solo 3 GB            |
| Fine-tuning        | **Mac**         | 1-2 giorni          | **Pieno controllo**  |
| Costo              | GRATIS          | -                   | Nessun runtime limit |
| Controllo          | ✅ Totale       | -                   | Lavori in locale     |
| **TOTALE**         | **~2-3 giorni** | Workflow più fluido |

---

## 🎯 Vantaggi del Workflow Colab + Mac

### ✅ Sfrutta i punti di forza di entrambi

**Google Colab**:

- Ambiente Linux + CUDA (per SSVP-SLT)
- GPU T4 gratis (per feature extraction)
- Setup veloce (5 minuti)

**Mac Locale**:

- Fine-tuning senza limiti di tempo
- Pieno controllo del processo
- Nessun problema di disconnessione
- Salvataggio checkpoints affidabile

### ✅ Dati più leggeri

| Fase          | Trasferimento      | Dimensione    |
| ------------- | ------------------ | ------------- |
| Video → Colab | Upload             | 50 GB         |
| Feature → Mac | Download           | **3 GB**      |
| **Risparmio** | **47 GB in meno!** | 94% riduzione |

### ✅ Riproducibilità

- Feature extraction: Ambiente standard (Colab)
- Fine-tuning: Tuo ambiente (Mac)
- Risultati riproducibili e confrontabili

---

## 📝 Timeline Dettagliata

### Giorno 1 (Setup + Test)

```
09:00 - Upload cartella test su Google Drive (5 min)
09:05 - Setup Colab notebook (5 min)
09:10 - Test estrazione 5 video (10 min)
09:20 - ✅ Verifica funzionamento

09:30 - Inizia upload video completi (2-5 ore in background)
14:30 - Upload completato
```

### Giorno 1-2 (Feature Extraction)

```
14:30 - Avvia estrazione train (3-4 ore)
18:30 - Avvia estrazione val (2-3 ore)
21:30 - Avvia estrazione test (3-4 ore)
01:30 - ✅ Tutte le feature estratte (giorno 2)
```

### Giorno 2 (Download + Setup Fine-tuning)

```
09:00 - Download feature da Drive (30 min)
09:30 - Verifica feature locali (10 min)
10:00 - Setup script fine-tuning (30 min)
10:30 - ✅ Pronto per training
```

### Giorno 2-4 (Fine-Tuning)

```
10:30 - Avvia fine-tuning Stage 1 (24h)
10:30 (giorno 3) - Avvia Stage 2 (24h)
10:30 (giorno 4) - ✅ Training completato

Valutazione finale:
- BLEU-4: 30-35%
- Confronto con Landmarks
```

**TOTALE**: 3-4 giorni dal setup ai risultati finali

---

## 🤔 Domande Frequenti

### Q: Perché non fare tutto sul Mac?

**A**: Perché SSVP-SLT (necessario per feature extraction) richiede Linux + CUDA. Il Mac M-series non è supportato.

### Q: Perché non fare tutto su Colab?

**A**: Si può, ma:

- Runtime limit (12h free tier)
- Possibili disconnessioni
- Meno controllo
- Fine-tuning richiede giorni (non ore)

### Q: Le feature estratte sono compatibili con Mac?

**A**: SÌ! Le feature sono semplici array NumPy (.npy), funzionano su qualsiasi piattaforma.

### Q: Posso usare le feature per altro?

**A**: SÌ! Una volta estratte, puoi:

- Fine-tune SONAR
- Trainare altri modelli (LSTM, Transformer)
- Fare analisi esplorative
- Usarle per altri task

### Q: E se non voglio usare Colab?

**A**: Alternative:

- **Server Linux con GPU**: Segui INSTALL.md ufficiale SSVP-SLT
- **Landmarks approach**: Già pronto sul Mac, nessun Colab necessario

---

## 🎓 Per la Tesi

Questo workflow è **metodologicamente corretto** perché:

1. **Giustificazione chiara**: Mac non supporta SSVP-SLT → Uso Colab
2. **Best practice**: Separazione feature extraction / fine-tuning
3. **Riproducibile**: Chiunque può ripetere con stesso ambiente Colab
4. **Efficient**: Solo la parte GPU-intensive su Colab

**Nella tesi puoi scrivere**:

> "A causa dell'incompatibilità della piattaforma macOS con i requisiti di SSVP-SLT
> (Linux + CUDA), abbiamo adottato un approccio in due fasi: (1) estrazione di
> feature visuali su Google Colab (ambiente Linux + CUDA T4), producendo
> rappresentazioni compatte (3 GB) dai video originali (50 GB); (2) fine-tuning
> del modello SONAR in locale, sfruttando le feature pre-estratte. Questo workflow
> combina i vantaggi dell'ambiente cloud (accesso a GPU, compatibilità) con il
> controllo locale durante il training."

---

## ✅ Conclusione

**Workflow Colab + Mac**:

- ✅ Risolve incompatibilità piattaforma
- ✅ Sfrutta meglio le risorse
- ✅ Più veloce ed efficiente
- ✅ Metodologicamente solido
- ✅ Dati leggeri (3 GB vs 50 GB)

**Prossimo passo**: Segui `README.md` per iniziare! 🚀
