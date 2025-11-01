# 🔧 Bark TTS - Troubleshooting & FAQ

## ⚠️ Problema: PyTorch 2.9+ Compatibility Error

### Errore:

```
_pickle.UnpicklingError: Weights only load failed...
torch.load with `weights_only` set to `False` will likely succeed...
```

### Causa:

PyTorch 2.6+ ha cambiato il default di `weights_only=True` per sicurezza, ma Bark usa checkpoint salvati con il vecchio formato.

### ✅ Soluzione:

Il progetto include già un **patch automatico** (`src/tts/bark/pytorch_patch.py`) che risolve il problema.

**Come usarlo:**

```python
# Il patch viene applicato automaticamente quando importi:
from src.tts.bark.tts_generator import generate_emotional_audio

# Oppure applicalo manualmente:
from src.tts.bark import pytorch_patch
```

Il patch è **sicuro** perché:

- Bark è un modello open-source affidabile (Suno AI)
- Applica `weights_only=False` solo per il caricamento di Bark
- Non influenza altri caricamenti di PyTorch

---

## 🐌 Problema: Generazione molto lenta

### Sintomo:

```
No GPU being used. Careful, inference might be very slow!
Generazione richiede 30-60 secondi per clip
```

### Causa:

Bark usa la CPU invece della GPU.

### ✅ Soluzioni:

#### Opzione 1: Pre-carica modelli (raccomandato)

```python
from src.tts.bark.tts_generator import preload_bark_models

# Pre-carica UNA VOLTA all'inizio
preload_bark_models()

# Poi genera molti audio velocemente
for video in videos:
    audio = generate_emotional_audio(..., preload=True)
```

#### Opzione 2: Usa GPU (MPS su Mac, CUDA su Linux/Windows)

Bark usa automaticamente la GPU se disponibile tramite PyTorch.

**Verifica GPU:**

```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

#### Opzione 3: Riduci batch size

Genera audio uno alla volta invece che in batch.

---

## 💾 Problema: Out of Memory

### Errore:

```
RuntimeError: [enforce fail at alloc_cpu.cpp:73] posix_memalign...
Out of memory
```

### Causa:

Bark richiede ~10GB RAM per caricare tutti i modelli.

### ✅ Soluzioni:

#### Opzione 1: NON pre-caricare i modelli

```python
generate_emotional_audio(..., preload=False)
```

- Pro: Usa meno RAM (~2-3GB)
- Contro: Più lento (carica modelli ogni volta)

#### Opzione 2: Aumenta swap/virtual memory

macOS/Linux:

```bash
# Verifica RAM disponibile
free -h  # Linux
vm_stat  # macOS
```

#### Opzione 3: Chiudi altre applicazioni

Libera RAM chiudendo browser, IDE pesanti, etc.

---

## 📥 Problema: Download modelli fallito

### Errore:

```
Failed to download model from HuggingFace...
Connection timeout
```

### ✅ Soluzioni:

#### Opzione 1: Riprova (spesso è temporaneo)

```bash
python test_bark_quick.py
```

#### Opzione 2: Download manuale

```python
from bark import preload_models
preload_models()  # Forza il download
```

#### Opzione 3: Usa cache HuggingFace locale

```bash
export HF_HOME=/path/to/large/disk
python test_bark_quick.py
```

---

## 🎵 Problema: Audio generato è silenzioso/distorto

### Causa:

Temperature troppo bassa o speaker non adatto.

### ✅ Soluzioni:

#### Soluzione 1: Aumenta temperature

```python
# In emotion_mapper.py
EMOTION_BARK_MAPPING = {
    "Positive": {
        "temperature": 0.9,  # Era 0.7
        ...
    }
}
```

#### Soluzione 2: Prova speaker alternativi

```python
audio = generate_emotional_audio(
    emotion="Positive",
    alternative_speaker=1,  # Prova 0, 1, 2
    ...
)
```

---

## 🔄 Problema: Audio diverso ogni volta

### Causa:

Bark è un modello generativo - produce variazioni naturali.

### ✅ Per audio consistente:

#### Opzione 1: Riduci temperature

```python
EMOTION_BARK_MAPPING = {
    "Positive": {
        "temperature": 0.3,  # Più deterministico
        ...
    }
}
```

#### Opzione 2: Usa stesso seed (se supportato)

Attualmente Bark non supporta seed fissi.

---

## 📦 Test Rapidi

### Test minimale (1 audio):

```bash
python test_bark_quick.py
```

### Test completo (3 audio + baseline):

```bash
python test_tts_bark.py
```

### Pipeline ViViT completa:

```bash
python src/models/two_classes/vivit/test_golden_labels_vivit.py \
  --model_uri mlartifacts/.../artifacts \
  --batch_size 1 \
  --save_results \
  --generate_tts
```

---

## 📊 Metriche attese

### Performance tipiche (CPU):

| Metrica                       | Valore  | Note                       |
| ----------------------------- | ------- | -------------------------- |
| Caricamento modelli           | 30-60s  | Solo prima volta           |
| Generazione 1 clip            | 10-30s  | Dipende da lunghezza testo |
| RAM richiesta (con preload)   | ~10GB   |                            |
| RAM richiesta (senza preload) | ~2-3GB  |                            |
| Dimensione audio WAV          | 0.5-2MB | Dipende da durata          |
| Sample rate                   | 24kHz   | Fisso                      |

### Con GPU (MPS/CUDA):

| Metrica             | Valore |
| ------------------- | ------ |
| Caricamento modelli | 10-20s |
| Generazione 1 clip  | 3-8s   |

---

## 🆘 Supporto

Se i problemi persistono:

1. **Verifica installazione:**

```bash
python -c "import bark; print(bark.__version__)"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

2. **Reinstalla Bark:**

```bash
pip uninstall bark -y
pip install git+https://github.com/suno-ai/bark.git
```

3. **Controlla log dettagliati:**

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

4. **Issues GitHub:**

- [Bark Issues](https://github.com/suno-ai/bark/issues)
- Repository progetto: issue tracker

---

## ✅ Checklist Debug

- [ ] Patch PyTorch applicato (`pytorch_patch.py` importato)
- [ ] RAM disponibile > 10GB (con preload) o > 3GB (senza)
- [ ] Bark installato correttamente
- [ ] PyTorch funzionante
- [ ] Directory output esiste e writable
- [ ] Connessione internet (per download modelli prima volta)
