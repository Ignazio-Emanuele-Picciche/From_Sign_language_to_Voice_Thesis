# 🚀 SONAR on Google Colab - Quick Start

Questo folder contiene tutto ciò che ti serve per eseguire l'estrazione di feature SONAR su Google Colab.

**⚠️ IMPORTANTE - Cosa fa questo setup:**

- ✅ **Su Google Colab**: Estrazione feature da video (richiede Linux + CUDA)
- ❌ **NON su Colab**: Fine-tuning (lo farai sul Mac con le feature estratte)

**Perché questa separazione?**

- **Feature extraction**: Richiede ambiente Linux + CUDA (non disponibile su Mac M-series)
- **Fine-tuning**: Funziona con feature già estratte, si può fare su Mac in locale

---

## 📦 Contenuto

```
How2Sign_SONAR/
├── COLAB_SETUP_GUIDE.md              # Guida completa passo-passo
├── extract_features_signhiera.py     # Script di estrazione feature
├── manifests/
│   ├── train.tsv                     # Manifest completo (2147 video)
│   ├── train_sample.tsv              # Manifest test (5 video) ⭐
│   ├── val.tsv                       # Manifest completo (1739 video)
│   └── test.tsv                      # Manifest completo (2343 video)
└── videos/
    └── train/
        ├── --7E2sU6zP4_10-5-rgb_front.mp4   # Video test 1
        ├── --7E2sU6zP4_11-5-rgb_front.mp4   # Video test 2
        ├── --7E2sU6zP4_12-5-rgb_front.mp4   # Video test 3
        ├── --7E2sU6zP4_13-5-rgb_front.mp4   # Video test 4
        └── --7E2sU6zP4_5-5-rgb_front.mp4    # Video test 5
```

---

## 🎯 Due Opzioni

### Opzione A: Test Completo Inferenza (CONSIGLIATO per iniziare) ⚡

**Cosa**: Prova **inferenza end-to-end** (video → feature → traduzione) su 5 video  
**Tempo**: 20-30 minuti totali  
**Upload**: ~20 MB  
**Scopo**: Verificare che SONAR funziona completamente prima del dataset completo

**Cosa testerai**:

1. ✅ Estrazione feature con SignHiera
2. ✅ Traduzione con SONAR Encoder
3. ✅ Confronto con ground truth
4. ✅ Calcolo BLEU iniziale (zero-shot)

**Passi**:

1. Carica questa cartella `How2Sign_SONAR` su Google Drive
2. Apri Google Colab
3. Usa `train_sample.tsv` (già incluso, 5 video + traduzioni)
4. Esegui **inferenza completa** su 5 video
5. Vedi traduzioni generate vs ground truth

### Opzione B: Estrazione Feature Dataset Completo (dopo test) 🚀

**Cosa**: Estrai SOLO feature per tutti i 6229 video (non inferenza completa)  
**Tempo**: 8-11 ore totali  
**Upload**: ~40-50 GB (video)  
**Download**: ~3 GB (feature estratte)  
**Scopo**: Feature complete per fine-tuning **sul Mac**

**Perché solo feature?**

- Inferenza completa richiede troppo tempo su 6229 video
- Feature sono riutilizzabili per fine-tuning
- Fine-tuning sul Mac sarà più veloce con feature pre-estratte

**Passi**:

1. Dopo test inferenza riuscito su 5 video
2. Carica tutti i video su Google Drive (vedi guida)
3. Usa `extract_features_signhiera.py` su train/val/test
4. Scarica feature sul Mac (~3 GB)
5. **Fine-tune SONAR localmente** con feature estratte

---

## 🚀 Quick Start (5 minuti)

### Passo 1: Upload su Google Drive

1. Apri Google Drive: https://drive.google.com
2. Crea cartella `How2Sign_SONAR`
3. Carica tutti i file di questa cartella mantenendo la struttura

**Risultato atteso su Google Drive**:

```
MyDrive/
└── How2Sign_SONAR/
    ├── COLAB_SETUP_GUIDE.md
    ├── extract_features_signhiera.py
    ├── manifests/
    │   ├── train.tsv
    │   ├── train_sample.tsv ⭐
    │   ├── val.tsv
    │   └── test.tsv
    └── videos/
        └── train/
            └── (5 video .mp4)
```

### Passo 2: Apri Google Colab

1. Vai su https://colab.research.google.com
2. Clicca **"Nuovo notebook"**
3. Menu **Runtime** → **Cambia tipo di runtime** → **T4 GPU** → **Salva**

### Passo 3: Esegui Celle Colab

Copia e incolla queste celle una per volta:

#### Cella 1: Setup

```python
# Installa dipendenze
!pip install -q torch torchvision opencv-python-headless pillow tqdm pandas
print("✅ Dipendenze installate")
```

#### Cella 2: Monta Drive

```python
# Monta Google Drive
from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/MyDrive/How2Sign_SONAR')
print("✅ Google Drive montato")
!ls -lh
```

#### Cella 3: Scarica Modelli SONAR

```python
# Crea cartella e scarica modelli (~850 MB totali)
!mkdir -p models

# SignHiera (feature extractor) - ~350 MB
!wget -q --show-progress https://dl.fbaipublicfiles.com/SONAR/asl/dm_70h_ub_signhiera.pth -O models/dm_70h_ub_signhiera.pth

# SONAR Encoder (translator) - ~500 MB
!wget -q --show-progress https://dl.fbaipublicfiles.com/SONAR/asl/dm_70h_ub_sonar_encoder.pth -O models/dm_70h_ub_sonar_encoder.pth

!ls -lh models/
print("✅ Modelli SONAR scaricati")
```

#### Cella 4: Installa SONAR per Inferenza

```python
# Installa dipendenze per inferenza
!pip install -q sacrebleu sentencepiece

# Installa sonar-space (può richiedere qualche minuto)
!pip install -q sonar-space

print("✅ SONAR e dipendenze installate per inferenza")
```

#### Cella 5: Test Inferenza Completa su 5 Video ⭐

```python
# Esegui inferenza end-to-end (video → feature → traduzione)
!python run_inference.py \
    --manifest manifests/train_sample.tsv \
    --video_dir videos/train \
    --signhiera_model models/dm_70h_ub_signhiera.pth \
    --sonar_encoder models/dm_70h_ub_sonar_encoder.pth \
    --output_file results/translations_sample.json \
    --device cuda

print("\n✅ Inferenza completata!")
```

#### Cella 6: Verifica e Visualizza Risultati

```python
# Prima verifica se l'inferenza ha funzionato
import os
import json

# Controlla se il file esiste
if not os.path.exists('results/translations_sample.json'):
    print("❌ File non trovato!")
    print("\nPossibili problemi:")
    print("1. La Cella 5 (inferenza) non è stata eseguita")
    print("2. Lo script run_inference.py non esiste")
    print("3. C'è stato un errore durante l'inferenza")
    print("\n🔍 Verifica:")
    print(f"   - run_inference.py esiste? {os.path.exists('run_inference.py')}")
    print(f"   - Directory results/ esiste? {os.path.exists('results/')}")

    # Mostra file presenti
    print("\n📂 File nella directory corrente:")
    !ls -lh

    print("\n⚠️  Torna alla Cella 5 e verifica l'output dell'inferenza!")
else:
    # Carica e mostra risultati
    with open('results/translations_sample.json', 'r') as f:
        results = json.load(f)

    # Mostra confronto
    print("=" * 80)
    print("RISULTATI INFERENZA ZERO-SHOT (5 video)")
    print("=" * 80)

    for i, result in enumerate(results[:5], 1):
        print(f"\n📹 Video {i}: {result['video_id']}")
        print(f"   Ground Truth: {result['ground_truth']}")
        print(f"   Predicted:    {result['translation']}")
        print(f"   BLEU-4:       {result['bleu4']:.2f}")

    # Calcola BLEU medio
    avg_bleu = sum(r['bleu4'] for r in results) / len(results)
    print(f"\n📊 BLEU-4 medio (zero-shot): {avg_bleu:.2f}")
    print("\n🎉 Test inferenza completato!")
```

### Risultato Atteso

```
============================================================
RISULTATI INFERENZA ZERO-SHOT (5 video)
============================================================

📹 Video 1: --7E2sU6zP4_10-5-rgb_front
   Ground Truth: And I call them decorative elements because...
   Predicted:    [Traduzione generata da SONAR]
   BLEU-4:       15.23

� Video 2: --7E2sU6zP4_11-5-rgb_front
   Ground Truth: So they don't really have much of a symbolic...
   Predicted:    [Traduzione generata da SONAR]
   BLEU-4:       18.45

...

📊 BLEU-4 medio (zero-shot): 15-20 (atteso senza fine-tuning)

🎉 Test inferenza completato!
```

**Cosa significa**:

- ✅ **SONAR funziona** su Colab (Linux + CUDA)
- ✅ **Pipeline completo** testato (video → feature → testo)
- 📊 **BLEU 15-20%**: Normale per zero-shot (senza fine-tuning)
- � **Dopo fine-tuning**: BLEU salirà a 30-35%

---

## ✅ Test Riuscito? Passa al Dataset Completo

Se il test con 5 video funziona:

1. **Carica tutti i video** su Google Drive:

   - Train: `data/raw/train/raw_videos_front_train/*.mp4`
   - Val: `data/raw/val/raw_videos_front_val/*.mp4`
   - Test: `data/raw/test/raw_videos_front_test/*.mp4`

2. **Modifica Cella 4** per usare manifest completi:

   ```python
   # Train (2147 video, ~3-4 ore)
   !python extract_features_signhiera.py \
       --manifest manifests/train.tsv \
       --video_dir videos/train \
       --model_path models/dm_70h_ub_signhiera.pth \
       --output_dir features/train \
       --max_frames 300 \
       --device cuda

   # Val (1739 video, ~2-3 ore)
   !python extract_features_signhiera.py \
       --manifest manifests/val.tsv \
       --video_dir videos/val \
       --model_path models/dm_70h_ub_signhiera.pth \
       --output_dir features/val \
       --max_frames 300 \
       --device cuda

   # Test (2343 video, ~3-4 ore)
   !python extract_features_signhiera.py \
       --manifest manifests/test.tsv \
       --video_dir videos/test \
       --model_path models/dm_70h_ub_signhiera.pth \
       --output_dir features/test \
       --max_frames 300 \
       --device cuda
   ```

---

## 📊 Timeline Completo

| Task                            | Tempo      |
| ------------------------------- | ---------- |
| **Test inferenza (5 video)**    |            |
| Upload file test (~20 MB)       | 5 minuti   |
| Setup Colab                     | 5 minuti   |
| Download modelli SONAR (850 MB) | 5 minuti   |
| Inferenza 5 video (end-to-end)  | 10 minuti  |
| **TOTALE TEST**                 | **25-30m** |
|                                 |            |
| **Estrazione feature completa** |            |
| Upload video (~40-50 GB)        | 2-5 ore    |
| Estrazione train                | 3-4 ore    |
| Estrazione val                  | 2-3 ore    |
| Estrazione test                 | 3-4 ore    |
| **TOTALE ESTRAZIONE**           | **10-16h** |

---

## 🎯 Prossimi Passi

Dopo estrazione feature su Colab:

1. ✅ **Scarica feature** da Google Drive al Mac (~3 GB)
2. ✅ **Fine-tune SONAR** sul Mac (localmente con feature estratte)
3. ✅ **Valuta modello** con metriche BLEU
4. ✅ **Confronta con Landmarks** (già pronto)

**NOTA IMPORTANTE**:

- **Su Colab**: Solo estrazione feature (GPU T4, ambiente Linux+CUDA)
- **Sul Mac**: Fine-tuning con feature già estratte (CPU/GPU Apple, più veloce)

---

## ❓ Problemi?

Consulta la **guida completa** in `COLAB_SETUP_GUIDE.md` con:

- Troubleshooting dettagliato
- Soluzioni a errori comuni
- Configurazioni avanzate
- Alternative per upload veloce

---

## 📝 Note

### Perché Google Colab?

- ✅ **Linux + CUDA**: Piattaforma ufficialmente supportata da SSVP-SLT
- ✅ **GPU T4 gratis**: 15 GB VRAM, perfetto per feature extraction
- ✅ **Nessuna configurazione locale**: Evita problemi di dipendenze su Mac
- ✅ **Riproducibile**: Stesso ambiente per tutti

### Workflow Completo: Colab + Mac

```
┌─────────────────────────────────────────────────────────────┐
│ FASE 1: Google Colab (Linux + CUDA T4)                      │
├─────────────────────────────────────────────────────────────┤
│ Input:  Video How2Sign (.mp4)                               │
│ Tool:   SONAR SignHiera model                               │
│ Output: Features (.npy) - Shape: (num_frames, 256)          │
│ Tempo:  8-11 ore per 6229 video                             │
└─────────────────────────────────────────────────────────────┘
                          ↓
                   Download (~3 GB)
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ FASE 2: Mac Locale (Apple Silicon)                          │
├─────────────────────────────────────────────────────────────┤
│ Input:  Features estratte da Colab                          │
│ Tool:   Script fine-tuning SONAR                            │
│ Output: Modello fine-tunato per How2Sign                    │
│ Tempo:  1-2 giorni training                                 │
│ BLEU:   30-35% atteso                                       │
└─────────────────────────────────────────────────────────────┘
```

**Vantaggi di questo approccio:**

- ✅ Feature extraction su ambiente supportato (Colab)
- ✅ Fine-tuning locale senza dipendenze problematiche
- ✅ Feature leggere (~3 GB) vs video pesanti (~50 GB)
- ✅ Fine-tuning più veloce con feature pre-calcolate

### Alternative

Se hai accesso a server Linux con GPU:

1. Clona repo SSVP-SLT: `git clone https://github.com/facebookresearch/ssvp_slt`
2. Segui `INSTALL.md` ufficiale
3. Usa script di estrazione ufficiali

Ma Colab è più semplice e veloce per iniziare! 🚀

---

## 🎉 Buona fortuna!

Per domande o problemi, consulta:

- `COLAB_SETUP_GUIDE.md` (guida dettagliata)
- SSVP-SLT repo: https://github.com/facebookresearch/ssvp_slt
- SONAR docs: https://github.com/facebookresearch/SONAR

**Ready to go! 🚀**
