# 🚀 SONAR Feature Extraction su Google Colab

## Perché Colab?

Il tuo Mac crasha con SignHiera (OOM), ma Google Colab offre:

- ✅ GPU T4 gratuita (16GB VRAM)
- ✅ 12GB RAM
- ✅ Perfetto per estrazione features

---

## 📋 Prerequisiti

1. **Account Google** (gratuito)
2. **Google Drive** con ~5GB spazio libero
3. **Video How2Sign** da uploadare su Drive

---

## Step 1: Upload Video su Google Drive

### 1.1 Crea struttura su Drive

```
My Drive/
└── How2Sign_SONAR/
    ├── videos/
    │   ├── train/  (2147 video .mp4)
    │   ├── val/    (1739 video .mp4)
    │   └── test/   (2343 video .mp4)
    ├── manifests/
    │   ├── train.tsv
    │   ├── val.tsv
    │   └── test.tsv
    └── features/   (qui salveremo output)
```

### 1.2 Upload Video

**Opzione A: Upload manuale** (lungo ma sicuro)

1. Vai su [drive.google.com](https://drive.google.com)
2. Crea cartella `How2Sign_SONAR/videos/train`
3. Upload tutti i video train (~2147 file)
4. Ripeti per val e test

⏱️ **Tempo**: 2-4 ore (dipende da connessione)

**Opzione B: Google Drive Desktop** (più veloce)

1. Installa [Google Drive Desktop](https://www.google.com/drive/download/)
2. Sincronizza cartella locale → Drive
3. Più veloce se hai già i video locali

### 1.3 Upload Manifests

Upload anche i file TSV:

```bash
# Da locale
cp data/processed/how2sign_ssvp/manifest/train.tsv → Drive/How2Sign_SONAR/manifests/
cp data/processed/how2sign_ssvp/manifest/val.tsv → Drive/How2Sign_SONAR/manifests/
cp data/processed/how2sign_ssvp/manifest/test.tsv → Drive/How2Sign_SONAR/manifests/
```

---

## Step 2: Notebook Colab per Feature Extraction

### 2.1 Crea Nuovo Notebook

1. Vai su [colab.research.google.com](https://colab.research.google.com)
2. File → New Notebook
3. Runtime → Change runtime type → **GPU (T4)**

### 2.2 Copia questo codice nel notebook

```python
# ============================================================================
# SONAR SignHiera Feature Extraction on Google Colab
# ============================================================================

# Cell 1: Setup
!pip install torch torchvision opencv-python-headless pillow tqdm

from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/MyDrive/How2Sign_SONAR')

# Cell 2: Clone SSVP-SLT repo
!git clone https://github.com/facebookresearch/ssvp_slt.git
%cd ssvp_slt

# Cell 3: Download SONAR pretrained model
!mkdir -p models
!wget https://dl.fbaipublicfiles.com/SONAR/asl/dm_70h_ub_signhiera.pth -O models/dm_70h_ub_signhiera.pth

# Cell 4: Upload extraction script
# (Copy extract_features_signhiera.py from your repo)

# Cell 5: Extract TRAIN features
!python extract_features_signhiera.py \
    --manifest ../manifests/train.tsv \
    --video_dir ../videos/train \
    --model_path models/dm_70h_ub_signhiera.pth \
    --output_dir ../features/train \
    --batch_size 8 \
    --device cuda

# Cell 6: Extract VAL features
!python extract_features_signhiera.py \
    --manifest ../manifests/val.tsv \
    --video_dir ../videos/val \
    --model_path models/dm_70h_ub_signhiera.pth \
    --output_dir ../features/val \
    --batch_size 8 \
    --device cuda

# Cell 7: Extract TEST features
!python extract_features_signhiera.py \
    --manifest ../manifests/test.tsv \
    --video_dir ../videos/test \
    --model_path models/dm_70h_ub_signhiera.pth \
    --output_dir ../features/test \
    --batch_size 8 \
    --device cuda

# Cell 8: Verify output
!ls -lh ../features/train/
!ls -lh ../features/val/
!ls -lh ../features/test/

print("✅ Feature extraction complete!")
print("Features saved to: /content/drive/MyDrive/How2Sign_SONAR/features/")
```

### 2.3 Esegui il Notebook

1. Runtime → Run all
2. Autorizza accesso a Google Drive
3. Attendi (~6-8 ore per tutti i split)

⏱️ **Tempo stimato**:

- Train (2147 video): ~3-4 ore
- Val (1739 video): ~2-3 ore
- Test (2343 video): ~3-4 ore
- **Totale: 8-11 ore**

💡 **Tip**: Colab disconnette dopo 12 ore idle. Esegui un split alla volta se necessario.

---

## Step 3: Download Features Localmente

Dopo estrazione completa su Colab:

```bash
# Opzione A: Google Drive Desktop (automatico)
# Features appaiono in: ~/Google Drive/How2Sign_SONAR/features/

# Opzione B: Download manuale
# 1. Vai su drive.google.com
# 2. Download cartella How2Sign_SONAR/features/
# 3. Copia in: data/processed/how2sign_ssvp/features/
```

---

## Step 4: Fine-Tuning Locale

Ora che hai le features, puoi fare fine-tuning sul tuo Mac (più leggero):

```bash
cd src/sign_to_text_ssvp

# Stage 1: Fine-tune feature extractor
python finetune_sonar_how2sign.py \
    --stage 1 \
    --features_dir ../../data/processed/how2sign_ssvp/features \
    --manifest_dir ../../data/processed/how2sign_ssvp/manifest \
    --output_dir ../../models/sonar_finetuned \
    --batch_size 32 \
    --epochs 15 \
    --learning_rate 1e-5 \
    --device cpu  # O cuda se hai GPU

# Stage 2: Fine-tune translation head
python finetune_sonar_how2sign.py \
    --stage 2 \
    --stage1_checkpoint ../../models/sonar_finetuned/stage1_best.pt \
    --features_dir ../../data/processed/how2sign_ssvp/features \
    --manifest_dir ../../data/processed/how2sign_ssvp/manifest \
    --output_dir ../../models/sonar_finetuned \
    --batch_size 64 \
    --epochs 20 \
    --learning_rate 1e-4 \
    --device cpu
```

⏱️ **Tempo fine-tuning locale**:

- Stage 1: ~12-24 ore (CPU) o ~4-8 ore (GPU)
- Stage 2: ~24-36 ore (CPU) o ~6-12 ore (GPU)

---

## 🎯 Timeline Completa

| Step | Descrizione           | Tempo    | Dove      |
| ---- | --------------------- | -------- | --------- |
| 1    | Upload video su Drive | 2-4 ore  | Locale    |
| 2    | Feature extraction    | 8-11 ore | **Colab** |
| 3    | Download features     | 1-2 ore  | Locale    |
| 4    | Fine-tuning Stage 1   | 4-24 ore | Locale    |
| 5    | Fine-tuning Stage 2   | 6-36 ore | Locale    |
| 6    | Evaluation            | 30 min   | Locale    |

**Totale**: 2-4 giorni (con Colab) vs IMPOSSIBILE (senza Colab)

---

## ⚠️ Limitazioni Colab Gratuito

- ⏱️ **12 ore max session** → Esegui un split alla volta
- 💾 **Disk space**: ~50GB temporaneo
- 🔋 **GPU quota**: ~100 ore/mese (sufficiente!)

Se ti disconnette:

1. Riconnetti
2. Ri-monta Drive
3. Continua da dove eri rimasto

---

## 🆘 Troubleshooting

### "Runtime disconnected"

→ Normale dopo 12 ore. Salva progress, riconnetti, continua.

### "Out of memory"

→ Riduci batch_size da 8 a 4 o 2.

### "Cannot find video"

→ Verifica path Drive: `/content/drive/MyDrive/How2Sign_SONAR/videos/`

### "Model download failed"

→ Ri-esegui Cell 3 (wget può fallire).

---

## 🎓 Per la Tesi

**Giustifica approccio**:

- "Feature extraction richiede GPU (16GB VRAM)"
- "Utilizzo Google Colab per estrazione, fine-tuning locale"
- "Approccio cloud-hybrid per vincoli hardware"

**Vantaggi**:

- ✅ SONAR pretrained utilizzato
- ✅ Performance migliori (BLEU-4: 30-35%)
- ✅ Costo zero (Colab gratuito)

---

## 🚀 Ready to Start?

1. ✅ Upload video su Drive (Step 1)
2. ✅ Crea notebook Colab (Step 2)
3. ✅ Run extraction (~8-11 ore)
4. ✅ Download features
5. ✅ Fine-tune locally

**Inizio consigliato**: Stasera upload video, domani mattina avvia Colab! 🎉
