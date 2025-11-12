# 🎯 Fase Successiva: Estrazione Feature Completa

## 📊 Stato Attuale

✅ **Completato**:

- Setup Colab funzionante
- Pipeline video → feature → traduzione verificata
- Test inferenza zero-shot (BLEU 1-2% come atteso)

❌ **Conclusione Zero-Shot**:

- BLEU troppo basso (1-2%) per qualsiasi decoder
- Le feature non sono ottimizzate per How2Sign
- GPT-2/T5 generano testo casuale
- **Impossibile migliorare senza fine-tuning**

---

## 🚀 NEXT STEP: Estrazione Feature Completa

### 📦 Cosa fare ORA

#### **Step 1: Upload Dataset Completo su Google Drive**

1. **Scarica tutti i video How2Sign** (se non li hai già):

   - Train: 2147 video
   - Val: 1739 video
   - Test: 2343 video
   - **Totale**: ~80-100 GB

2. **Crea struttura su Google Drive**:

   ```
   How2Sign_SONAR/
   ├── videos/
   │   ├── train/      (2147 video)
   │   ├── val/        (1739 video)
   │   └── test/       (2343 video)
   ├── manifests/
   │   ├── train.tsv
   │   ├── val.tsv
   │   └── test.tsv
   ├── models/         (già presente)
   └── extract_features_signhiera.py
   ```

3. **Upload su Google Drive**:
   - Usa Google Drive Desktop
   - Oppure upload manuale (lento, ~8-12 ore)
   - Oppure usa `rclone` (più veloce)

---

#### **Step 2: Esegui Estrazione Feature su Colab**

**Cella Completa per Estrazione**:

```python
#!/usr/bin/env python3
"""
Estrazione feature SignHiera per TUTTO il dataset How2Sign
Tempo stimato: 8-11 ore su T4 GPU
Output: ~3 GB di file .pt
"""

import os
import sys
import torch
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# === Feature Extractor ===
class SignHieraFeatureExtractor:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device

        # Carica modello pre-addestrato
        from torchvision import models
        self.model = models.resnet50(pretrained=True)

        # Carica checkpoint SONAR
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            if 'model' in ckpt:
                self.model.load_state_dict(ckpt['model'], strict=False)
                print("✅ Loaded SignHiera checkpoint")

        self.model = self.model.to(device)
        self.model.eval()

    def extract(self, video_path, max_frames=300):
        """Estrae feature da un video"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        frames = []
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocessing
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frame = frame.astype(np.float32) / 255.0
            frame = np.transpose(frame, (2, 0, 1))
            frames.append(frame)

        cap.release()

        if not frames:
            return None

        # Convert to tensor
        frames_tensor = torch.from_numpy(np.stack(frames)).float()
        frames_tensor = frames_tensor.to(self.device)

        # Extract features
        with torch.no_grad():
            features = self.model(frames_tensor)

        return features.cpu()


# === Main Extraction Loop ===
def extract_all_features(split='train'):
    """
    Estrae feature per un intero split (train/val/test)

    Args:
        split: 'train', 'val', or 'test'
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Device: {device}")

    # Paths
    video_dir = Path(f'videos/{split}')
    manifest_path = f'manifests/{split}.tsv'
    output_dir = Path(f'features/{split}')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load extractor
    print("📦 Loading SignHiera model...")
    extractor = SignHieraFeatureExtractor(
        'models/dm_70h_ub_signhiera.pth',
        device=device
    )
    print("✅ Model loaded\n")

    # Load manifest
    manifest = pd.read_csv(manifest_path, sep='\t')
    print(f"📊 Processing {len(manifest)} videos from {split} split\n")

    # Extract features
    failed = []
    for idx, row in tqdm(manifest.iterrows(), total=len(manifest), desc=f"Extracting {split}"):
        video_id = row['id']
        video_path = video_dir / f"{video_id}.mp4"
        output_path = output_dir / f"{video_id}.pt"

        # Skip if already extracted
        if output_path.exists():
            continue

        # Extract
        try:
            features = extractor.extract(video_path)
            if features is not None:
                torch.save({
                    'features': features,
                    'video_id': video_id,
                    'text': row['text'],
                    'duration': row['duration']
                }, output_path)
            else:
                failed.append(video_id)
        except Exception as e:
            print(f"❌ Error on {video_id}: {e}")
            failed.append(video_id)

    print(f"\n✅ Extraction complete!")
    print(f"📊 Processed: {len(manifest) - len(failed)}/{len(manifest)}")
    print(f"❌ Failed: {len(failed)}")

    if failed:
        with open(f'features/{split}_failed.txt', 'w') as f:
            f.write('\n'.join(failed))
        print(f"💾 Failed IDs saved to features/{split}_failed.txt")


# === Execute ===
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'val', 'test'])
    args = parser.parse_args()

    extract_all_features(args.split)
```

**Esegui su Colab**:

```python
# Train (più lungo: ~8 ore)
!python extract_features_signhiera.py --split train

# Val (~6 ore)
!python extract_features_signhiera.py --split val

# Test (~7 ore)
!python extract_features_signhiera.py --split test
```

---

#### **Step 3: Download Feature sul Mac**

Dopo l'estrazione, scarica le feature estratte:

```bash
# Su Colab, comprimi le feature
!tar -czf features_train.tar.gz features/train/
!tar -czf features_val.tar.gz features/val/
!tar -czf features_test.tar.gz features/test/

# Download manualmente da Google Drive
# Oppure usa gdown:
!pip install gdown
!gdown --folder <google_drive_folder_id>
```

Sul Mac:

```bash
cd /Users/ignazioemanuelepicciche/Documents/TESI\ Magistrale\ UCBM/Improved_EmoSign_Thesis

# Estrai
tar -xzf features_train.tar.gz -C data/processed/
tar -xzf features_val.tar.gz -C data/processed/
tar -xzf features_test.tar.gz -C data/processed/
```

---

#### **Step 4: Fine-Tuning Locale (Mac)**

Ora puoi addestrare il decoder **sul Mac** con le feature estratte:

```bash
# Attiva environment
source .venv_sonar/bin/activate

# Training
python src/sign_to_text_ssvp/train_sonar_decoder.py \
    --features_dir data/processed/features \
    --output_dir models/sonar_finetuned \
    --batch_size 32 \
    --epochs 50 \
    --learning_rate 1e-4
```

---

## ⏱️ Timeline Stimata

| Fase              | Durata        | Dove         |
| ----------------- | ------------- | ------------ |
| Upload dataset    | 8-12 ore      | Google Drive |
| Estrazione train  | 8 ore         | Colab T4     |
| Estrazione val    | 6 ore         | Colab T4     |
| Estrazione test   | 7 ore         | Colab T4     |
| Download features | 2-3 ore       | Mac          |
| Fine-tuning       | 2-3 giorni    | Mac          |
| **TOTALE**        | **~4 giorni** |              |

---

## 📊 Risultati Attesi

### Zero-Shot (ATTUALE):

- BLEU: 1-2%
- Traduzioni: Casuali/Template

### Fine-Tuned (DOPO):

- BLEU: **30-35%**
- Traduzioni: **Accurate e specifiche**
- Loss: ~0.5-1.0

---

## 🎯 Decisione

Vuoi procedere con:

**A) Estrazione Feature Completa** (consigliato)

- Upload dataset completo su Drive
- Estrai tutte le feature su Colab
- Fine-tune sul Mac
- **Risultato**: Modello production-ready

**B) Continua esperimenti Zero-Shot** (non consigliato)

- Prova altri decoder (BART, mBART, etc)
- BLEU resterà comunque ~1-5%
- **Risultato**: Tempo perso, nessun miglioramento

**C) Usa Dataset più piccolo** (alternativa)

- Estrai solo 500-1000 video
- Fine-tuning più veloce (12-24 ore)
- BLEU: ~20-25% (inferiore ma accettabile)

---

Quale opzione preferisci? 🚀
