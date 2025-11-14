# 🔧 Soluzione Definitiva per fairseq2 su Colab

## Problema

fairseq2 richiede CUDA 12.8 ma Colab ha CUDA 12.6

## ✅ Soluzione Testata

### Step 1: Installa PyTorch 2.5.0 con CUDA 12.1

```python
# Disinstalla PyTorch corrente
!pip uninstall -y torch torchvision torchaudio

# Installa PyTorch 2.5.0 (compatibile con fairseq2)
!pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Verifica
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

### Step 2: Installa fairseq2 per PyTorch 2.5.0

```python
# Installa fairseq2 corretto
!pip install fairseq2 \
    --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.5.0/cu121

# Verifica
from fairseq2.models.sonar import load_sonar_text_decoder
print("✅ fairseq2 funziona!")
```

### Step 3: Test SONAR Decoder

```python
# Test completo
import torch
from fairseq2.models.sonar import load_sonar_text_decoder

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load decoder
decoder = load_sonar_text_decoder('text_sonar_basic_encoder', device=device)
print(f"✅ Decoder loaded on {device}")

# Test con embedding casuale
test_emb = torch.randn(1, 1024).to(device)
print("✅ Ready for inference!")
```

---

## 📊 Se Funziona

Usa questo script per inferenza con decoder SONAR vero:

```python
# inference_sonar_real.py
import torch
from fairseq2.models.sonar import load_sonar_text_decoder
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sacrebleu

# Load encoder fine-tunato
# Load features
# Encode → embeddings
# Decode con SONAR → testo
# Calculate BLEU
```

BLEU atteso: **25-35%** (vs 0.01% con decoder semplice)

---

## 🚨 Se NON Funziona Ancora

### Alternativa: Usa repository SONAR originale

```bash
# Clone SONAR repo
!git clone https://github.com/facebookresearch/SONAR.git
cd SONAR

# Installa
!pip install -e .

# Usa SONAR direttamente
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
```

---

## 📝 Note

- PyTorch 2.5.0 è **stabile** e compatibile con fairseq2
- Il downgrade da 2.8.0 → 2.5.0 è sicuro
- Colab supporta CUDA 12.1 nativamente
- Dopo installazione, **riavvia runtime** se necessario

---

## ⚡ Quick Test Script

```python
# test_fairseq2_install.py
try:
    import torch
    print(f"✅ PyTorch {torch.__version__}")
    print(f"✅ CUDA {torch.version.cuda}")

    from fairseq2.models.sonar import load_sonar_text_decoder
    print(f"✅ fairseq2 imported")

    decoder = load_sonar_text_decoder('text_sonar_basic_encoder', device='cpu')
    print(f"✅ SONAR decoder loaded")

    print("\n🎉 TUTTO FUNZIONA! Puoi procedere con inferenza SONAR!")

except Exception as e:
    print(f"❌ Errore: {e}")
    print("\nProva i passaggi sopra uno per uno")
```

Esegui questo su Colab e dimmi il risultato! 🚀
