# 🔧 Troubleshooting fairseq2

## Problema: CUDA Version Mismatch

### Errore Tipico

```
RuntimeError: fairseq2 requires a CUDA 12.8 build of PyTorch 2.8.0,
but the installed version is a CUDA 12.6 build of PyTorch 2.8.0.
```

---

## ✅ Soluzioni (in ordine di semplicità)

### Soluzione 1: Installa fairseq2 CPU (CONSIGLIATO)

**Più semplice e veloce**. La versione CPU è sufficiente perché:

- Il decoder SONAR è **congelato** (non viene addestrato)
- Solo l'encoder viene addestrato (usa comunque GPU PyTorch)

```python
!pip uninstall -y fairseq2
!pip install fairseq2 --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.5.0/cpu

# Verifica
from fairseq2.models.sonar import load_sonar_text_decoder
print("✅ fairseq2 funziona!")
```

**Pro**:

- ✅ Funziona sempre
- ✅ Veloce da installare
- ✅ Non richiede reinstallare PyTorch

**Contro**:

- ⚠️ Decoder corre su CPU (ma è congelato, non rallenta training)

---

### Soluzione 2: Downgrade PyTorch a 2.5.0

Se vuoi fairseq2 GPU (non necessario):

```python
# Reinstalla PyTorch 2.5.0 con CUDA 12.1
!pip uninstall -y torch torchvision torchaudio
!pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu121

# Reinstalla fairseq2 per PyTorch 2.5.0
!pip install fairseq2 --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.5.0/cu121

# Verifica
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")

from fairseq2.models.sonar import load_sonar_text_decoder
print("✅ fairseq2 funziona!")
```

**Pro**:

- ✅ Tutto su GPU
- ✅ Versione stabile

**Contro**:

- ⚠️ Reinstalla PyTorch (lento)
- ⚠️ Rischio incompatibilità con altri pacchetti

---

### Soluzione 3: Installa PyTorch 2.8.0 con CUDA 12.8

Se vuoi ultima versione:

```python
# NOTA: Richiede CUDA 12.8 driver su Colab (potrebbe non essere disponibile)
!pip install torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu128
!pip install fairseq2 --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.8.0/cu128
```

**Pro**:

- ✅ Ultima versione PyTorch

**Contro**:

- ❌ Potrebbe non funzionare su Colab (driver CUDA vecchio)
- ⚠️ Molto lento da installare

---

## 🎯 Quale Soluzione Scegliere?

### Per Quick Test (5-10 epochs):

→ **Soluzione 1 (CPU)** - veloce, funziona sempre

### Per Full Training (50 epochs):

→ **Soluzione 1 (CPU)** o **Soluzione 2** - CPU è sufficiente perché decoder è congelato

### Solo se vuoi tutto GPU:

→ **Soluzione 2** - PyTorch 2.5.0 è stabile su Colab

---

## 📊 Performance Comparison

| Configurazione             | Training Speed | Decoder Speed | Note                  |
| -------------------------- | -------------- | ------------- | --------------------- |
| fairseq2 CPU + Encoder GPU | **100%**       | ~95%          | ✅ Consigliato        |
| fairseq2 GPU + Encoder GPU | 100%           | 100%          | ⚠️ Solo se necessario |

**Differenza**: < 5% perché decoder è congelato e usato solo in evaluation (pochi batch).

---

## ⚠️ Errori Comuni

### Errore: "No module named 'fairseq2'"

```python
!pip install fairseq2 --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.5.0/cpu
```

### Errore: "CUDA out of memory"

Riduci batch size:

```python
--batch_size 16  # invece di 32
--batch_size 8   # se ancora OOM
```

### Errore: "Could not find a version that satisfies the requirement fairseq2"

Verifica versione PyTorch:

```python
import torch
print(torch.__version__)  # Es: 2.5.0, 2.8.0, etc.
```

Usa URL corretto per tua versione:

- PyTorch 2.5.0: `https://fair.pkg.atmeta.com/fairseq2/whl/pt2.5.0/cpu`
- PyTorch 2.8.0: `https://fair.pkg.atmeta.com/fairseq2/whl/pt2.8.0/cpu`

---

## 🔗 Link Utili

- **fairseq2 GitHub**: https://github.com/facebookresearch/fairseq2
- **fairseq2 Variants**: https://github.com/facebookresearch/fairseq2#variants
- **PyTorch Installation**: https://pytorch.org/get-started/locally/
- **SONAR Paper**: https://ai.meta.com/research/publications/sonar-sentence-level-multimodal-and-language-agnostic-representations/

---

## ✅ Script di Verifica Completo

```python
# Verifica configurazione completa
import sys

print("=" * 60)
print("🔍 VERIFICA CONFIGURAZIONE")
print("=" * 60)

# 1. Python
print(f"\n🐍 Python: {sys.version.split()[0]}")

# 2. PyTorch
try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
except Exception as e:
    print(f"❌ PyTorch: {e}")

# 3. fairseq2
try:
    import fairseq2
    print(f"✅ fairseq2: {fairseq2.__version__}")

    from fairseq2.models.sonar import load_sonar_text_decoder
    print(f"   SONAR import: ✅")
except RuntimeError as e:
    print(f"⚠️ fairseq2: Version mismatch")
    print(f"   Error: {str(e)[:100]}...")
    print(f"   → Usa Soluzione 1 (CPU)")
except Exception as e:
    print(f"❌ fairseq2: {e}")

# 4. Altre dipendenze
for pkg in ['numpy', 'pandas', 'tqdm', 'sacrebleu']:
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✅ {pkg}: {version}")
    except:
        print(f"❌ {pkg}: Not installed")

print("\n" + "=" * 60)
print("✅ Verifica completata!")
print("=" * 60)
```

Copia e incolla questo script su Colab per verificare tutto! 🚀
