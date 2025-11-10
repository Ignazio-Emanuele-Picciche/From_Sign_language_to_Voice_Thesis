# 🎯 SSVP-SLT Integration Guide

Guida all'integrazione del modello SSVP-SLT di Facebook Research nella pipeline EmoSign.

---

## 📌 Overview

Abbiamo integrato **SSVP-SLT** (Self-Supervised Video Pretraining for Sign Language Translation) come modello alternativo/complementare al Seq2Seq Transformer per la traduzione da ASL a testo.

### Perché SSVP-SLT?

| Caratteristica       | Seq2Seq Transformer (Nostro) | SSVP-SLT (Facebook)      |
| -------------------- | ---------------------------- | ------------------------ |
| **Approccio**        | Landmarks-based              | Video-based (end-to-end) |
| **Pretraining**      | ❌ From scratch              | ✅ Self-supervised (MAE) |
| **BLEU-4**           | ~25-30% (target)             | **38-40%** (SOTA)        |
| **Robustezza**       | Dipende da landmarks         | Più robusto              |
| **Interpretabilità** | ✅ High (pose/hands/face)    | ⚠️ Black-box             |
| **Efficienza**       | ✅ Più veloce (35 fps)       | ⚠️ Più lento (12 fps)    |
| **Memoria**          | ✅ 4GB VRAM                  | ❌ 8-16GB VRAM           |

---

## 📂 Struttura Integrazione

```
src/sign_to_text_ssvp/           # Nuovo modulo SSVP-SLT
├── README.md                     # Documentazione completa
├── __init__.py
│
├── configs/                      # Configurazioni fine-tuning
│   ├── finetune_quick.yaml      # Test rapido (3 epochs)
│   ├── finetune_base.yaml       # Full training (30 epochs)
│   └── finetune_large.yaml      # Large model (max performance)
│
├── scripts/                      # Script helper
│   ├── install_ssvp.sh          # Installazione automatica
│   └── prepare_all_splits.sh    # Preparazione dataset
│
├── models/                       # Modelli e checkpoints
│   ├── checkpoints/             # Pretrained models
│   └── ssvp_slt_repo/           # Clone repository SSVP-SLT
│
├── docs/                         # Documentazione
│   └── QUICKSTART.md            # Quick start guide
│
└── Python scripts:
    ├── download_pretrained.py    # Download modelli pretrained
    ├── prepare_how2sign_for_ssvp.py  # Conversione dataset
    ├── finetune_how2sign.py      # Fine-tuning (placeholder)
    ├── evaluate_how2sign.py      # Evaluation (placeholder)
    └── compare_models.py         # Comparazione modelli (placeholder)
```

---

## 🚀 Quick Start

### 1. Installazione (5 minuti)

```bash
cd src/sign_to_text_ssvp
bash scripts/install_ssvp.sh
```

### 2. Download Modello Pretrained

```bash
python download_pretrained.py --model base
```

### 3. Preparazione Dataset

```bash
bash scripts/prepare_all_splits.sh
```

### 4. Fine-tuning

```bash
# Quick test
python finetune_how2sign.py --config configs/finetune_quick.yaml

# Full training
python finetune_how2sign.py --config configs/finetune_base.yaml
```

**Nota**: Gli script Python di fine-tuning, evaluation e comparison sono placeholder che richiedono implementazione dopo installazione SSVP-SLT.

---

## 🎯 Use Cases

### 1. **Benchmark State-of-the-Art**

Usa SSVP-SLT per stabilire upper bound performance:

```python
# Valuta entrambi i modelli
python src/sign_to_text_ssvp/evaluate_how2sign.py --checkpoint ... --split test
python src/sign_to_text/evaluate_how2sign.py --checkpoint ... --split test

# Confronta risultati
python src/sign_to_text_ssvp/compare_models.py --ssvp ... --seq2seq ...
```

### 2. **Production Deployment**

Scegli modello basato su trade-off:

```
Accuracy prioritaria → SSVP-SLT (40% BLEU)
Speed prioritaria → Seq2Seq (35 fps)
Interpretabilità → Seq2Seq (landmarks)
```

### 3. **Ensemble Model**

Combina predizioni per robustezza:

```python
# Weighted ensemble
final_prediction = 0.6 * ssvp_output + 0.4 * seq2seq_output
```

### 4. **Thesis Contribution**

Mostra nella tesi:

- ✅ Comparison landmarks-based vs video-based
- ✅ Trade-off accuracy vs efficiency
- ✅ Quando usare quale approccio
- ✅ Ablation study: effetto pretraining

---

## 📊 Performance Target

### Expected Results (dopo fine-tuning su How2Sign)

| Model             | BLEU-4     | BLEU-1 | WER    | CER    | Speed   |
| ----------------- | ---------- | ------ | ------ | ------ | ------- |
| **SSVP-SLT Base** | **38-40%** | 52-55% | 25-30% | 12-16% | 12 fps  |
| Seq2Seq (nostro)  | 25-30%     | 42-45% | 40-50% | 25-30% | 35 fps  |
| **Improvement**   | **+13%**   | +10%   | -15%   | -13%   | -23 fps |

---

## 🔄 Integrazione Pipeline EmoSign

### Pipeline Attuale

```
Video ASL → Landmarks (OpenPose) → Seq2Seq → Text → Emotion → TTS
```

### Pipeline con SSVP-SLT

```
Video ASL → SSVP-SLT → Text → Emotion → TTS
```

### Pipeline Ensemble (proposta)

```
                    ┌─ Landmarks → Seq2Seq ─┐
Video ASL ──┬──────┤                         ├─→ Text (ensemble) → Emotion → TTS
            └──────→ SSVP-SLT ───────────────┘
```

**Vantaggi Ensemble**:

- Robustezza a failure di un modello
- Combina interpretabilità + accuracy
- Fallback se landmarks extraction fails

---

## 📝 TODO List

### Implementazione Completa

- [x] Setup directory structure
- [x] Documentazione completa (README.md)
- [x] Script installazione (install_ssvp.sh)
- [x] Script download pretrained (download_pretrained.py)
- [x] Script preparazione dataset (prepare_how2sign_for_ssvp.py)
- [x] File configurazione YAML (3 configs)
- [x] Quick start guide (QUICKSTART.md)
- [ ] **Script fine-tuning** (finetune_how2sign.py) → Richiede studio API SSVP-SLT
- [ ] **Script evaluation** (evaluate_how2sign.py) → Dopo fine-tuning
- [ ] **Script comparison** (compare_models.py) → Dopo entrambi i modelli trained
- [ ] Test fine-tuning completo su How2Sign
- [ ] Evaluation e benchmark vs Seq2Seq
- [ ] Integrazione in pipeline EmoSign

### Ricerca e Tesi

- [ ] Ablation study: effect of pretraining
- [ ] Comparison landmarks vs video approach
- [ ] Error analysis: dove SSVP-SLT > Seq2Seq?
- [ ] Ensemble experiments
- [ ] Write thesis section on model comparison

---

## 🔧 Next Steps

### Step 1: Installazione e Setup (oggi)

```bash
cd src/sign_to_text_ssvp
bash scripts/install_ssvp.sh
python download_pretrained.py --model base
bash scripts/prepare_all_splits.sh
```

### Step 2: Implementazione Fine-tuning (1-2 giorni)

Studiare API SSVP-SLT e implementare `finetune_how2sign.py`:

```bash
# Riferimenti
ls -lh models/ssvp_slt_repo/translation/
cat models/ssvp_slt_repo/translation/README.md
```

### Step 3: Training e Evaluation (1 settimana)

```bash
# Quick test
python finetune_how2sign.py --config configs/finetune_quick.yaml

# Full training
python finetune_how2sign.py --config configs/finetune_base.yaml

# Evaluate
python evaluate_how2sign.py --checkpoint ... --split test
```

### Step 4: Comparison e Tesi (1 settimana)

```bash
# Compare models
python compare_models.py --ssvp ... --seq2seq ...

# Write thesis sections
# - Model comparison
# - Results analysis
# - Discussion
```

---

## 📚 References

1. **SSVP-SLT Paper**: [Rust et al. 2024 - ACL](https://aclanthology.org/2024.acl-long.467/)
2. **Repository**: [facebookresearch/ssvp_slt](https://github.com/facebookresearch/ssvp_slt)
3. **How2Sign Dataset**: [Duarte et al. 2021](https://how2sign.github.io/)
4. **Our Seq2Seq**: `src/sign_to_text/README.md`

---

## 💡 Tips

### Per Training Efficiente

- Usa `finetune_quick.yaml` per test rapidi
- Monitora loss: deve scendere sotto 2.0 rapidamente
- Salva checkpoint ogni 5 epoche

### Per Debugging

- Usa `--max_samples 100` per test veloci
- Controlla manifest TSV prima di training
- Verifica video paths con symlink

### Per Performance

- Usa modello Base su V100/A100
- Abilita mixed precision (fp16)
- Batch size 16 per Base, 8 per Large

---

## 📞 Support

- **Documentazione SSVP-SLT**: `src/sign_to_text_ssvp/README.md`
- **Quick Start**: `src/sign_to_text_ssvp/docs/QUICKSTART.md`
- **SSVP-SLT Issues**: https://github.com/facebookresearch/ssvp_slt/issues
- **Paper**: https://arxiv.org/abs/2402.09611
