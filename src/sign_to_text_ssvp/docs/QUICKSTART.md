# 🚀 SSVP-SLT Quick Start Guide

Guida rapida per iniziare con SSVP-SLT sul dataset How2Sign.

---

## ⚡ Quick Setup (5 minuti)

### 1. Installazione SSVP-SLT

```bash
cd src/sign_to_text_ssvp

# Installa SSVP-SLT e dipendenze
bash scripts/install_ssvp.sh
```

**Output atteso**:

```
✅ SSVP-SLT Installation Complete!
```

---

### 2. Download Modello Pretrained

```bash
# Download Base model (~340 MB)
python download_pretrained.py --model base

# O Large model per performance massime (~1.2 GB)
# python download_pretrained.py --model large
```

**Output atteso**:

```
✅ Successfully downloaded: ssvp_base.pt
   Size: 340.2 MB
```

---

### 3. Preparazione Dataset How2Sign

#### Opzione A: Prepara tutti gli splits

```bash
cd scripts
bash prepare_all_splits.sh
```

#### Opzione B: Prepara split singolo

```bash
# Train split
python prepare_how2sign_for_ssvp.py \
    --input_csv ../../results/how2sign_splits/train_split.csv \
    --video_dir ../../data/raw/how2sign/videos \
    --output_dir ../../data/how2sign_ssvp \
    --split train

# Val split
python prepare_how2sign_for_ssvp.py \
    --input_csv ../../results/how2sign_splits/val_split.csv \
    --video_dir ../../data/raw/how2sign/videos \
    --output_dir ../../data/how2sign_ssvp \
    --split val
```

**Verifica**:

```bash
ls -lh ../../data/how2sign_ssvp/clips/train/  # Video files
ls -lh ../../data/how2sign_ssvp/manifest/     # TSV manifests
```

---

### 4. Quick Test Fine-tuning (3 epochs, 1000 samples)

```bash
# Test rapido per verificare che tutto funzioni
python finetune_how2sign.py \
    --config configs/finetune_quick.yaml \
    --device mps  # o cuda/cpu
```

**Tempo stimato**: ~20-30 minuti su M1/M2

---

### 5. Full Fine-tuning (30 epochs, dataset completo)

```bash
# Fine-tuning completo
python finetune_how2sign.py \
    --config configs/finetune_base.yaml \
    --device cuda  # o mps per M1/M2

# Background execution
nohup python finetune_how2sign.py \
    --config configs/finetune_base.yaml \
    --device cuda \
    > logs/finetune_base.log 2>&1 &
```

**Tempo stimato**:

- M1/M2 MPS: ~15-20 ore
- V100 (16GB): ~12 ore
- A100 (40GB): ~6 ore

---

### 6. Evaluation

```bash
# Valuta su validation set
python evaluate_how2sign.py \
    --checkpoint results/ssvp_finetune_base/best_checkpoint.pt \
    --split val \
    --output results/evaluation_val.json

# Valuta su test set
python evaluate_how2sign.py \
    --checkpoint results/ssvp_finetune_base/best_checkpoint.pt \
    --split test \
    --output results/evaluation_test.json
```

---

### 7. Comparazione con Seq2Seq Transformer

```bash
# Confronta SSVP-SLT vs tuo Seq2Seq
python compare_models.py \
    --ssvp_checkpoint results/ssvp_finetune_base/best_checkpoint.pt \
    --seq2seq_checkpoint ../sign_to_text/models/sign_to_text/how2sign/best_checkpoint.pt \
    --split val \
    --output results/model_comparison.json
```

---

## 📊 Performance Attese

### SSVP-SLT Base (dopo 30 epochs)

| Metric | Expected Value | Status    |
| ------ | -------------- | --------- |
| BLEU-4 | 38-40%         | 🎯 Target |
| BLEU-1 | 52-55%         | 🎯        |
| WER    | 25-30%         | 🎯        |
| CER    | 12-16%         | 🎯        |

### Comparazione

| Model         | BLEU-4        | WER           | Inference Speed |
| ------------- | ------------- | ------------- | --------------- |
| **SSVP-SLT**  | **38-40%** ✅ | **25-30%** ✅ | 12 fps          |
| Seq2Seq (tuo) | 25-30%        | 40-50%        | 35 fps ⚡       |

---

## 🔧 Troubleshooting

### Problema: "CUDA out of memory"

**Soluzione**:

```yaml
# Riduci batch_size in config
training:
  batch_size: 8 # era 16
  accumulation_steps: 2 # compensa batch size ridotto
```

### Problema: "ffmpeg not found"

**Soluzione**:

```bash
# macOS
brew install ffmpeg

# Linux
sudo apt-get install ffmpeg
```

### Problema: "Video file not found"

**Soluzione**:

```bash
# Verifica percorso video
ls -lh data/raw/how2sign/videos/ | head

# Usa symlink invece di copy (più veloce)
python prepare_how2sign_for_ssvp.py \
    --input_csv ... \
    --split train
    # NON usare --copy
```

---

## 📈 Monitoring Training

### TensorBoard

```bash
# Avvia TensorBoard
tensorboard --logdir results/ssvp_finetune_base/tensorboard

# Apri browser
open http://localhost:6006
```

### Training Logs

```bash
# Segui logs in tempo reale
tail -f results/ssvp_finetune_base/train.log

# Cerca errori
grep ERROR results/ssvp_finetune_base/train.log
```

---

## 🎯 Workflow Completo

```
1. Setup (una tantum)
   ├─ install_ssvp.sh
   ├─ download_pretrained.py
   └─ prepare_all_splits.sh

2. Fine-tuning
   ├─ Quick test (finetune_quick.yaml)
   └─ Full training (finetune_base.yaml)

3. Evaluation
   ├─ evaluate_how2sign.py (val)
   ├─ evaluate_how2sign.py (test)
   └─ compare_models.py

4. Integration
   └─ Use in EmoSign pipeline
```

---

## 📚 Next Steps

1. ✅ **Setup completato** → Procedi con quick test
2. ✅ **Quick test OK** → Lancia full fine-tuning
3. ✅ **Fine-tuning completo** → Evalua su val/test
4. ✅ **Evaluation OK** → Confronta con Seq2Seq
5. ✅ **Comparazione** → Integra in pipeline EmoSign

---

## 💡 Tips

### Per Training Veloce

- Usa `finetune_quick.yaml` con subset dati
- Testa su M1/M2 MPS prima di GPU remota
- Monitora loss: dovrebbe scendere sotto 2.0 in poche epoche

### Per Performance Massime

- Usa `finetune_large.yaml` con A100
- Abilita mixed precision (fp16)
- Usa data augmentation

### Per Debugging

- Aggiungi `--max_samples 100` per test rapidi
- Controlla log ogni 10 steps
- Salva checkpoint ogni epoca

---

## 📞 Support

- **SSVP-SLT Issues**: https://github.com/facebookresearch/ssvp_slt/issues
- **Paper**: https://arxiv.org/abs/2402.09611
- **Documentazione completa**: `src/sign_to_text_ssvp/README.md`
