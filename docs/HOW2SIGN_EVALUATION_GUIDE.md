# How2Sign Evaluation Guide

## 📊 Overview

Questo documento spiega come valutare il modello Sign-to-Text con metriche complete.

---

## 🎯 Workflow Completo

```
1. TUNING (opzionale)      → Trova migliori hyperparameters usando val_loss
2. TRAINING               → Train con best params, monitora val_loss
3. EVALUATION (questo!)   → Calcola BLEU/WER/CER/etc su best model
```

---

## 📝 Metriche Implementate

### Durante Tuning/Training:

- **Loss** (CrossEntropyLoss)
- **Perplexity** (exp(loss))

### Durante Evaluation (script creato ora):

1. **BLEU-1, BLEU-2, BLEU-3, BLEU-4** - N-gram overlap
2. **WER** (Word Error Rate) - Edit distance parole
3. **CER** (Character Error Rate) - Edit distance caratteri
4. **ROUGE** (optional) - Recall-oriented metric
5. **Perplexity** - Surprise del modello
6. **Vocab Coverage** - % vocabolario usato
7. **Caption Length Stats** - Media/std lunghezze
8. **Exact Match Rate** - % predictions identiche

---

## 🚀 Come Usare

### Opzione 1: Script Launcher (RACCOMANDATO)

```bash
# Evaluate on validation set
./run_how2sign_evaluation.sh

# Evaluate specific checkpoint on val
./run_how2sign_evaluation.sh models/sign_to_text/how2sign_tuned/best_checkpoint.pt val

# Evaluate on test set (golden labels)
./run_how2sign_evaluation.sh models/sign_to_text/how2sign/best_checkpoint.pt test
```

**Output:**

- `results/how2sign_evaluation/metrics_val.json` - Metriche numeriche
- `results/how2sign_evaluation/examples_val.txt` - Esempi qualitativi

---

### Opzione 2: Python Diretto

```bash
# Validation set
python src/sign_to_text/evaluate_how2sign.py \
    --checkpoint models/sign_to_text/how2sign/best_checkpoint.pt \
    --split val \
    --batch_size 8 \
    --num_examples 20 \
    --save_examples results/how2sign_evaluation/examples_val.txt \
    --save_metrics results/how2sign_evaluation/metrics_val.json

# Test set (golden labels)
python src/sign_to_text/evaluate_how2sign.py \
    --checkpoint models/sign_to_text/how2sign/best_checkpoint.pt \
    --split test \
    --test_csv data/processed/golden_label_sentiment.csv \
    --test_openpose_dir data/processed/landmarks \
    --num_examples 20
```

---

## 📊 Output Esempio

### Console Output:

```
================================================================================
📊 EVALUATION RESULTS
================================================================================

🎯 Primary Metrics:
   Loss:         2.4567
   Perplexity:   11.67

📝 BLEU Scores:
   BLEU-1:      0.2834
   BLEU-2:      0.1892
   BLEU-3:      0.1245
   BLEU-4:      0.0856

🔤 Error Rates:
   WER:          0.4523
   CER:          0.3412

📏 Caption Statistics:
   Avg Pred Len: 8.3 ± 2.1
   Avg Ref Len:  8.7 ± 2.5
   Exact Match:  3.2%

📚 Vocabulary:
   Unique Words: 1247
   Vocab Cover:  31.2%
================================================================================
```

### JSON Output (`metrics_val.json`):

```json
{
  "loss": 2.4567,
  "perplexity": 11.67,
  "bleu_1": 0.2834,
  "bleu_2": 0.1892,
  "bleu_3": 0.1245,
  "bleu_4": 0.0856,
  "wer": 0.4523,
  "cer": 0.3412,
  "rouge1_fmeasure": 0.3021,
  "rouge2_fmeasure": 0.1534,
  "rougeL_fmeasure": 0.2789,
  "vocab_coverage": 0.312,
  "unique_words_generated": 1247,
  "avg_pred_length": 8.3,
  "avg_ref_length": 8.7,
  "std_pred_length": 2.1,
  "std_ref_length": 2.5,
  "exact_match_rate": 0.032
}
```

### Examples File (`examples_val.txt`):

```
================================================================================
QUALITATIVE EXAMPLES
================================================================================

Example 1:
  Video:     video_00123
  Frames:    87
  Reference: the cat is sitting on the mat
  Predicted: the cat sits on a mat

Example 2:
  Video:     video_00456
  Frames:    134
  Reference: i went to the store yesterday
  Predicted: i go to store yesterday

...
```

---

## 🎯 Target Metrics (How2Sign)

### Baseline (Untrained/Random):

- BLEU-4: ~0.00 - 0.02
- WER: ~0.80 - 1.00
- Perplexity: >100

### Good Model (After Training):

- **BLEU-4: 0.10 - 0.20** ✅
- **WER: 0.40 - 0.60** ✅
- **Perplexity: 10 - 30** ✅
- BLEU-1: 0.25 - 0.35

### State-of-the-art (Paper):

- BLEU-4: 0.20 - 0.35
- WER: 0.30 - 0.45
- (Con pre-training + fine-tuning)

---

## 📈 Interpretazione Metriche

### BLEU (0.0 - 1.0, higher = better)

- **0.00 - 0.10**: Poor (inaccettabile)
- **0.10 - 0.20**: Fair (baseline accettabile)
- **0.20 - 0.30**: Good (pubblicabile)
- **0.30 - 0.40**: Very Good (state-of-the-art)
- **> 0.40**: Excellent (raro in sign-to-text)

### WER (0.0 - 1.0+, lower = better)

- **< 0.10**: Excellent
- **0.10 - 0.20**: Very Good
- **0.20 - 0.30**: Good
- **0.30 - 0.50**: Acceptable
- **> 0.50**: Poor

### Perplexity (>1, lower = better)

- **< 10**: Excellent (molto confidente)
- **10 - 30**: Good (buona calibrazione)
- **30 - 50**: Acceptable
- **> 50**: Poor (modello incerto)

---

## 🔧 Troubleshooting

### Error: "Checkpoint not found"

```bash
# Verifica che il checkpoint esista
ls -lh models/sign_to_text/how2sign/best_checkpoint.pt

# Oppure specifica path corretto
./run_how2sign_evaluation.sh models/sign_to_text/how2sign_tuned/best_checkpoint.pt
```

### Error: "BLEU skipped"

```bash
# Installa torchmetrics
pip install torchmetrics
```

### Error: "WER/CER skipped"

```bash
# Installa jiwer
pip install jiwer
```

### Evaluation troppo lenta

```bash
# Riduci batch size
python src/sign_to_text/evaluate_how2sign.py \
    --checkpoint ... \
    --batch_size 4  # invece di 8
    --num_workers 0  # no multiprocessing
```

---

## 📚 Per la Tesi

### Tabella Risultati (esempio):

| Metric     | Baseline | How2Sign | How2Sign Tuned | Improvement |
| ---------- | -------- | -------- | -------------- | ----------- |
| BLEU-1     | 0.05     | 0.28     | 0.31           | +520%       |
| BLEU-4     | 0.00     | 0.15     | 0.18           | +∞          |
| WER        | 0.85     | 0.45     | 0.42           | -51%        |
| CER        | 0.72     | 0.38     | 0.34           | -53%        |
| Perplexity | 145      | 24       | 19             | -87%        |

### Grafici Consigliati:

1. **Bar chart**: BLEU-1/2/3/4 comparison
2. **Error bars**: WER/CER con confidence intervals
3. **Length distribution**: Histogram predicted vs reference
4. **Qualitative examples**: Table con best/worst predictions
5. **Confusion matrix**: Parole più comuni (corrette/sbagliate)

---

## 🎯 Workflow Completo per Tesi

```bash
# 1. Train model
./run_how2sign_training.sh

# 2. Evaluate on validation
./run_how2sign_evaluation.sh models/sign_to_text/how2sign/best_checkpoint.pt val

# 3. Evaluate on test (golden labels)
./run_how2sign_evaluation.sh models/sign_to_text/how2sign/best_checkpoint.pt test

# 4. Compare results
python scripts/compare_metrics.py \
    results/how2sign_evaluation/metrics_val.json \
    results/how2sign_evaluation/metrics_test.json
```

---

## 🔍 Note Tecniche

### Greedy Decoding vs Beam Search

Attualmente usa **greedy decoding** (più veloce):

```python
next_token = logits[:, -1, :].argmax(dim=-1)
```

Per **beam search** (migliore quality ma +lento):

```python
# TODO: Implementare beam search con beam_width=4
# Migliora BLEU di ~2-3 punti ma 4x più lento
```

### Teacher Forcing vs Autoregressive

- **Loss computation**: Teacher forcing (usa ground truth)
- **Prediction generation**: Autoregressive (usa own predictions)

Questo è corretto per evaluation!

---

## ✅ Checklist Pre-Evaluation

- [ ] Checkpoint esiste e è completo
- [ ] Tokenizer disponibile in `models/sign_to_text/tokenizer.json`
- [ ] Dataset split esiste (train/val/test CSV)
- [ ] OpenPose landmarks estratti
- [ ] Dipendenze installate (`jiwer`, `torchmetrics`)
- [ ] Output directory esiste/è writable

---

## 📞 Quick Reference

```bash
# Evaluate validation set (default)
./run_how2sign_evaluation.sh

# Evaluate test set
./run_how2sign_evaluation.sh models/sign_to_text/how2sign/best_checkpoint.pt test

# Compare multiple checkpoints
for ckpt in models/sign_to_text/*/best_checkpoint.pt; do
    echo "Evaluating $ckpt"
    ./run_how2sign_evaluation.sh "$ckpt" val
done
```

---

**Pronto per valutare il modello! 🚀**

Dopo il training, lancia semplicemente:

```bash
./run_how2sign_evaluation.sh
```

E ottieni tutte le metriche automaticamente! 📊
