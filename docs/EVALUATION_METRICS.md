# Metriche di Valutazione Sign-to-Text

## 📊 Panoramica

Per valutare il modello Sign-to-Text usiamo diverse metriche a seconda della fase:

---

## 🔍 1. METRICHE DURANTE TUNING (Optuna)

### Metrica principale: **Validation Loss**

```python
criterion = nn.CrossEntropyLoss(
    ignore_index=pad_token_id,
    label_smoothing=0.1
)
```

**Perché solo val_loss durante tuning?**

- ✅ Veloce da calcolare (no decoding necessario)
- ✅ Differenziabile (supporta gradient-based optimization)
- ✅ Correlata con BLEU/WER (loss bassa → predictions migliori)
- ✅ Permette pruning veloce di trial cattivi

**Limitazioni**:

- ❌ Non cattura quality linguistica (grammatica, semantica)
- ❌ Non direttamente interpretabile dall'utente

---

## 📝 2. METRICHE POST-TRAINING (Evaluation)

### A. **BLEU Score** (Bilingual Evaluation Understudy)

Misura sovrapposizione n-gram tra predicted e reference caption.

```python
from torchmetrics.text import BLEUScore

bleu_1 = BLEUScore(n_gram=1)  # Unigram overlap
bleu_2 = BLEUScore(n_gram=2)  # Bigram overlap
bleu_3 = BLEUScore(n_gram=3)  # Trigram overlap
bleu_4 = BLEUScore(n_gram=4)  # 4-gram overlap (standard)

# Esempio
predicted = "the cat sits on the mat"
reference = "the cat is sitting on the mat"

bleu_4_score = bleu_4([predicted], [[reference]])
# Output: ~0.45 (45% overlap)
```

**Interpretazione**:

- `0.0 - 0.10`: Poor (inaccettabile)
- `0.10 - 0.20`: Fair (baseline)
- `0.20 - 0.30`: Good (pubblicabile)
- `0.30 - 0.40`: Very Good (state-of-the-art molti domini)
- `> 0.40`: Excellent (raro in sign-to-text)

**Pro**:

- ✅ Standard in machine translation
- ✅ Facile da calcolare
- ✅ Comparabile con letteratura

**Contro**:

- ❌ Non cattura significato semantico
- ❌ Penalizza synonyms/paraphrasi
- ❌ Biased verso caption lunghe

---

### B. **METEOR** (Metric for Evaluation of Translation with Explicit ORdering)

Migliora BLEU considerando:

- Synonyms (WordNet)
- Stemming (eat/eating)
- Word order

```python
from torchmetrics.text import METEORScore

meteor = METEORScore()
score = meteor([predicted], [[reference]])
# Output: 0.0-1.0 (higher is better)
```

**Correlazione migliore** con human judgment rispetto a BLEU.

---

### C. **WER** (Word Error Rate)

Usato in speech recognition. Misura edit distance a livello di parole.

```python
from jiwer import wer

reference = "the cat sits on the mat"
hypothesis = "the dog sits on a mat"

error_rate = wer(reference, hypothesis)
# Output: 0.33 (33% delle parole sbagliate)
```

**Formula**:

```
WER = (S + D + I) / N
```

- S = Substitutions (dog ← cat)
- D = Deletions (the → ∅)
- I = Insertions (∅ → a)
- N = Total reference words

**Interpretazione**:

- `< 0.10`: Excellent
- `0.10 - 0.20`: Good
- `0.20 - 0.30`: Acceptable
- `> 0.30`: Poor

---

### D. **CER** (Character Error Rate)

Come WER ma a livello di caratteri (più granulare).

```python
from jiwer import cer

cer_score = cer(reference, hypothesis)
# Output: 0.0-1.0 (lower is better)
```

**Utile per**:

- Detecting spelling errors
- Languages con morfologia complessa
- Short captions

---

### E. **ROUGE Score** (Recall-Oriented Understudy for Gisting Evaluation)

Usato in summarization. Misura recall invece di precision.

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
scores = scorer.score(reference, predicted)

# Output:
# rouge1: precision=0.8, recall=0.7, fmeasure=0.75
# rouge2: ...
# rougeL: longest common subsequence
```

---

### F. **Perplexity**

Misura quanto il modello è "sorpreso" dalla caption.

```python
perplexity = torch.exp(val_loss)
```

**Interpretazione**:

- Lower is better
- `< 10`: Excellent
- `10 - 50`: Good
- `> 50`: Poor

---

## 🎯 3. METRICHE CONSIGLIATE PER IL TUO PROGETTO

### Durante Tuning Optuna:

```python
# Objective function
return val_loss  # CrossEntropyLoss
```

### Evaluation Script (dopo training):

```python
metrics = {
    # Primary metrics (standard in Sign-to-Text papers)
    'BLEU-1': bleu_1_score,
    'BLEU-4': bleu_4_score,
    'WER': wer_score,

    # Secondary metrics (per analisi approfondita)
    'METEOR': meteor_score,
    'CER': cer_score,
    'Perplexity': perplexity,

    # Custom metrics (per tesi)
    'Vocab Coverage': vocab_coverage,  # % parole vocab usate
    'Avg Caption Length': avg_length,
    'Exact Match': exact_match_rate,   # % predictions identiche a reference
}
```

---

## 📈 4. BENCHMARK ATTESI (How2Sign)

### Baseline (Random/Naive):

- BLEU-4: ~0.00 - 0.02
- WER: ~0.80 - 1.00
- CER: ~0.70 - 0.90

### Dopo Tuning (tuo modello):

- BLEU-4: **0.10 - 0.20** (target)
- WER: **0.40 - 0.60**
- CER: **0.30 - 0.50**
- Perplexity: **< 30**

### State-of-the-art (letteratura):

- BLEU-4: 0.20 - 0.35 (How2Sign paper)
- WER: 0.30 - 0.45
- Con pre-training + fine-tuning

---

## 🔧 5. IMPLEMENTAZIONE

### Durante Training (opzionale - rallenta):

```python
# In validation loop
if epoch % 5 == 0:  # Ogni 5 epochs
    # Generate predictions
    predictions = model.generate(val_batch, beam_width=4)
    references = val_batch['captions']

    # Compute BLEU
    bleu4 = compute_bleu(predictions, references)

    print(f"BLEU-4: {bleu4:.4f}")
```

### Evaluation Script (dopo training):

```python
# src/sign_to_text/evaluate_how2sign.py

from torchmetrics.text import BLEUScore
from jiwer import wer, cer

# Load model
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Evaluate
all_predictions = []
all_references = []

for batch in test_loader:
    predictions = model.generate(batch, beam_width=4)
    all_predictions.extend(predictions)
    all_references.extend(batch['caption_texts'])

# Compute metrics
bleu1 = BLEUScore(n_gram=1)(all_predictions, all_references)
bleu4 = BLEUScore(n_gram=4)(all_predictions, all_references)
wer_score = wer(all_references, all_predictions)
cer_score = cer(all_references, all_predictions)

print(f"BLEU-1: {bleu1:.4f}")
print(f"BLEU-4: {bleu4:.4f}")
print(f"WER: {wer_score:.4f}")
print(f"CER: {cer_score:.4f}")
```

---

## 📊 6. METRICHE PER TESI

### Tabella Risultati (esempio):

| Metric     | Baseline | After Tuning | Improvement |
| ---------- | -------- | ------------ | ----------- |
| BLEU-1     | 0.05     | 0.28         | +460%       |
| BLEU-4     | 0.00     | 0.15         | +∞          |
| WER        | 0.85     | 0.45         | -47%        |
| CER        | 0.72     | 0.38         | -47%        |
| Perplexity | 145      | 24           | -83%        |

### Grafici Consigliati:

1. **Training curves**: Loss vs. Epochs
2. **BLEU progression**: BLEU vs. Epochs
3. **Hyperparameter importance**: Bar chart (da Optuna)
4. **Error analysis**: Confusion matrix parole comuni
5. **Length distribution**: Predicted vs. Reference caption lengths

---

## 🎯 RACCOMANDAZIONE FINALE

### Per il tuning (Optuna):

✅ **Usa solo val_loss** (veloce, efficiente, permette pruning)

### Per evaluation finale:

✅ **Calcola tutte le metriche**:

- BLEU-1, BLEU-4 (primary)
- WER, CER (error analysis)
- METEOR, Perplexity (secondary)

### Per la tesi:

✅ **Report completo**:

- Tabella con tutte le metriche
- Confronto con baseline ASLLRP
- Confronto con state-of-the-art (se disponibile)
- Qualitative analysis (esempi predictions)

---

## 📚 REFERENCES

- **BLEU**: Papineni et al. (2002) - "BLEU: a Method for Automatic Evaluation of Machine Translation"
- **METEOR**: Banerjee & Lavie (2005) - "METEOR: An Automatic Metric for MT Evaluation"
- **WER**: Standard in ASR (Automatic Speech Recognition)
- **How2Sign**: Duarte et al. (2021) - "How2Sign: A Large-scale Multimodal Dataset for Continuous American Sign Language"

---

**In sintesi**: Durante il tuning usa **val_loss** per velocità, poi calcola **BLEU-4 + WER** per evaluation finale e tesi! 🎯
