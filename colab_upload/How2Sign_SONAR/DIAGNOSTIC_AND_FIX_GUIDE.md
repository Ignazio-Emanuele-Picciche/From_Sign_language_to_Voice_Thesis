# 🔬 Guida Diagnostica e Fix per SONAR Training

## 📋 Problema Iniziale

**Sintomi:**

- Training loss iniziale molto bassa (0.0009) → sospetto
- BLEU Score quasi zero (0.02%) dopo 5 epochs
- Quick test troppo piccolo (100 samples) per valutare correttamente

**Domanda:** Il problema è architetturale o è solo il dataset di test troppo piccolo?

## 🎯 Strategia di Diagnosi

Invece di aspettare 50 epochs (2-4 ore) per scoprire che c'è un problema fondamentale, abbiamo implementato:

### 1️⃣ **Analisi Embeddings Diagnostica**

**Script:** Cella "🔬 ANALISI EMBEDDINGS ENCODER" nel notebook

**Cosa analizza:**

- **Norme L2** degli embeddings ASL vs SONAR
- **Diversità** (rileva collapse: tutti gli embeddings uguali)
- **Mapping** (quanto ASL embeddings sono vicini a SONAR space)
- **Decoder compatibility** (test su sample reali)

**Output:**

```
📊 STATISTICHE EMBEDDINGS
   Norme L2: mean, std, min, max

🔍 COLLAPSE DETECTION
   Cosine similarity off-diagonal:
   - > 0.95 → ❌ COLLAPSE!
   - > 0.80 → ⚠️ WARNING
   - < 0.80 → ✅ OK

🎯 CONFRONTO CON SONAR
   ASL vs SONAR norms
   Cosine similarity ASL→SONAR

🧪 TEST DECODER
   5 sample predictions
```

### 2️⃣ **Fix Automatici**

**Script:** `apply_training_fixes.py` o cella notebook

**Fix applicati:**

#### Fix 1: Normalizzazione L2 Output Encoder

```python
# PRIMA (potenziale problema di scala)
return features_avg

# DOPO (embeddings normalizzati)
features_avg = torch.nn.functional.normalize(features_avg, p=2, dim=1)
return features_avg
```

**Beneficio:** Embeddings con norma = 1.0, compatibili con SONAR space

---

#### Fix 2: Cosine Loss invece di MSE

```python
# PRIMA (MSE loss)
loss = torch.nn.functional.mse_loss(embeddings, target_embeddings)

# DOPO (Cosine loss)
target_norm = torch.nn.functional.normalize(target_embeddings, p=2, dim=1)
cosine_sim = (embeddings * target_norm).sum(dim=1).mean()
loss = 1.0 - cosine_sim  # Range [0, 2], ottimo = 0
```

**Benefici:**

- Loss interpretabile (0 = perfetto, 2 = opposti)
- Migliore per embeddings normalizzati
- Più stabile con gradienti

---

#### Fix 3: Gradient Monitoring

```python
# Calcola norma gradiente
total_norm = 0.0
for p in self.encoder.parameters():
    if p.grad is not None:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
total_norm = total_norm ** 0.5
```

**Beneficio:** Rileva problemi in tempo reale:

- `total_norm ≈ 0` → **collapse** (encoder non impara)
- `total_norm > 10` → **esplosione** (instabilità)
- `total_norm ∈ [0.1, 1.0]` → **OK**

---

#### Fix 4: Logging Avanzato

```python
pbar.set_postfix({
    "loss": f"{loss.item():.4f}",
    "grad_norm": f"{total_norm:.4f}",
    "cosine_sim": f"{cosine_sim.item():.4f}"
})
```

**Beneficio:** Vedi subito se qualcosa va storto durante training

---

#### Fix 5: Validation Metrics Estese

```python
log_entry = {
    "epoch": epoch,
    "train_loss": avg_loss,
    "val_bleu": bleu_score,
    "val_cosine_sim": 1.0 - avg_loss,  # NUOVO
}
```

**Beneficio:** Traccia anche la similarity (complementare a BLEU)

## 📊 Aspettative dopo Fix

### Script Originale (MSE Loss)

```
Epoch 1: Loss=0.0009 (sospetto!)
Epoch 5: Loss=0.0002
BLEU: 0.02%
```

### Script Migliorato (Cosine Loss + Normalizzazione)

```
Epoch 1: Loss≈1.0 (ragionevole, range 0-2)
         Cosine Sim≈0.0 (basso, normale all'inizio)
         Grad Norm≈0.5 (OK, non collapse)

Epoch 5: Loss≈0.5 (miglioramento!)
         Cosine Sim≈0.5 (embeddings più vicini)
         Grad Norm≈0.3 (stabile)

BLEU: > 1% (dovrebbe essere maggiore, anche se quick test)
```

## 🚀 Workflow Completo

### Step 1: Diagnostica

```python
# Nel notebook, esegui cella:
"🔬 ANALISI EMBEDDINGS ENCODER"
```

**Output atteso:**

- Statistiche embeddings
- Rilevamento problemi (collapse, scala, mapping)
- Riepilogo con soluzioni proposte

### Step 2: Applicazione Fix

```python
# Metodo A: Script standalone (consigliato)
!python apply_training_fixes.py

# Metodo B: Cella notebook
# Esegui cella "🔧 Fix Automatico"
```

**Output atteso:**

- Backup: `train_sonar_finetuning_BACKUP_<timestamp>.py`
- Improved: `train_sonar_finetuning_IMPROVED.py`
- Messaggio: "✅ FIX APPLICATI: 5/5"

### Step 3: Quick Test Improved

```python
!python train_sonar_finetuning_IMPROVED.py \
    --epochs 5 --max_samples 50 \
    --output_dir checkpoints/test_improved
```

**Output atteso:**

```
Epoch 1/5: Loss=1.0234 | grad_norm=0.4521 | cosine_sim=0.0123
Epoch 2/5: Loss=0.8123 | grad_norm=0.3891 | cosine_sim=0.1877
...
Epoch 5/5: Loss=0.5234 | grad_norm=0.2156 | cosine_sim=0.4766

Validation BLEU: 2.34% (miglioramento!)
```

### Step 4: Confronto

```python
# Cella automatica nel notebook confronta:
VECCHIO vs NUOVO
   Loss iniziale    | 0.0009  | 1.0234  | +1.0225
   Loss finale      | 0.0002  | 0.5234  | +0.5232
   BLEU finale      | 0.02%   | 2.34%   | +2.32%
```

### Step 5: Full Training (se migliora)

```python
# Sostituisci script originale
mv train_sonar_finetuning.py train_sonar_finetuning_OLD.py
mv train_sonar_finetuning_IMPROVED.py train_sonar_finetuning.py

# Rilancia full training
!python train_sonar_finetuning.py \
    --epochs 50 \
    --output_dir checkpoints/sonar_full_finetuned
```

## 🔍 Interpretazione Risultati Diagnostica

### Scenario 1: COLLAPSE Rilevato

```
📊 Cosine Similarity off-diagonal: 0.98
❌ PROBLEMA: Embeddings troppo simili!
```

**Causa:** Encoder produce output quasi identici per tutti i sample

**Soluzione:**

- ✅ Fix 1 (Normalizzazione) + Fix 2 (Cosine Loss)
- Aumenta learning rate (1e-3)
- Riduci batch size (8)

### Scenario 2: SCALA Sbagliata

```
📊 CONFRONTO NORME:
   Encoder ASL: 50.23 ± 12.4
   SONAR Text:  1.02 ± 0.15
   Differenza:  49.21  ❌ MOLTO DIVERSA!
```

**Causa:** Encoder produce embeddings con scala molto diversa da SONAR

**Soluzione:**

- ✅ Fix 1 (Normalizzazione L2) → forza norma = 1.0

### Scenario 3: NO MAPPING

```
🎯 Cosine Similarity ASL→SONAR: 0.12
❌ PROBLEMA: Embeddings ASL molto diversi da SONAR!
```

**Causa:** Encoder non sta imparando a mappare nello spazio SONAR

**Soluzione:**

- ✅ Fix 2 (Cosine Loss) → ottimizza direttamente similarity
- Aumenta epochs (100+)
- Verifica che target embeddings siano corretti

### Scenario 4: TUTTO OK

```
📊 Norme simili (diff < 5)
🔍 Diversità OK (cosine sim < 0.80)
🎯 Mapping OK (ASL→SONAR sim > 0.5)
✅ NESSUN PROBLEMA CRITICO!
```

**Conclusione:** BLEU basso probabilmente dovuto a:

- Quick test troppo piccolo (100 samples)
- Training troppo breve (5 epochs)

**Azione:** Aspetta full training (50 epochs)!

## 📁 Files Generati

```
How2Sign_SONAR/
├── train_sonar_finetuning.py              # Script originale
├── train_sonar_finetuning_BACKUP_*.py     # Backup automatico
├── train_sonar_finetuning_IMPROVED.py     # Script con fix
├── apply_training_fixes.py                # Script di fix standalone
├── DIAGNOSTIC_AND_FIX_GUIDE.md           # Questa guida
└── checkpoints/
    ├── sonar_full_test/                   # Quick test originale
    │   ├── best_model.pt
    │   └── training_log.json
    └── sonar_test_improved/               # Quick test improved
        ├── best_model.pt
        └── training_log.json
```

## 🎯 Checklist Finale

Prima del Full Training, verifica:

- [ ] Diagnostica eseguita su checkpoint quick test
- [ ] Problemi identificati (collapse, scala, mapping)
- [ ] Fix applicati (5/5 fix)
- [ ] Quick test improved completato
- [ ] BLEU improved > BLEU originale
- [ ] Loss range corretto (~0.5-1.5 con cosine loss)
- [ ] Gradient norm stabile (0.1-1.0)
- [ ] Script sostituito (`_IMPROVED.py` → `.py`)

**Se tutti i check ✅ → Procedi con Full Training (50 epochs)!**

## 📊 Target BLEU Atteso

| Dataset                                 | BLEU Atteso | Note              |
| --------------------------------------- | ----------- | ----------------- |
| Quick Test (50 samples, 5 epochs)       | 2-5%        | Baseline post-fix |
| Quick Test (100 samples, 5 epochs)      | 1-3%        | Troppo piccolo    |
| Full Training (1252 samples, 50 epochs) | **30-40%**  | Target finale     |

**Se Full Training < 5% BLEU:** Problema più profondo (features corrotte, decoder incompatibile)

## 💡 Troubleshooting

### Fix non applicati (0/5 o parziali)

**Causa:** Script già modificato o pattern non corrisponde

**Soluzione:**

1. Usa backup originale
2. Scarica script fresh da repository
3. Verifica encoding UTF-8

### Quick Test Improved peggiora

**Causa possibile:** Learning rate troppo alto per cosine loss

**Soluzione:**

```python
--learning_rate 5e-5  # Riduci da 1e-4
```

### Gradient norm = 0 anche dopo fix

**Causa:** Encoder frozen o optimizer non configurato

**Soluzione:** Verifica che `self.encoder.requires_grad = True`

---

**Autore:** Ignazio Picciche  
**Data:** Novembre 2024  
**Versione:** 1.0
