# Guida: Training con SONAR Encoder Unfrozen

## ✅ Modifiche Applicate

### 1. **Differential Learning Rates**

```python
SONAR Encoder:        LR = base_lr / 5  (es. 1e-5)  ← Preserva pre-training
Projection/Attention: LR = base_lr      (es. 5e-5)  ← Adatta a T5
T5:                   LR = base_lr      (es. 5e-5)  ← Normale
```

**Perché 1/5x per SONAR?**

- SONAR è già pre-trained su ASL con 50 epochs
- LR troppo alto potrebbe "rovinare" la specializzazione
- LR basso permette fine-tuning graduale verso T5

### 2. **Gradient Flow Controllato**

```python
if freeze_encoder:
    with torch.no_grad():
        sonar_embedding = encoder(features)  # No gradient
else:
    sonar_embedding = encoder(features)      # Compute gradient ✅
```

### 3. **Optimizer Migliorato**

```python
# Prima (tutto stesso LR):
optimizer = AdamW(model.parameters(), lr=5e-5)

# Dopo (differential LR):
optimizer = AdamW([
    {'params': encoder_params, 'lr': 1e-5},
    {'params': projection_params, 'lr': 5e-5},
    {'params': t5_params, 'lr': 5e-5}
])
```

---

## 🚀 Come Usare

### **Opzione A: Encoder FROZEN (come prima)**

```bash
python train_sonar_with_t5.py \
    --sonar_checkpoint checkpoints/sonar_encoder_finetuned/best_encoder.pt \
    --train_features features/train \
    --train_manifest manifests/train.tsv \
    --val_features features/val \
    --val_manifest manifests/val.tsv \
    --output_dir checkpoints/sonar_t5_frozen \
    --t5_model t5-small \
    --freeze_encoder \    ← AGGIUNTO: Freezes encoder
    --epochs 20 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --warmup_steps 500 \
    --device cuda
```

**Output:**

```
🔧 Optimizer: Single LR (encoder frozen)
   SONAR Encoder: FROZEN ❄️
   Trainable params: ~60M (solo T5 + projection)
```

**BLEU atteso:** 1-3% (attuale)

---

### **Opzione B: Encoder UNFROZEN** ⭐ (NUOVO!)

```bash
python train_sonar_with_t5.py \
    --sonar_checkpoint checkpoints/sonar_encoder_finetuned/best_encoder.pt \
    --train_features features/train \
    --train_manifest manifests/train.tsv \
    --val_features features/val \
    --val_manifest manifests/val.tsv \
    --output_dir checkpoints/sonar_t5_unfrozen \
    --t5_model t5-small \
    --epochs 30 \           ← Aumenta epochs (più parametri da ottimizzare)
    --batch_size 16 \
    --learning_rate 5e-5 \
    --warmup_steps 500 \
    --device cuda
```

**⚠️ IMPORTANTE:** **NON** aggiungere `--freeze_encoder`!

**Output:**

```
🔧 Optimizer: Differential Learning Rates
   SONAR Encoder LR: 1.00e-05 (1/5x, preserve pre-training)
   Projection/Attention LR: 5.00e-05 (1x)
   T5 LR: 5.00e-05 (1x)
   Trainable params:
     - SONAR Encoder: 6 (layers)
     - Projection/Attention: 4
     - T5: 220

🔥 SONAR Encoder TRAINABLE (will be fine-tuned further)
```

**BLEU atteso:** 5-12% (miglioramento significativo!)

---

## 📊 Confronto

| Metodo       | Params Trainable | LR SONAR | LR T5 | Epochs | BLEU Atteso | Training Time |
| ------------ | ---------------- | -------- | ----- | ------ | ----------- | ------------- |
| **Frozen**   | 60M (no encoder) | N/A      | 5e-5  | 20     | 1-3%        | 4h            |
| **Unfrozen** | 63M (full model) | 1e-5     | 5e-5  | 30     | **5-12%**   | 6h            |

---

## 🧪 Test Prima di Trainare

Verifica che tutto funzioni:

```bash
python test_unfrozen_training.py
```

**Output atteso:**

```
TEST 1: Encoder FROZEN (freeze_encoder=True)
✅ PASS: Encoder correctly frozen!

TEST 2: Encoder TRAINABLE (freeze_encoder=False)
✅ PASS: Encoder correctly unfrozen!

TEST 3: Forward Pass con Gradient
✅ Loss computed: 2.3456
✅ SONAR Encoder has gradients: True
✅ Projection has gradients: True
✅ PASS: Gradients computed correctly!

TEST 4: Differential Learning Rates
✅ Optimizer created with 3 param groups
   Group 0 (SONAR): LR=1.00e-05, 6 params
   Group 1 (Projection): LR=5.00e-05, 4 params
   Group 2 (T5): LR=5.00e-05, 220 params
✅ PASS: SONAR has 1/5 learning rate!
```

---

## ⚙️ Hyperparameters Raccomandati

### **Per Encoder Unfrozen:**

```bash
--learning_rate 5e-5    # Base LR (SONAR usa 1e-5 automaticamente)
--epochs 30             # Più epochs per convergenza
--batch_size 16         # Stesso
--warmup_steps 500      # Stesso
```

### **Se Vuoi Essere Più Conservativo:**

```bash
--learning_rate 3e-5    # LR più basso (SONAR = 6e-6)
--epochs 40             # Ancora più epochs
--warmup_steps 1000     # Più warmup
```

### **Se Vuoi Essere Più Aggressivo:**

```bash
--learning_rate 1e-4    # LR più alto (SONAR = 2e-5)
--epochs 25             # Meno epochs
--warmup_steps 300      # Meno warmup
```

---

## 📈 Monitoring Durante Training

Guarda questi indicatori:

### **Se SONAR sta imparando troppo velocemente:**

```
Epoch 1: Train Loss = 2.5 → 1.2  (troppo veloce!)
Epoch 2: Val BLEU = 0.5% → 0.3%  (sta dimenticando ASL!)
```

**Soluzione:** Riduci LR encoder a 1/10x invece di 1/5x

### **Se SONAR sta imparando troppo lentamente:**

```
Epoch 10: Train Loss = 2.1 → 2.0  (plateau)
Epoch 10: Val BLEU = 2.5% → 2.6%  (no improvement)
```

**Soluzione:** Aumenta LR encoder a 1/3x invece di 1/5x

### **Se sta funzionando bene:**

```
Epoch 1:  Train Loss = 2.5, Val BLEU = 1.0%
Epoch 5:  Train Loss = 2.0, Val BLEU = 3.5%
Epoch 10: Train Loss = 1.7, Val BLEU = 6.2%
Epoch 15: Train Loss = 1.5, Val BLEU = 8.5%
Epoch 20: Train Loss = 1.4, Val BLEU = 10.1%
```

**✅ Continua così!**

---

## 🎯 Next Steps

1. ✅ **Test rapido** (2 epochs):

   ```bash
   python train_sonar_with_t5.py \
       --sonar_checkpoint checkpoints/sonar_encoder_finetuned/best_encoder.pt \
       --train_features features/train \
       --train_manifest manifests/train.tsv \
       --val_features features/val \
       --val_manifest manifests/val.tsv \
       --output_dir checkpoints/sonar_t5_unfrozen_test \
       --epochs 2 \
       --batch_size 16 \
       --learning_rate 5e-5 \
       --device cuda
   ```

2. ✅ **Se BLEU >2% dopo 2 epochs** → Training completo 30 epochs

3. ✅ **Se BLEU <1% dopo 2 epochs** → Aumenta LR encoder a 2e-5

---

## ❓ FAQ

**Q: Quando usare frozen vs unfrozen?**

- **Frozen:** Se hai poco tempo (4h), o se dataset è piccolo (<500 samples)
- **Unfrozen:** Se hai tempo (6h), e dataset è decente (>1000 samples) ✅

**Q: SONAR perderà specializzazione ASL?**

No, perché:

1. LR molto basso (1/5x)
2. Pre-training forte (50 epochs)
3. Dataset How2Sign è ASL puro

**Q: Posso unfreezearlo dopo alcune epochs?**

Sì! Progressive unfreezing:

```bash
# Epochs 1-10: Frozen
# Epochs 11-30: Unfrozen
```

Ma il codice attuale non supporta questo. Richiede modifica.

---

## 🔬 Risultati Attesi

### **Timeline:**

```
Epoch 2:  BLEU ~2-3%   (prima era 0.5%)
Epoch 5:  BLEU ~4-6%   (prima era 1.2%)
Epoch 10: BLEU ~6-9%   (prima era 1.5%)
Epoch 20: BLEU ~8-11%  (prima era 1.66%)
Epoch 30: BLEU ~10-14% (target!)
```

### **Best Case:**

```
BLEU: 12-15%
Loss: 1.2-1.4
Samples: Traduzioni coerenti e specifiche
```

### **Worst Case:**

```
BLEU: 3-5%
Loss: 1.8-2.0
Samples: Ancora generiche ma migliori
```

---

## ✅ Checklist

Prima di lanciare training:

- [ ] Test script eseguito con successo
- [ ] Checkpoint SONAR presente in `checkpoints/sonar_encoder_finetuned/`
- [ ] Features presenti in `features/train/` e `features/val/`
- [ ] Manifests presenti in `manifests/train.tsv` e `manifests/val.tsv`
- [ ] GPU disponibile (controlla `nvidia-smi`)
- [ ] `--freeze_encoder` RIMOSSO dal comando
- [ ] Output directory non esiste o vuota

---

Buon training! 🚀
