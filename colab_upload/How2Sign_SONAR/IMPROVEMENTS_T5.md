# 🔧 Miglioramenti per BLEU T5

## Problema Attuale

- BLEU: 1.06% (troppo basso)
- Atteso: 18-25%
- Miglioramento vs LSTM: 106x (buono ma non sufficiente)

---

## 🎯 Modifiche Applicate

### **1. Sequence Expansion (CRITICO!)** ⭐⭐⭐

**Problema**: T5 riceveva solo 1 token (B, 1, 512)
**Fix**: Ora riceve 8 token (B, 8, 512) con learnable expander

```python
# Prima (MALE)
t5_embedding = embedding.unsqueeze(1)  # (B, 1, 512) ❌

# Dopo (BENE)
t5_embedding = embedding.unsqueeze(1).repeat(1, 8, 1)  # (B, 8, 512) ✅
t5_embedding = t5_embedding + learnable_expander  # Aggiungi variabilità
```

**Impatto atteso**: BLEU da 1% → 8-12%

---

### **2. Projection Layer Più Profonda** ⭐⭐

**Problema**: 1 layer troppo semplice (1024→512)
**Fix**: 2 layer con hidden (1024→768→512)

```python
# Prima
projection = Linear(1024, 512)  # ❌ Troppo semplice

# Dopo
projection = Sequential(
    Linear(1024, 768),
    ReLU(),
    Dropout(0.1),
    Linear(768, 512),
    LayerNorm(512)
)  # ✅ Più espressivo
```

**Impatto atteso**: +2-3% BLEU

---

### **3. Altre Ottimizzazioni da Provare** ⭐

#### **A. Learning Rate Più Basso**

```bash
--learning_rate 5e-5  # Invece di 1e-4
```

#### **B. Gradient Accumulation**

```python
# Simula batch size più grande
--batch_size 8 \
--gradient_accumulation_steps 2  # Effective batch = 16
```

#### **C. Label Smoothing**

```python
# In T5 forward
loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
```

#### **D. Più Epochs**

```bash
--epochs 20  # Invece di 10
```

---

## 🚀 Come Rilanciare Training Migliorato

### **Opzione 1: Quick Test (2 epochs)**

```bash
!python train_sonar_with_t5.py \
    --sonar_checkpoint checkpoints/sonar_encoder_finetuned/best_encoder.pt \
    --train_features features/train \
    --train_manifest manifests/train.tsv \
    --val_features features/val \
    --val_manifest manifests/val.tsv \
    --output_dir checkpoints/sonar_t5_v2 \
    --t5_model t5-small \
    --freeze_encoder \
    --epochs 2 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --warmup_steps 100 \
    --device cuda
```

**Se BLEU > 5% dopo 2 epochs → continua con training completo!**

---

### **Opzione 2: Training Completo Ottimizzato**

```bash
!python train_sonar_with_t5.py \
    --sonar_checkpoint checkpoints/sonar_encoder_finetuned/best_encoder.pt \
    --train_features features/train \
    --train_manifest manifests/train.tsv \
    --val_features features/val \
    --val_manifest manifests/val.tsv \
    --output_dir checkpoints/sonar_t5_optimized \
    --t5_model t5-small \
    --freeze_encoder \
    --epochs 20 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --warmup_steps 500 \
    --device cuda
```

**BLEU atteso: 12-18%** (più realistico con dataset piccolo)

---

## 📊 Risultati Attesi

### **Progressione BLEU Attesa**

| Epoch | BLEU (prima) | BLEU (dopo fix) | Miglioramento |
| ----- | ------------ | --------------- | ------------- |
| 1     | 0.5%         | 3-5%            | ✅ 6-10x      |
| 2     | 0.8%         | 6-8%            | ✅ 7-10x      |
| 5     | 1.0%         | 10-12%          | ✅ 10-12x     |
| 10    | 1.06%        | 14-16%          | ✅ 13-15x     |
| 20    | -            | 16-18%          | ✅ Target!    |

---

## 🔬 Debugging: Se BLEU Ancora Basso

### **Test 1: Verifica Sample Translations**

```python
# Durante validation, guarda cosa genera T5
print("\n📝 Sample Translations:")
for i in range(10):
    print(f"GT:   {references[i]}")
    print(f"Pred: {predictions[i]}")
    print()
```

**Cosa cercare:**

- ✅ **GOOD**: Pred contiene parole rilevanti (anche se ordine sbagliato)
- ⚠️ **MEH**: Pred contiene parole generiche ("the", "a", "is")
- ❌ **BAD**: Pred è vuoto o nonsense completo

---

### **Test 2: Verifica Embedding Quality**

```python
# Aggiungi prima del training loop
def test_embeddings(model, val_loader):
    """Test se embeddings hanno senso"""
    model.eval()

    embeddings = []
    texts = []

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)

            # Get SONAR embeddings
            sonar_emb = model.sonar_encoder(features)
            embeddings.extend(sonar_emb.cpu().numpy())
            texts.extend(batch["texts"])

    # Check similarity
    from sklearn.metrics.pairwise import cosine_similarity

    # Trova frasi simili
    similar_pairs = [
        ("hello", "hi"),
        ("goodbye", "bye"),
        ("thank you", "thanks")
    ]

    for word1, word2 in similar_pairs:
        # Trova indici
        idx1 = [i for i, t in enumerate(texts) if word1.lower() in t.lower()]
        idx2 = [i for i, t in enumerate(texts) if word2.lower() in t.lower()]

        if idx1 and idx2:
            emb1 = embeddings[idx1[0]]
            emb2 = embeddings[idx2[0]]
            sim = cosine_similarity([emb1], [emb2])[0][0]
            print(f"Similarity '{word1}' vs '{word2}': {sim:.3f}")
            # Expected: >0.7 se encoder è buono

# Chiama prima del training
test_embeddings(model, val_loader)
```

---

### **Test 3: Verifica T5 Decoder**

```python
# Test se T5 può generare testo inglese base
def test_t5_decoder(model):
    """Test T5 con embedding random"""
    model.eval()

    # Crea embedding random
    dummy_embedding = torch.randn(1, 8, 512).to(model.device)

    with torch.no_grad():
        generated_ids = model.t5.generate(
            inputs_embeds=dummy_embedding,
            max_length=20,
            num_beams=2
        )
        text = model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print(f"T5 output with random input: '{text}'")
    # Expected: Inglese plausibile (anche se senza senso)

test_t5_decoder(model)
```

---

## 💡 Se Ancora BLEU < 10% Dopo Fix

### **Opzione A: Unfreeze Encoder (Risky!)** ⚠️

```bash
!python train_sonar_with_t5.py \
    ... \
    --freeze_encoder False \  # ← Train anche encoder!
    --learning_rate 1e-5 \     # ← LR molto più basso
    --epochs 30
```

**Pro**: Encoder si adatta meglio a T5
**Contro**: Rischio overfitting, perde pre-training

---

### **Opzione B: Usa T5-base (Più Potente)** ⭐

```bash
!python train_sonar_with_t5.py \
    ... \
    --t5_model t5-base \  # 220M params invece di 60M
    --batch_size 8 \      # Ridotto per memoria
    --epochs 15
```

**BLEU atteso: 15-22%** (ma più lento)

---

### **Opzione C: Prefix Tuning** (Advanced)

Invece di dare embedding diretto a T5, usa **prefix tokens**:

```python
# In __init__
self.prefix_tokens = nn.Parameter(torch.randn(16, 512))  # 16 prefix

# In forward
prefix = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1, -1)
combined = torch.cat([prefix, t5_embedding_final], dim=1)  # (B, 24, 512)

# Pass to T5
outputs = self.t5(inputs_embeds=combined, ...)
```

---

## 📝 Checklist Troubleshooting

- [ ] Sample translations mostrano parole rilevanti?
- [ ] Embeddings SONAR hanno alta similarità per frasi simili?
- [ ] T5 genera inglese plausibile con embedding random?
- [ ] Loss scende sotto 1.0 dopo 5 epochs?
- [ ] BLEU sale oltre 5% entro epoch 3?

Se **tutti NO** → Problema nell'architettura (rivedi pipeline)
Se **alcuni Sì** → Training funziona, serve solo più tempo/dati

---

## 🎯 Conclusione

**Modifiche applicate:**

1. ✅ Sequence expansion (1→8 tokens)
2. ✅ Projection più profonda
3. ✅ Learnable expander

**Prossimi passi:**

1. Rilancia training con modifiche
2. Aspetta 2 epochs (~30 min)
3. Se BLEU > 5% → continua fino a 20 epochs
4. Se BLEU < 5% → debugga con test sopra

**BLEU target realistico: 12-18%** (con dataset piccolo)
**BLEU ideale: 18-25%** (se tutto va bene)

🚀 **Rilancia ora e fammi sapere BLEU dopo 2 epochs!**
