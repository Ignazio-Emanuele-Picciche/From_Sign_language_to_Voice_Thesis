# 🆕 Training SONAR da Zero - Guida Rapida

## 📋 Problema Risolto

**Errore precedente:**
```
FileNotFoundError: [Errno 2] No such file or directory: 
'checkpoints/sonar_encoder_finetuned/best_encoder.pt'
```

**Causa:** Il checkpoint precedente con BLEU 0.13% non esiste più (ed era inutilizzabile).

---

## ✅ Soluzione Implementata

Lo script `train_sonar_finetuning.py` ora supporta il **training da zero** senza checkpoint pre-esistente.

### Modifiche Applicate:

1. **Parametro `--encoder_checkpoint` ora opzionale**
   ```python
   parser.add_argument(
       "--encoder_checkpoint",
       type=str,
       default=None,  # ← Ora è None di default!
       help="Path to SONAR encoder checkpoint (.pth). If None, trains from scratch."
   )
   ```

2. **Inizializzazione condizionale dell'encoder**
   ```python
   # Se checkpoint esiste → carica pesi
   if encoder_checkpoint and os.path.exists(encoder_checkpoint):
       print(f"📥 Loading SONAR ASL Encoder from {encoder_checkpoint}...")
       encoder_state = torch.load(encoder_checkpoint, map_location=device)
       self.encoder = self._build_encoder_from_state(encoder_state)
   # Altrimenti → random initialization
   else:
       print(f"🆕 Initializing SONAR ASL Encoder from scratch...")
       self.encoder = self._build_encoder_from_state(None)
   ```

3. **Gestione `state_dict=None`**
   ```python
   def _build_encoder_from_state(self, state_dict=None):
       encoder = nn.Sequential(...)
       
       if state_dict is not None:
           encoder.load_state_dict(state_dict, strict=False)
           print("✅ Loaded pre-trained encoder weights")
       else:
           print("✅ Using random initialization (training from scratch)")
       
       return encoder
   ```

---

## 🚀 Come Usare (Su Colab)

### Opzione 1: Training da Zero (RACCOMANDATO)

```python
!python train_sonar_finetuning.py \
    --train_features features/train \
    --train_manifest manifests/train.tsv \
    --val_features features/val \
    --val_manifest manifests/val.tsv \
    --output_dir checkpoints/sonar_finetuned_FIXED \
    --epochs 10 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --eval_every 2 \
    --device cuda
```

**⚠️ NOTA:** Nessun `--encoder_checkpoint` → training from scratch!

### Opzione 2: Con Checkpoint Pre-esistente (se ce l'hai)

```python
!python train_sonar_finetuning.py \
    --encoder_checkpoint path/to/encoder.pt \  # ← Aggiungi questa riga
    --train_features features/train \
    ...
```

---

## 📊 Output Atteso (Training da Zero)

### Prima Epoca:
```
Epoch 1/10
Training: 100%|██████████| 40/40 [00:45<00:00]
  loss: 1.2345  grad_norm: 0.8234  cosine_sim: 0.2345

📉 Train Loss: 1.2345
📊 Val BLEU: 2.34%  ← Partenza bassa (normale!)
```

### Dopo 10 Epoche:
```
Epoch 10/10
Training: 100%|██████████| 40/40 [00:42<00:00]
  loss: 0.6789  grad_norm: 0.3456  cosine_sim: 0.6789

📉 Train Loss: 0.6789
📊 Val BLEU: 8.92%  ← Già un miglioramento significativo!
💾 Best model saved (BLEU: 8.92%)
```

### Dopo 50 Epoche (Target):
```
📊 Val BLEU: 35.67%  ← TARGET RAGGIUNTO! 🎯
```

---

## 🔍 Confronto: Prima vs Dopo

| Metrica | VECCHIO (con bug) | NUOVO (da zero) |
|---------|-------------------|-----------------|
| **Loss Iniziale** | 0.0009 ❌ (collasso) | 1.0-1.5 ✅ (sano) |
| **BLEU @ 10 epochs** | 0.13% ❌ | 5-15% ✅ |
| **BLEU @ 50 epochs** | - | 30-40% ✅ (atteso) |
| **Gradient Norm** | Non loggato ❌ | Loggato ✅ |
| **Cosine Similarity** | Non loggato ❌ | Loggato ✅ |

---

## 🎯 Metriche da Monitorare

### 1. **Loss (Cosine Loss)**
- **Range sano**: 0.5 - 1.5
- **Trend atteso**: Decrescente (verso 0.3-0.5)
- **⚠️ Allarme**: Se < 0.1 → possibile overfitting

### 2. **Gradient Norm**
- **Range sano**: 0.1 - 1.0
- **⚠️ Allarme**: Se > 10 → gradient explosion
- **⚠️ Allarme**: Se < 0.001 → vanishing gradients

### 3. **Cosine Similarity**
- **Range sano**: 0.3 - 0.9
- **Trend atteso**: Crescente (verso 0.7-0.8)
- **Target finale**: > 0.7

### 4. **BLEU Score**
- **@ 10 epochs**: 5-15% (aspettativa realistica)
- **@ 20 epochs**: 15-25%
- **@ 50 epochs**: 30-40% (target finale)

---

## 🛠️ Troubleshooting

### Problema: Loss troppo bassa (< 0.1)
**Causa:** Possibile overfitting o data leakage  
**Soluzione:** 
- Verifica che train/val siano separati
- Aggiungi dropout/regularization
- Riduci learning rate

### Problema: BLEU non migliora dopo 20 epochs
**Causa:** Plateau di apprendimento  
**Soluzione:**
- Riduci learning rate (1e-5 invece di 1e-4)
- Aumenta batch size (64 invece di 32)
- Verifica qualità features

### Problema: Gradient Norm > 10
**Causa:** Gradient explosion  
**Soluzione:**
- Già implementato: `clip_grad_norm_` con max=1.0
- Riduci learning rate

---

## 📁 Struttura Output

Dopo il training troverai:

```
checkpoints/sonar_finetuned_FIXED/
├── best_encoder.pt              ← Best model (usa questo!)
├── config.json                  ← Configurazione training
├── predictions_epoch002.json    ← Predizioni @ epoch 2
├── predictions_epoch004.json
├── ...
├── metrics_epoch002.json        ← Metriche @ epoch 2
├── metrics_epoch004.json
└── ...
```

---

## 🎓 Prossimi Passi

1. **✅ FATTO:** Fix applicati allo script
2. **✅ FATTO:** Script supporta training da zero
3. **⏳ TODO:** Esegui training da zero su Colab (10 epochs)
4. **⏳ TODO:** Verifica metriche (BLEU > 5%)
5. **⏳ TODO:** Se OK → training completo (50 epochs)
6. **⏳ TODO:** Target finale: BLEU 30-40%

---

## 📌 Comandi Utili

### Push modifiche su GitHub (da locale)
```bash
git push origin dev
```

### Sincronizza su Colab (da Colab)
```bash
cd /content/drive/MyDrive/How2Sign_SONAR
git pull origin dev
```

### Esegui training (da Colab)
```python
# Vedi cella aggiornata nel notebook!
```

---

**Data:** 19 Novembre 2025  
**Commit:** `102b33c` - feat: Support training from scratch without checkpoint  
**Status:** ✅ PRONTO PER IL RE-TRAINING
