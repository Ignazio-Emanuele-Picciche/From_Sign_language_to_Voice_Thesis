# How2Sign Training - Ready to Launch! 🚀

## ✅ Setup Completato

### 📊 Dataset Verificato

- **Train**: 29,602 samples (95% di 31k hanno landmarks validi)
- **Val**: 1,739 samples
- **Landmarks**: OpenPose 411 features già estratti
- **Caption length**: media 17.5 parole (max 50)
- **Vocab**: 15,548 unique words

### 🧠 Modello Configurato

- **Architettura**: Seq2Seq Transformer
- **Parameters**: 33.7M trainable
- **Input**: 411 features (OpenPose: pose + hands + face)
- **Model dim**: 512
- **Encoder layers**: 4
- **Decoder layers**: 4
- **Attention heads**: 8

### ⚙️ Training Configuration

- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)
- **Loss**: CrossEntropyLoss con label smoothing 0.1
- **Scheduler**: ReduceLROnPlateau (patience=3)
- **Batch size**: 4-16 (dipende da memoria GPU/MPS)
- **Gradient clipping**: 1.0
- **Device**: MPS (Apple Silicon)

---

## 🚀 Come Lanciare il Training

### Opzione 1: Script Automatico (RACCOMANDATO)

```bash
# Training completo (30 epochs)
./run_how2sign_training.sh
```

### Opzione 2: Comando Manuale

```bash
# Training test (2 epochs, batch piccolo)
python src/sign_to_text/train_how2sign.py \
  --epochs 2 \
  --batch_size 8 \
  --num_workers 0

# Training completo (30 epochs)
python src/sign_to_text/train_how2sign.py \
  --epochs 30 \
  --batch_size 16 \
  --lr 1e-4 \
  --d_model 512 \
  --num_encoder_layers 4 \
  --num_decoder_layers 4 \
  --num_workers 4
```

### Opzione 3: Training con Hyperparameters Ottimizzati

```bash
# Usa i migliori parametri da Optuna tuning
python src/sign_to_text/train_how2sign.py \
  --epochs 50 \
  --batch_size 16 \
  --lr 3.24e-05 \
  --d_model 512 \
  --num_encoder_layers 2 \
  --num_decoder_layers 2 \
  --nhead 4 \
  --dropout 0.367 \
  --weight_decay 0.00071 \
  --label_smoothing 0.021
```

---

## 📁 File Creati

### Scripts

- ✅ `src/sign_to_text/train_how2sign.py` - Training script principale
- ✅ `src/sign_to_text/data/how2sign_dataset.py` - PyTorch Dataset per OpenPose
- ✅ `src/sign_to_text/data_preparation/prepare_how2sign_dataset.py` - Preparazione splits
- ✅ `run_how2sign_training.sh` - Launcher script

### Data

- ✅ `results/how2sign_splits/train_split.csv` - Train split (29.6k samples)
- ✅ `results/how2sign_splits/val_split.csv` - Val split (1.7k samples)
- ✅ `results/how2sign_analysis/how2sign_statistics.json` - Statistiche dataset

### Model Output

- 📁 `models/sign_to_text/how2sign/` - Checkpoints training
  - `best_checkpoint.pt` - Best model (val loss)
  - `last_checkpoint.pt` - Last epoch
  - `config.json` - Hyperparameters
  - `history.json` - Training history

---

## ⏱️ Tempo Stimato Training

### Con batch_size=16, ~7400 batches/epoch:

- **1 epoch**: ~20-30 min (su MPS)
- **30 epochs**: ~10-15 ore
- **50 epochs**: ~16-25 ore

### Suggerimenti:

- Lancia in background: `nohup ./run_how2sign_training.sh &`
- Monitora progress: `tail -f logs/how2sign_training_*.log`
- Usa early stopping: training si ferma se val loss non migliora per 5+ epochs

---

## 📊 Metriche di Successo

### Baseline ASLLRP (prima del tuning):

- Train Loss: 2.70
- Val Loss: 5.22
- BLEU: ~0.00

### Target How2Sign (con 17x più dati):

- Train Loss: <2.0
- Val Loss: <3.0
- BLEU-1: >0.15
- BLEU-4: >0.05

### Best Case (con tuning + 31k samples):

- Val Loss: <2.5
- BLEU-4: >0.10

---

## 🎯 Prossimi Step Dopo Training

### 1. Evaluation

```bash
# Crea script evaluate_how2sign.py
python src/sign_to_text/evaluate_how2sign.py \
  --checkpoint models/sign_to_text/how2sign/best_checkpoint.pt \
  --test_csv results/how2sign_splits/val_split.csv
```

### 2. Test su Golden Label

```bash
# Testa sul test set con emotion labels
python src/sign_to_text/evaluate_golden_label.py \
  --checkpoint models/sign_to_text/how2sign/best_checkpoint.pt
```

### 3. Fine-tuning su ASLLRP (opzionale)

```bash
# Pre-train How2Sign → Fine-tune ASLLRP
python src/sign_to_text/train.py \
  --checkpoint models/sign_to_text/how2sign/best_checkpoint.pt \
  --train_csv results/utterances_analysis/train_split.csv \
  --val_csv results/utterances_analysis/val_split.csv \
  --epochs 10 \
  --lr 1e-5
```

---

## 🐛 Troubleshooting

### Memory Issues

```bash
# Riduci batch size
--batch_size 4

# Riduci max_frames
--max_frames 100

# Riduci model dim
--d_model 256
```

### Slow Training

```bash
# Aumenta num_workers (se hai CPU cores disponibili)
--num_workers 4

# Usa gradient accumulation
# (implementa in train_how2sign.py)
```

### NaN Loss

```bash
# Riduci learning rate
--lr 5e-5

# Aumenta gradient clipping
--grad_clip 0.5
```

---

## ✅ Checklist Pre-Launch

- [x] Dataset How2Sign preparato (29.6k train, 1.7k val)
- [x] Splits CSV creati
- [x] Landmarks OpenPose verificati (95% disponibilità)
- [x] Tokenizer presente (vocab 4000)
- [x] Model architecture adattata (411 features input)
- [x] Training script testato (loss scende correttamente)
- [x] Output directory creata
- [x] Launcher script pronto

---

## 🚀 READY TO LAUNCH!

```bash
# Lancia il training completo
./run_how2sign_training.sh

# O in background
nohup ./run_how2sign_training.sh > logs/training.log 2>&1 &

# Monitora progress
tail -f logs/training.log
```

**Il modello Sign-to-Text su How2Sign è pronto per il training! 🎉**
