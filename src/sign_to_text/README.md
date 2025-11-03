# 🔤 Sign-to-Text Translation Pipeline

**Video Sign Language → Testo** usando Transformer Encoder-Decoder.

---

## 📋 Overview

Questo modulo implementa un sistema completo per tradurre video in American Sign Language (ASL) in testo inglese.

### Pipeline Completa:

```
Video ASL (.mp4)
    ↓
MediaPipe Landmarks Extraction
    ↓
Landmarks (n_frames, 375)
    ↓
Transformer Encoder
    ↓
Context Vectors
    ↓
Transformer Decoder (autoregressive)
    ↓
Caption Text
```

---

## ✅ Stato Implementazione

- [x] **Estrazione Landmarks** - MediaPipe (2,120 video processati)
- [x] **Tokenizer BPE** - 2,916 tokens, special tokens
- [x] **Dataset Loader** - PyTorch Dataset con padding/masking
- [x] **Modello Transformer** - Encoder-Decoder (9M parametri)
- [x] **Training Pipeline** - MLflow, early stopping, scheduler
- [x] **Hyperparameter Tuning** - Optuna con MLflow integration
- [ ] **Evaluation Script** - BLEU, WER, CER metrics
- [ ] **Inference API** - Video → Text endpoint

---

## 🚀 Quick Start

### 1. Preparazione Dati (GIÀ FATTO)

```bash
# Estrai landmarks da video
python extract_landmarks_mediapipe.py --resume

# Analizza dataset e crea splits
python analyze_utterances_dataset.py
```

**Output**:

- `data/processed/sign_language_landmarks/` - 2,120 file .npz
- `results/utterances_analysis/{train,val,test}_split.csv`

### 2. Training Tokenizer (GIÀ FATTO)

```bash
python src/sign_to_text/data/tokenizer.py
```

**Output**:

- `models/sign_to_text/tokenizer.json` - Tokenizer BPE
- Vocab: 2,916 tokens

### 3. Hyperparameter Tuning (CONSIGLIATO)

Prima di lanciare training lungo, ottimizza i parametri:

#### Quick Tuning (20 trials, ~1-2 ore)

```bash
./run_tuning.sh
```

#### Custom Tuning

```bash
# Tuning veloce su subset
.venv/bin/python src/sign_to_text/tune.py \
    --n_trials 20 \
    --epochs 3 \
    --subset_fraction 0.3 \
    --optimize bleu

# Tuning completo
.venv/bin/python src/sign_to_text/tune.py \
    --n_trials 30 \
    --epochs 5 \
    --optimize bleu
```

**Output**:

- `results/best_hyperparameters.json` - Migliori parametri
- `results/param_importances.html` - Importanza parametri
- MLflow UI con tutti i trials

📚 **Guida completa**: `docs/HYPERPARAMETER_TUNING_GUIDE.md`

### 4. Training Modello

#### Quick Test (1 epoch, modello piccolo)

```bash
python test_sign_to_text_training.py
```

#### Training con Best Params da Tuning

```bash
# Carica automaticamente da results/best_hyperparameters.json
python src/sign_to_text/train_from_tuning.py
```

#### Training Manuale

**Small Model** (buono per test veloce):

```bash
python src/sign_to_text/train.py \
    --epochs 50 \
    --batch_size 16 \
    --d_model 128 \
    --nhead 4 \
    --num_encoder_layers 2 \
    --num_decoder_layers 2 \
    --lr 1e-4 \
    --patience 10
```

**Full Model** (miglior performance):

```bash
python src/sign_to_text/train.py \
    --epochs 100 \
    --batch_size 16 \
    --d_model 256 \
    --nhead 8 \
    --num_encoder_layers 4 \
    --num_decoder_layers 4 \
    --lr 1e-4 \
    --patience 15
```

**Large Model** (se hai GPU potente):

```bash
python src/sign_to_text/train.py \
    --epochs 100 \
    --batch_size 32 \
    --d_model 512 \
    --nhead 8 \
    --num_encoder_layers 6 \
    --num_decoder_layers 6 \
    --lr 5e-5 \
    --patience 20
```

### 5. Monitoring Training

```bash
# Avvia MLflow UI
mlflow ui

# Apri browser: http://localhost:5000
# Esperimento: "sign_to_text"
```

### 5. Inference (TODO)

```python
from src.sign_to_text import SignToTextInference

# Carica modello
model = SignToTextInference.load("models/sign_to_text/checkpoints/best.pt")

# Predici
caption = model.predict("path/to/video.mp4")
print(caption)  # "Hello my name is John"
```

---

## 📁 Struttura File

```
src/sign_to_text/
├── README.md (questo file)
├── __init__.py
│
├── data/
│   ├── __init__.py
│   ├── tokenizer.py         ✅ BPE tokenizer (2,916 tokens)
│   └── dataset.py           ✅ PyTorch Dataset + DataLoader
│
├── models/
│   ├── __init__.py
│   └── seq2seq_transformer.py  ✅ Transformer Encoder-Decoder
│
├── train.py                 ✅ Training script con MLflow
└── evaluate.py              ⏳ Evaluation script (TODO)
```

---

## 🎯 Architettura Modello

### Transformer Encoder-Decoder

**Input**: Landmarks MediaPipe (B, 200 frames, 375 features)

**Encoder**:

```
Landmarks (B, 200, 375)
    ↓ Linear Projection
(B, 200, d_model)
    ↓ Positional Encoding
(B, 200, d_model)
    ↓ N × TransformerEncoderLayer
        - Multi-Head Self-Attention
        - Feed-Forward Network
        - Layer Normalization
(B, 200, d_model) = Context Vectors
```

**Decoder**:

```
Token IDs (B, max_len)
    ↓ Embedding + Positional Encoding
(B, max_len, d_model)
    ↓ N × TransformerDecoderLayer
        - Masked Self-Attention (causal)
        - Cross-Attention (su context vectors)
        - Feed-Forward Network
        - Layer Normalization
(B, max_len, d_model)
    ↓ Linear Projection
(B, max_len, vocab_size) = Logits
```

### Hyperparameters

| Parameter            | Small | Medium | Large   |
| -------------------- | ----- | ------ | ------- |
| `d_model`            | 128   | 256    | 512     |
| `nhead`              | 4     | 8      | 8       |
| `num_encoder_layers` | 2     | 4      | 6       |
| `num_decoder_layers` | 2     | 4      | 6       |
| `dim_feedforward`    | 512   | 1024   | 2048    |
| `dropout`            | 0.1   | 0.1    | 0.1     |
| **Total Params**     | ~1.7M | ~9M    | ~36M    |
| **GPU Memory**       | ~2GB  | ~4GB   | ~8GB    |
| **Training Time**    | ~2h   | ~4-6h  | ~12-24h |

---

## 📊 Dataset

### Statistiche

- **Total videos**: 2,123 (con caption valide)
- **Vocab size**: 2,916 tokens
- **Avg caption length**: 10.77 parole
- **Avg video length**: 138 frames (~9 secondi @ 15 FPS)

### Splits

| Split | Samples     | Batches (BS=16) |
| ----- | ----------- | --------------- |
| Train | 1,486 (70%) | 93              |
| Val   | 318 (15%)   | 20              |
| Test  | 319 (15%)   | 20              |

### Formato Landmarks

- **Shape**: `(n_frames, 375)`
- **375 features** = 75 landmarks × 5 coordinate:
  - 33 pose landmarks
  - 21 left hand landmarks
  - 21 right hand landmarks
- **Coordinate**: (x, y, z, visibility, presence)

---

## 🔧 Training Details

### Loss Function

- **CrossEntropyLoss** con:
  - `ignore_index=PAD_TOKEN` (ignora padding nel calcolo loss)
  - `label_smoothing=0.1` (regularizzazione)

### Optimizer

- **AdamW**:
  - Learning rate: 1e-4
  - Weight decay: 0.01
  - Betas: (0.9, 0.98)
  - Gradient clipping: max_norm=1.0

### Learning Rate Scheduler

- **ReduceLROnPlateau**:
  - Factor: 0.5
  - Patience: 5 epochs
  - Monitor: validation loss

### Early Stopping

- Patience: 10-15 epochs
- Monitor: validation loss
- Salva best model automaticamente

### Metriche

- **Training**: Loss
- **Validation**: Loss + BLEU-4 score
- **Logging**: MLflow (ogni 50 step)

---

## 📈 Risultati Attesi

### Target Metriche

| Metrica      | Target | Ottimo |
| ------------ | ------ | ------ |
| **BLEU-4**   | >0.20  | >0.30  |
| **BLEU-1**   | >0.40  | >0.50  |
| **WER**      | <60%   | <40%   |
| **Val Loss** | <3.0   | <2.0   |

### Training Curves

- **Loss**: Dovrebbe decrementare da ~7.0 a ~2.0-3.0
- **BLEU**: Dovrebbe incrementare da ~0.0 a ~0.20-0.30
- **Convergenza**: 20-50 epoch (con early stopping)

---

## 💡 Tips & Troubleshooting

### GPU Out of Memory

Riduci:

- `batch_size` (16 → 8 → 4)
- `d_model` (256 → 128)
- `max_frames` (200 → 150)

### Training troppo lento (CPU)

- Riduci `num_workers` in DataLoader
- Usa modello Small
- Considera GPU cloud (Google Colab, AWS)

### Overfitting

- Aumenta dropout (0.1 → 0.2)
- Data augmentation (random frame skipping)
- Riduci model size

### Underfitting

- Aumenta model capacity (d_model, layers)
- Più epoch
- Riduci dropout

### BLEU basso

- Controlla che tokenizer decodifichi correttamente
- Verifica sample predictions manualmente
- Considera beam search invece di greedy decoding

---

## 🔍 Debugging

### Controlla Dataset

```python
from src.sign_to_text.data.dataset import SignLanguageDataset
from src.sign_to_text.data.tokenizer import SignLanguageTokenizer

tokenizer = SignLanguageTokenizer.load("models/sign_to_text/tokenizer.json")
dataset = SignLanguageDataset(
    split_csv="results/utterances_analysis/train_split.csv",
    landmarks_dir="data/processed/sign_language_landmarks",
    tokenizer=tokenizer
)

sample = dataset[0]
print(sample['landmarks'].shape)  # (200, 375)
print(sample['caption_text'])
print(tokenizer.decode(sample['caption_ids'].tolist()))
```

### Controlla Modello

```python
from src.sign_to_text.models.seq2seq_transformer import SignToTextTransformer

model = SignToTextTransformer(vocab_size=3000, d_model=256)
print(f"Parameters: {model.count_parameters():,}")

# Forward pass test
import torch
src = torch.randn(2, 150, 375)
tgt = torch.randint(0, 3000, (2, 20))
logits = model(src, tgt)
print(logits.shape)  # (2, 20, 3000)
```

### Monitora Training

```bash
# MLflow UI
mlflow ui

# TensorBoard (alternativa)
tensorboard --logdir mlruns
```

---

## 📚 References

### Papers

- **Attention Is All You Need** (Vaswani et al., 2017)
- **Sign Language Translation** (Camgoz et al., 2018)
- **MediaPipe Holistic** (Bazarevsky et al., 2020)

### Codebases

- [sign-language-translator](https://github.com/sign-language-translator/sign-language-translator)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PyTorch Seq2Seq](https://github.com/bentrevett/pytorch-seq2seq)

---

## 🎓 Per la Tesi

### Deliverable

1. **Pipeline completa** funzionante
2. **Metriche quantitative**:
   - BLEU-4, BLEU-1, WER, CER
   - Training/validation curves
3. **Analisi qualitativa**:
   - Esempi traduzione corrette
   - Error analysis
   - Confronto con baseline
4. **Codice documentato**:
   - README dettagliati
   - Docstrings complete
   - Test funzionanti

### Capitolo Tesi Suggerito

```
Capitolo 5: Sign Language Translation

5.1 Introduzione
    - Motivazione (accessibilità, comunicazione)
    - State of the art

5.2 Dataset
    - ASLLRP dataset (2,123 video)
    - Preprocessing MediaPipe
    - Splits e statistiche

5.3 Architettura
    - Transformer Encoder-Decoder
    - Positional encoding
    - Attention mechanism

5.4 Training
    - Loss, optimizer, scheduler
    - Hyperparameter tuning
    - Early stopping

5.5 Risultati
    - Metriche (BLEU, WER)
    - Curve training
    - Esempi qualitativi
    - Error analysis

5.6 Conclusioni e Lavori Futuri
    - Limitazioni
    - Possibili miglioramenti
```

---

## ✅ Next Steps

### Immediato

1. ✅ **Training completo** - Esegui training con modello Medium
2. ⏳ **Evaluation script** - Implementa metriche BLEU/WER su test set
3. ⏳ **Inference API** - Script per predire da nuovo video

### Avanzato

4. **Beam Search** - Migliora qualità generazione
5. **Data Augmentation** - Temporal augmentation dei landmarks
6. **Ensemble** - Combina modelli diversi
7. **Back-translation** - Text→Sign per data augmentation

---

**Ready to train!** 🚀

```bash
# Start training
python src/sign_to_text/train.py --epochs 50 --batch_size 16
```
