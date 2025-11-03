# 🎯 Sign-to-Text Pipeline - Progress Report

**Data**: 2 Novembre 2025  
**Stato**: ✅ Fase 1 & 2 COMPLETE - Pronto per Model Development

---

## ✅ COMPLETATO OGGI

### 1. Estrazione Landmarks MediaPipe (8+ ore)
- **2,120 video processati** con successo (100%)
- **Tempo totale**: 486.6 minuti (~8 ore)
- **Output**: 288.8 MB di landmarks compressi (.npz)
- **Formato**: (n_frames, 375) = 75 landmarks × 5 coordinate
- **Location**: `data/processed/sign_language_landmarks/`

**Statistiche**:
- Frames medi: 138.1 per video
- File size medio: 139.5 KB
- Tempo medio: 13.77 sec/video

### 2. Analisi Dataset Utterances
- **2,123 utterances** con caption valide
- **2,404 parole uniche** nel vocabolario
- **Media caption**: 10.77 parole (range: 2-64)
- **Split creati**:
  - Train: 1,486 samples (70%)
  - Val: 318 samples (15%)
  - Test: 319 samples (15%)

**Output Files**:
- `results/utterances_analysis/statistics.json`
- `results/utterances_analysis/{train,val,test}_split.csv`
- `results/utterances_analysis/word_frequencies.txt`
- `results/utterances_analysis/utterances_analysis_plots.png`

### 3. Tokenizer BPE Implementato ✅
**File**: `src/sign_to_text/data/tokenizer.py`

**Features**:
- Byte-Pair Encoding (BPE) con tokenizers library
- Vocab size: **2,916 tokens**
- Special tokens: `[PAD]`, `[UNK]`, `[SOS]`, `[EOS]`
- Normalizzazione: lowercase + rimozione accenti
- Metaspace encoding per preservare spazi

**Salvato in**: `models/sign_to_text/tokenizer.json`

**Test**:
```python
Input:  "Hello my name is John"
IDs:    [2, 95, 84, 42, 109, 1049, 93, 247, 3]
Output: "hello my name is john"  ✅
```

### 4. Dataset Loader PyTorch ✅
**File**: `src/sign_to_text/data/dataset.py`

**Features**:
- `SignLanguageDataset` class con:
  - Caricamento landmarks .npz
  - Tokenizzazione caption automatica
  - Padding/truncate frames (max 200)
  - Padding/truncate caption (max 30)
  - Maschere per attention mechanism
  - Normalizzazione landmarks

**Collate Function**:
- Batch tensori: `(batch_size, max_frames, 375)`
- Maschere per gestione lunghezze variabili

**Test Output**:
```
Landmarks: torch.Size([4, 200, 375])
Caption IDs: torch.Size([4, 30])
✅ 1,486 training samples caricati correttamente
```

---

## 📊 STRUTTURA DATI ATTUALE

```
data/processed/
├── utterances_with_translations.csv (2,127 video)
└── sign_language_landmarks/
    ├── 2751812_landmarks.npz
    ├── 98592049_landmarks.npz
    └── ... (2,120 files)

results/utterances_analysis/
├── train_split.csv (1,486 samples)
├── val_split.csv (318 samples)
├── test_split.csv (319 samples)
├── statistics.json
└── utterances_analysis_plots.png

models/sign_to_text/
├── tokenizer.json
└── tokenizer.metadata.json

src/sign_to_text/
├── __init__.py
├── data/
│   ├── __init__.py
│   ├── tokenizer.py ✅
│   └── dataset.py ✅
├── models/ (da creare)
└── features/ (opzionale)
```

---

## 🚀 PROSSIMI STEP

### Step 3: Implementare Modello Seq2Seq Transformer
**File da creare**: `src/sign_to_text/models/seq2seq_transformer.py`

**Architettura**:
```
Input: Landmarks (B, 200, 375)
    ↓
[Encoder]
- Input Projection: Linear(375 → d_model)
- Positional Encoding
- N × Transformer Encoder Layers
- Output: (B, 200, d_model) = context vectors
    ↓
[Decoder]
- Token Embedding: (vocab_size, d_model)
- Positional Encoding
- N × Transformer Decoder Layers (con cross-attention su encoder output)
- Output Projection: Linear(d_model → vocab_size)
    ↓
Output: Logits (B, max_caption_len, vocab_size)
```

**Hyperparameters da scegliere**:
- `d_model`: 256 o 512
- `nhead`: 8
- `num_encoder_layers`: 4-6
- `num_decoder_layers`: 4-6
- `dim_feedforward`: 1024 o 2048
- `dropout`: 0.1

### Step 4: Training Pipeline
**File**: `src/sign_to_text/train.py`

**Componenti**:
- Loss: CrossEntropyLoss (ignora PAD token)
- Optimizer: AdamW (lr=1e-4, weight_decay=0.01)
- Scheduler: ReduceLROnPlateau o CosineAnnealing
- Early Stopping: patience=10 epochs
- MLflow logging
- Checkpoint saving (best val loss)

**Metriche**:
- Training loss
- Validation loss
- BLEU-4 score
- Word Error Rate (WER)

### Step 5: Evaluation Script
**File**: `src/sign_to_text/evaluate.py`

**Output**:
- BLEU-1, BLEU-2, BLEU-3, BLEU-4
- WER (Word Error Rate)
- CER (Character Error Rate)
- Esempi qualitativi (prediction vs ground truth)

---

## 💡 RACCOMANDAZIONI TECNICHE

### Training Strategy
1. **Curriculum Learning**: Inizia con caption brevi, poi aumenta gradualmente
2. **Data Augmentation**:
   - Temporal subsampling (skip frames casualmente)
   - Gaussian noise su landmarks
3. **Teacher Forcing**: Usa 100% all'inizio, riduci gradualmente

### Ottimizzazione
- Usa **Mixed Precision Training** (torch.cuda.amp) per velocità
- **Gradient Clipping** (max_norm=1.0) per stabilità
- **Warmup** learning rate per 5-10% step totali

### Hardware Requirements
- **GPU consigliata**: 8GB+ VRAM (RTX 3070/4060 o superiore)
- **Batch size**: 16-32 (dipende da GPU)
- **Training time stimato**: 4-8 ore su GPU (10-20 epoch)

---

## 📝 COMANDI QUICK REFERENCE

### Test Dataset
```bash
python src/sign_to_text/data/dataset.py
```

### Test Tokenizer
```bash
python src/sign_to_text/data/tokenizer.py
```

### Crea DataLoaders (da Python)
```python
from src.sign_to_text.data.dataset import get_dataloaders

loaders = get_dataloaders(batch_size=16, max_frames=200, max_caption_len=30)
train_loader = loaders['train']  # 93 batches
val_loader = loaders['val']      # 20 batches
test_loader = loaders['test']    # 20 batches
```

---

## ✅ CHECKLIST PROGRESSO

- [x] Estrazione landmarks MediaPipe (2,120 video)
- [x] Analisi dataset e creazione splits
- [x] Implementazione tokenizer BPE
- [x] Implementazione dataset loader PyTorch
- [ ] Implementazione modello Seq2Seq Transformer
- [ ] Script training con MLflow
- [ ] Script evaluation con metriche BLEU/WER
- [ ] Fine-tuning hyperparameter
- [ ] Documentazione risultati finali

---

## 🎯 DELIVERABLE TESI

Quando completerai i prossimi step avrai:

1. **Pipeline completa Video→Text**:
   - MediaPipe landmarks extraction
   - Seq2Seq Transformer model
   - Inference script

2. **Metriche quantitative**:
   - BLEU-4 score (target: >0.25)
   - WER (target: <50%)
   - Training curves

3. **Analisi qualitativa**:
   - Esempi traduzione corrette
   - Casi difficili / errori comuni
   - Confronto con baseline

4. **Codice riutilizzabile**:
   - Moduli ben organizzati
   - Documentazione completa
   - Test unit per componenti critici

---

**Pronto per continuare con Step 3: Modello Transformer?** 🚀
