# Sign-to-Text Pipeline - Setup Completato ✅

## 📊 Dataset Utterances Analizzato

**File**: `data/processed/utterances_with_translations.csv`

### Statistiche Dataset:

- **Total utterances**: 2127
- **Utterances valide** (con caption): 2123
- **Video disponibili**: 2127 (100%)
- **Caption mancanti**: 4

### Statistiche Caption:

- **Unique words**: 2,404
- **Total words**: 22,868
- **Lunghezza caption**:
  - Min: 2 parole
  - Max: 64 parole
  - Media: 10.77 parole
  - Mediana: 9 parole
  - Std: 7.01

### Top 10 Parole:

1. `the` (5.6%)
2. `to` (3.3%)
3. `i` (2.9%)
4. `a` (2.1%)
5. `is` (1.9%)
6. `and` (1.5%)
7. `my` (1.4%)
8. `if` (1.3%)
9. `you` (1.2%)
10. `it` (1.2%)

### Source Collections:

Top 3 collections con più utterances:

- `Cory_2013-6-27_sc114`: 112 utterances (avg 8.3 words)
- `Cory_2013-6-27_sc115`: 112 utterances (avg 9.3 words)
- `Cory_2013-6-27_sc112`: 98 utterances (avg 8.4 words)

---

## ✂️ Train/Val/Test Split Creato

**Directory**: `results/utterances_analysis/`

### Split Sizes:

- **Train**: 1,486 utterances (70.0%)
- **Val**: 318 utterances (15.0%)
- **Test**: 319 utterances (15.0%)

### Files Generati:

- `train_split.csv` - Training set
- `val_split.csv` - Validation set
- `test_split.csv` - Test set
- `statistics.json` - Statistiche complete
- `word_frequencies.txt` - Frequenze parole
- `utterances_analysis_plots.png` - Visualizzazioni

---

## 🎬 Estrazione Landmarks MediaPipe

**Modello**: `sign-language-translator.models.MediaPipeLandmarksModel()`  
**Output Directory**: `data/processed/sign_language_landmarks/`

### Test Completato ✅

Testato su **3 video** con successo:

- **Tempo medio**: ~11.4 sec/video
- **Frames medi**: 143.3 frames/video
- **File size medio**: 134.2 KB/video

### Formato Landmarks:

- **Shape**: `(n_frames, 375)` = 75 landmarks × 5 coordinates
- **Landmarks**: 33 pose + 21 left hand + 21 right hand
- **Coordinates**: x, y, z, visibility, presence (3D world coordinates)
- **Formato file**: `.npz` (NumPy compressed)

### Contenuto File NPZ:

```python
{
    'landmarks_3d': np.array (n_frames, 375), dtype=float32,
    'metadata': {
        'video_name': str,
        'caption': str,
        'source_collection': str,
        'n_frames': int,
        'fps': float,
        'duration_sec': float,
        'landmark_shape': tuple,
        'extraction_timestamp': float
    }
}
```

---

## 🚀 Prossimi Step

### 1. Estrazione Landmarks Completa (IN CORSO)

```bash
# Estrai tutti i 2123 video
python extract_landmarks_mediapipe.py --resume

# Tempo stimato: ~7-8 ore (2123 video × 11.4 sec)
```

**Features**:

- ✅ Resume capability (salta video già processati)
- ✅ Progress bar con tqdm
- ✅ Error handling robusto
- ✅ Report JSON con statistiche
- ✅ Log video falliti

### 2. Implementazione Tokenizer

**File da creare**: `src/sign_to_text/data/tokenizer.py`

**Raccomandazioni**:

- **Vocab size**: 3,404 (2,404 unique + 1,000 buffer)
- **Max sequence length**: 26 (95th percentile + SOS/EOS)
- **Tokenizer type**: BPE (Byte-Pair Encoding) con `tokenizers` library
- **Special tokens**: `[PAD]`, `[UNK]`, `[SOS]`, `[EOS]`

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# Train BPE tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

trainer = trainers.BpeTrainer(
    vocab_size=3404,
    special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]"]
)

# Train on all captions
tokenizer.train_from_iterator(all_captions, trainer)
tokenizer.save("models/sign_to_text_tokenizer.json")
```

### 3. Dataset Loader

**File da creare**: `src/sign_to_text/data/dataset.py`

```python
class SignLanguageDataset(torch.utils.data.Dataset):
    """
    Dataset per Sign-to-Text translation.

    Returns:
        landmarks: torch.Tensor (n_frames, 375)
        caption_ids: torch.Tensor (max_seq_len,)
        attention_mask: torch.Tensor (max_seq_len,)
    """
    def __init__(self, csv_path, landmarks_dir, tokenizer):
        self.df = pd.read_csv(csv_path)
        self.landmarks_dir = Path(landmarks_dir)
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        # Load landmarks .npz
        # Tokenize caption
        # Return (landmarks, caption_ids, mask)
```

### 4. Modello Seq2Seq

**File da creare**: `src/sign_to_text/models/seq2seq_transformer.py`

**Architettura Consigliata**:

```
Input: Landmarks (n_frames, 375)
    ↓
Temporal Encoder (BiLSTM o Transformer)
    → Hidden: (n_frames, hidden_dim)
    ↓
Attention Mechanism
    ↓
Text Decoder (Transformer Decoder)
    → Output: (max_seq_len, vocab_size)
    ↓
Caption: [SOS] "I want to go" [EOS]
```

**Hyperparameters**:

- `hidden_dim`: 512
- `num_layers`: 6 (encoder) + 6 (decoder)
- `num_heads`: 8
- `dropout`: 0.1
- `vocab_size`: 3,404

### 5. Training Loop

**File da creare**: `src/sign_to_text/train.py`

```python
# Training configuration
config = {
    'batch_size': 16,  # Dipende da GPU
    'learning_rate': 1e-4,
    'num_epochs': 50,
    'warmup_steps': 4000,
    'gradient_clip': 1.0,
    'label_smoothing': 0.1
}

# Loss
criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.1)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Scheduler
scheduler = transformers.get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=4000,
    num_training_steps=total_steps
)
```

### 6. Evaluation Metrics

**Metrics da implementare**:

- **BLEU-4**: Target >0.30
- **WER** (Word Error Rate): Target <40%
- **CER** (Character Error Rate)
- **ROUGE-L**

```python
from nltk.translate.bleu_score import sentence_bleu
from jiwer import wer

# BLEU
bleu_score = sentence_bleu([reference], hypothesis, weights=(0.25, 0.25, 0.25, 0.25))

# WER
wer_score = wer(reference, hypothesis)
```

---

## 📁 Struttura File

```
Improved_EmoSign_Thesis/
│
├── data/
│   ├── processed/
│   │   ├── utterances_with_translations.csv  # Dataset principale
│   │   └── sign_language_landmarks/          # Landmarks estratti (2123 .npz)
│   └── raw/
│       └── ASLLRP/
│           └── batch_utterance_video_v3_1/   # Video MP4 originali
│
├── results/
│   └── utterances_analysis/
│       ├── train_split.csv
│       ├── val_split.csv
│       ├── test_split.csv
│       ├── statistics.json
│       ├── word_frequencies.txt
│       └── utterances_analysis_plots.png
│
├── src/
│   └── sign_to_text/
│       ├── data/
│       │   ├── tokenizer.py       # TODO
│       │   └── dataset.py         # TODO
│       ├── models/
│       │   └── seq2seq_transformer.py  # TODO
│       ├── features/
│       │   └── landmarks.py       # Utility landmarks
│       └── train.py               # TODO
│
├── models/
│   └── sign_to_text_tokenizer.json  # TODO: Tokenizer salvato
│
├── docs/
│   ├── VIDEO_TO_TEXT_PIPELINE_ROADMAP.md
│   ├── QUICKSTART_SIGN_TO_TEXT.md
│   └── SIGN_TO_TEXT_SETUP_COMPLETE.md  # Questo file
│
├── analyze_utterances_dataset.py    # ✅ Analisi dataset
└── extract_landmarks_mediapipe.py   # ✅ Estrazione landmarks
```

---

## 🎯 Timeline Stimata

| Phase       | Task                     | Duration | Status         |
| ----------- | ------------------------ | -------- | -------------- |
| **Phase 1** | Dataset Analysis         | 1 day    | ✅ DONE        |
| **Phase 1** | Landmark Extraction      | 1 day    | 🟡 IN PROGRESS |
| **Phase 2** | Tokenizer Implementation | 0.5 day  | ⏳ TODO        |
| **Phase 2** | Dataset Loader           | 0.5 day  | ⏳ TODO        |
| **Phase 3** | Seq2Seq Model            | 2 days   | ⏳ TODO        |
| **Phase 4** | Training Loop            | 1 day    | ⏳ TODO        |
| **Phase 5** | Model Training           | 3-5 days | ⏳ TODO        |
| **Phase 6** | Evaluation & Tuning      | 2 days   | ⏳ TODO        |

**Total**: ~10-12 giorni

---

## 💡 Note Importanti

### MediaPipe Performance:

- **Processing speed**: ~11.4 sec/video
- **Total time per 2123 video**: ~7-8 ore
- **Consiglio**: Lasciare girare overnight con `--resume`

### Data Augmentation (Opzionale):

- Temporal subsampling (skip frames)
- Random masking (mascherare landmark casuali)
- Speed perturbation (velocizzare/rallentare)

### Curriculum Learning:

Durante training, ordinare batch per lunghezza caption crescente:

- Epoch 1-10: Caption corte (2-8 parole)
- Epoch 11-30: Caption medie (9-15 parole)
- Epoch 31-50: Tutte le caption

### Hardware Consigliato:

- **GPU**: RTX 3090 / A100 (16GB+ VRAM)
- **Batch size**: 16-32
- **Training time**: 2-3 giorni

---

## 🔗 Risorse

### Documentazione:

- [sign-language-translator Docs](https://sign-language-translator.readthedocs.io/)
- [MediaPipe Holistic](https://google.github.io/mediapipe/solutions/holistic.html)
- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/)

### Papers Rilevanti:

- "Attention Is All You Need" (Transformer)
- "BERT: Pre-training of Deep Bidirectional Transformers" (Tokenization)
- "Neural Machine Translation by Jointly Learning to Align and Translate" (Seq2Seq + Attention)

---

**Last Updated**: 2025-01-31  
**Status**: Phase 1 Completata, Phase 2 Pronta per iniziare  
**Next Command**: `python extract_landmarks_mediapipe.py --resume`
