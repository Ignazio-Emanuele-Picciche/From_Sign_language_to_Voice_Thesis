# Sign Language to Text Pipeline

Pipeline completa per la trascrizione di video ASL (American Sign Language) in testo, con integrazione nel sistema di analisi emozioni esistente.

---

## 📋 Overview

Questo modulo estende il progetto Improved_EmoSign_Thesis con:

- **Feature extraction** da video ASL usando MediaPipe landmarks
- **Modello Seq2Seq** per traduzione Sign→Text
- **Integrazione** con pipeline ViViT esistente (Video→Emotion)

**Pipeline completa:**

```
Video ASL → Landmarks → Testo → Emozione → Audio TTS
            (SLT)      (Seq2Seq)  (ViViT)   (Edge-TTS)
```

---

## 🎯 Status Progetto

| Componente          | Status            | Note                         |
| ------------------- | ----------------- | ---------------------------- |
| Feature Extraction  | ⏳ In development | Usa sign-language-translator |
| Dataset Preparation | ⏳ In development | 202 video con ground truth   |
| Seq2Seq Model       | 📋 Planned        | Transformer encoder-decoder  |
| Training Pipeline   | 📋 Planned        | MLflow tracking              |
| Evaluation          | 📋 Planned        | BLEU, WER metrics            |
| Integration         | 📋 Planned        | E2E pipeline                 |

**Legenda:**

- ✅ Completato
- ⏳ In development
- 📋 Planned
- ❌ Bloccato

---

## 📊 Dataset

### Ground Truth

- **File:** `data/processed/golden_label_sentiment.csv`
- **Samples:** 202 video ASL
- **Colonne:** video_name, caption, emotion
- **Emozioni:** Positive, Negative (balanced)

### Esempio

```csv
video_name,caption,emotion
83664512.mp4,"I was like, 'Oh, wow, that's fine.'...",Positive
25328.mp4,There are so many things that are wrong...,Negative
```

### Split Proposto

- **Train:** 70% (~141 video)
- **Validation:** 15% (~30 video)
- **Test:** 15% (~31 video)

---

## 🛠️ Installazione

### Requisiti

```bash
# Core dependencies
pip install "sign-language-translator[all]"  # MediaPipe landmarks
pip install torch>=2.0.0
pip install transformers>=4.30.0
pip install tokenizers

# Evaluation
pip install sacrebleu  # BLEU metric
pip install jiwer      # WER/CER metrics

# Già installati nel progetto
# mlflow, numpy, pandas, matplotlib
```

### Quick Test

```bash
# Test feature extraction
python test_sign_language_extraction.py \
    --mode single \
    --video_path data/raw/ASLLRP/videos/83664512.mp4

# Test batch processing
python test_sign_language_extraction.py \
    --mode batch \
    --n_samples 5
```

---

## 📁 Struttura File

```
Improved_EmoSign_Thesis/
├── src/
│   ├── features/
│   │   ├── sign_language_embeddings.py    # NEW - Landmark extraction
│   │   └── landmark_preprocessing.py      # NEW - Normalization
│   │
│   ├── models/
│   │   ├── sign_to_text/                  # NEW - Modulo principale
│   │   │   ├── __init__.py
│   │   │   ├── model.py                   # Seq2Seq Transformer
│   │   │   ├── train.py                   # Training loop
│   │   │   ├── dataset.py                 # DataLoader
│   │   │   ├── tokenizer.py               # Caption tokenizer
│   │   │   └── evaluate.py                # BLEU/WER metrics
│   │   │
│   │   └── two_classes/vivit/             # EXISTING - Emotion classifier
│   │
│   ├── api/
│   │   └── sign_language_api.py           # NEW - REST API
│   │
│   └── demo/
│       └── sign_language_demo.py          # NEW - Gradio interface
│
├── data/
│   ├── processed/
│   │   ├── sign_language_embeddings/      # NEW - Landmarks salvati
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   │
│   │   └── golden_label_sentiment.csv     # EXISTING - Ground truth
│   │
│   └── raw/
│       └── ASLLRP/videos/                 # EXISTING - Video files
│
├── results/
│   ├── sign_to_text/                      # NEW - Training results
│   │   ├── checkpoints/
│   │   ├── logs/
│   │   └── predictions/
│   │
│   └── sign_language_test/                # NEW - Test outputs
│
├── docs/
│   ├── VIDEO_TO_TEXT_PIPELINE_ROADMAP.md  # Roadmap completa
│   ├── QUICKSTART_SIGN_TO_TEXT.md         # Quick start guide
│   └── SIGN_LANGUAGE_TRANSLATOR_DECISION.md # Analisi libreria
│
└── test_sign_language_extraction.py       # Script di test

```

---

## 🚀 Quick Start

### 1. Feature Extraction

```python
import sign_language_translator as slt

# Load video
video = slt.Video("data/raw/ASLLRP/videos/83664512.mp4")

# Extract 3D landmarks
model = slt.models.MediaPipeLandmarksModel()
landmarks = model.embed(video.iter_frames(), landmark_type="world")

print(f"Landmarks shape: {landmarks.shape}")  # (n_frames, 375)
```

### 2. Dataset Preparation (TODO)

```python
from src.features.sign_language_embeddings import extract_all_landmarks

# Estrai landmarks da tutti i video
extract_all_landmarks(
    csv_path="data/processed/golden_label_sentiment.csv",
    video_dirs=["data/raw/ASLLRP/videos"],
    output_dir="data/processed/sign_language_embeddings"
)
```

### 3. Model Training (TODO)

```python
from src.models.sign_to_text import SignToTextModel, train_model

# Initialize model
model = SignToTextModel(
    landmark_dim=375,
    vocab_size=10000,
    d_model=512,
    num_encoder_layers=6,
    num_decoder_layers=6
)

# Train
train_model(
    model=model,
    train_data_path="data/processed/sign_language_embeddings/train",
    val_data_path="data/processed/sign_language_embeddings/val",
    output_dir="results/sign_to_text"
)
```

### 4. Evaluation (TODO)

```python
from src.models.sign_to_text import evaluate_model

# Evaluate on test set
metrics = evaluate_model(
    model_path="results/sign_to_text/checkpoints/best_model.pt",
    test_data_path="data/processed/sign_language_embeddings/test"
)

print(f"BLEU-4: {metrics['bleu_4']:.3f}")
print(f"WER: {metrics['wer']:.3f}")
```

---

## 📊 Metriche di Successo

### MVP (Minimum Viable Product)

- [ ] BLEU-4 > 0.30
- [ ] WER < 40%
- [ ] Inference time < 5 sec per video (30 sec durata)
- [ ] Accuracy emozione > 70% con testo generato

### Target Finale

- [ ] BLEU-4 > 0.45
- [ ] WER < 25%
- [ ] Inference time < 2 sec
- [ ] Accuracy emozione comparabile a ground truth

---

## 🔬 Metodologia

### Architettura Modello

```python
class SignToTextModel(nn.Module):
    """
    Seq2Seq Transformer per traduzione Sign→Text

    Input:  Landmarks (n_frames, 375)
    Output: Text tokens (n_tokens,)
    """

    def __init__(self, landmark_dim=375, vocab_size=10000):
        # Encoder: processa sequenza landmarks
        self.landmark_projection = nn.Linear(landmark_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.encoder = TransformerEncoder(...)

        # Decoder: genera sequenza testo
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder = TransformerDecoder(...)
        self.output_projection = nn.Linear(d_model, vocab_size)
```

### Data Augmentation

- **Temporal:** Random subsampling, time warping
- **Spatial:** Rotation, scaling, translation (preservando relazioni)
- **Noise:** Gaussian noise sui landmarks

### Loss Function

```python
# CrossEntropy con label smoothing
loss = nn.CrossEntropyLoss(
    ignore_index=pad_token_id,
    label_smoothing=0.1
)
```

---

## 📈 Roadmap

### FASE 1: Setup (Settimana 1-2) ⏳

- [x] Analisi libreria sign-language-translator
- [x] Decisione architettura
- [x] Script di test feature extraction
- [ ] Installazione dipendenze
- [ ] Test su sample dataset

### FASE 2: Feature Extraction (Settimana 3-4)

- [ ] Implementare `sign_language_embeddings.py`
- [ ] Batch processing tutti i 202 video
- [ ] Salvataggio landmarks in formato efficiente
- [ ] Validazione qualità landmarks

### FASE 3: Modello Core (Settimana 5-8)

- [ ] Implementare tokenizer captions
- [ ] Implementare Seq2Seq model
- [ ] Implementare DataLoader
- [ ] Test forward pass

### FASE 4: Training (Settimana 9-11)

- [ ] Training loop con MLflow
- [ ] Hyperparameter tuning
- [ ] Curriculum learning
- [ ] Validation monitoring

### FASE 5: Integration (Settimana 12-13)

- [ ] Pipeline end-to-end
- [ ] API REST
- [ ] Demo Gradio

### FASE 6: Optimization (Settimana 14-15)

- [ ] Model quantization
- [ ] Benchmarking
- [ ] Documentazione finale

---

## 🧪 Testing

### Unit Tests (TODO)

```bash
pytest tests/test_landmark_extraction.py
pytest tests/test_model_architecture.py
pytest tests/test_tokenizer.py
```

### Integration Tests (TODO)

```bash
pytest tests/test_end_to_end_pipeline.py
```

### Performance Tests (TODO)

```bash
python benchmark_inference_speed.py
```

---

## 📚 Riferimenti

### Librerie Utilizzate

- **sign-language-translator:** https://github.com/sign-language-translator/sign-language-translator
- **MediaPipe:** https://google.github.io/mediapipe/solutions/holistic
- **Hugging Face Transformers:** https://huggingface.co/docs/transformers

### Paper Rilevanti

1. **"Sign Language Transformers"** (CVPR 2020)
2. **"Neural Sign Language Translation"** (CVPR 2018)
3. **"Continuous Sign Language Recognition"** (ICCV 2021)

### Dataset Pubblici (Reference)

- **PHOENIX-2014T:** German Sign Language
- **CSL-Daily:** Chinese Sign Language
- **WLASL:** Large-scale ASL (2000+ glosses)

---

## 🤝 Contributi

### Team

- **Ignazio Emanuele Picciche** - Sviluppo principale
- **GitHub Copilot** - Assistenza architettura e documentazione

### Supervisori

- TBD

---

## 📄 Licenza

Parte del progetto Improved_EmoSign_Thesis - Tesi Magistrale UCBM

---

## 🆘 Troubleshooting

### Problemi Comuni

**Q: MediaPipe non estrae landmarks da alcuni video**

```bash
# Verifica risoluzione video
ffprobe -v error -select_streams v:0 \
    -show_entries stream=width,height video.mp4

# Risoluzione minima: 480p
```

**Q: CUDA out of memory durante training**

```python
# Riduci batch size o usa gradient accumulation
trainer = Trainer(
    per_device_train_batch_size=4,  # ridotto da 16
    gradient_accumulation_steps=4   # accumula 4 batch
)
```

**Q: BLEU score molto basso**

```python
# Verifica tokenization consistency
# Train e test devono usare STESSO tokenizer
tokenizer = BPETokenizer.from_file("tokenizer.json")
```

---

## 📞 Contatti

Per domande o supporto:

- Issues: GitHub Issues del progetto
- Email: TBD
- Documentazione: `docs/`

---

**Ultimo aggiornamento:** 1 Novembre 2025  
**Status:** 🚧 In Development  
**Versione:** 0.1.0-alpha
