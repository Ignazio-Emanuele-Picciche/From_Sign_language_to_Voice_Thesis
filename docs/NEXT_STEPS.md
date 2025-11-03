# 📋 PROSSIMI PASSI - Video-to-Text Pipeline

**Data:** 1 Novembre 2025  
**Status:** ✅ Feature Extraction FUNZIONANTE

---

## ✅ COMPLETATO

### 1. Installazione ✅

- [x] Installata `sign-language-translator[all]` v0.8.1
- [x] MediaPipe landmarks model scaricato
- [x] Tutte le dipendenze funzionanti

### 2. Test Feature Extraction ✅

- [x] Test singolo video: **SUCCESS** (83664512.mp4)
- [x] Test batch 3 video: **SUCCESS** (100% successo)
- [x] Landmarks salvati in `results/sign_language_test/*.npz`

### 3. Validazione Output ✅

```
✓ Shape landmarks: (n_frames, 375)  # 75 landmarks × 5 coordinates
✓ Range valori: [-0.6942, 1.0000]
✓ Mean: ~0.10, Std: ~0.38
✓ Tutti i 4 video processati correttamente
```

---

## 📊 Statistiche Test

| Video        | Frames | Durata | Size File | Caption                                     |
| ------------ | ------ | ------ | --------- | ------------------------------------------- |
| 83664512.mp4 | 221    | 7.37s  | 393 KB    | "I was like, 'Oh, wow, that's fine.'..."    |
| 7983954.mp4  | 85     | 2.84s  | 169 KB    | "My friends crashed the party."             |
| 256351.mp4   | 204    | 6.81s  | 390 KB    | "A Deaf role model who has been trained..." |
| 5596228.mp4  | 67     | 2.24s  | 128 KB    | "Why is dad upset? I have no idea..."       |

**Media:** 118.7 frames/video, 3.96 secondi/video

---

## 🎯 PROSSIMI STEP (IN ORDINE)

### STEP 1: Analisi Dataset Completo (QUESTA SETTIMANA)

**Obiettivo:** Capire i dati prima di costruire il modello

#### 1.1 Analisi Caption

```bash
# TODO: Creare script
python analyze_captions.py
```

**Domande da rispondere:**

- [ ] Quante parole uniche nel vocabulary?
- [ ] Distribuzione lunghezza caption (min/max/media)?
- [ ] Parole più frequenti?
- [ ] Differenze tra Positive vs Negative?

#### 1.2 Analisi Landmarks

```bash
# TODO: Creare script
python analyze_landmarks.py
```

**Domande da rispondere:**

- [ ] Qualità landmarks su tutti i 200 video?
- [ ] Ci sono video con landmarks mancanti/problematici?
- [ ] Variabilità durata video?
- [ ] Correlazione frame count vs caption length?

#### 1.3 Data Cleaning

- [ ] Identificare video problematici (se esistono)
- [ ] Creare train/val/test split (70/15/15)
- [ ] Documentare statistiche finali

**Output:** `docs/DATASET_ANALYSIS.md`

---

### STEP 2: Estrazione Landmarks Completa (SETTIMANA PROSSIMA)

**Obiettivo:** Processare TUTTI i 200 video

```bash
# TODO: Creare script ottimizzato
python extract_all_landmarks.py \
    --csv_path data/processed/golden_label_sentiment.csv \
    --output_dir data/processed/sign_language_embeddings \
    --batch_size 10
```

**Tasks:**

- [ ] Script batch processing ottimizzato
- [ ] Progress bar (tqdm)
- [ ] Error handling robusto
- [ ] Salvataggio incrementale
- [ ] Log processamento

**Output:**

```
data/processed/sign_language_embeddings/
├── all_landmarks.h5          # HDF5 per efficienza
├── train_split.json
├── val_split.json
├── test_split.json
└── processing_log.txt
```

---

### STEP 3: Tokenizer Caption (SETTIMANA PROSSIMA)

**Obiettivo:** Preparare testo per il modello

```python
# TODO: Implementare
from tokenizers import BertWordPieceTokenizer

tokenizer = BertWordPieceTokenizer()
tokenizer.train(
    files=["captions.txt"],
    vocab_size=5000,  # Da determinare da analisi
    min_frequency=2
)
```

**Tasks:**

- [ ] Implementare `src/models/sign_to_text/tokenizer.py`
- [ ] Training tokenizer su caption
- [ ] Salvare vocabulary
- [ ] Test encoding/decoding

**Output:**

- `models/sign_to_text/tokenizer.json`
- `models/sign_to_text/vocab.txt`

---

### STEP 4: Dataset Loader (2 SETTIMANE)

**Obiettivo:** PyTorch DataLoader per training

```python
# TODO: Implementare
class SignLanguageDataset(Dataset):
    def __init__(self, landmarks_file, captions_file, tokenizer):
        ...

    def __getitem__(self, idx):
        landmarks = self.landmarks[idx]  # (n_frames, 375)
        caption = self.captions[idx]
        tokens = self.tokenizer.encode(caption)

        return {
            'landmarks': landmarks,
            'input_ids': tokens,
            'labels': tokens[1:],  # Shifted for teacher forcing
            'attention_mask': ...
        }
```

**Tasks:**

- [ ] Implementare Dataset class
- [ ] Padding/truncation landmarks
- [ ] Collate function per batching
- [ ] Data augmentation (temporal/spatial)
- [ ] Unit tests

**Output:** `src/models/sign_to_text/dataset.py`

---

### STEP 5: Modello Seq2Seq (3 SETTIMANE)

**Obiettivo:** Architettura Transformer

```python
# TODO: Implementare
class SignToTextModel(nn.Module):
    def __init__(self):
        # Encoder
        self.landmark_projection = nn.Linear(375, 512)
        self.encoder = TransformerEncoder(...)

        # Decoder
        self.decoder = TransformerDecoder(...)

    def forward(self, landmarks, text_ids):
        # Encode landmarks
        encoded = self.encoder(landmarks)

        # Decode to text
        decoded = self.decoder(text_ids, encoded)

        return decoded
```

**Tasks:**

- [ ] Implementare architettura base
- [ ] Test forward pass
- [ ] Gradient flow check
- [ ] Loss function setup
- [ ] Optimizer configuration

**Output:** `src/models/sign_to_text/model.py`

---

### STEP 6: Training Pipeline (2 SETTIMANE)

**Obiettivo:** Training loop + MLflow

```python
# TODO: Implementare
def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    for batch in tqdm(dataloader):
        ...

def train_model(config):
    # MLflow tracking
    mlflow.start_run()
    mlflow.log_params(config)

    for epoch in range(epochs):
        train_loss = train_epoch(...)
        val_loss = validate_epoch(...)

        mlflow.log_metrics({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'bleu': ...,
        }, step=epoch)
```

**Tasks:**

- [ ] Training loop implementation
- [ ] Validation loop
- [ ] MLflow integration
- [ ] Checkpoint saving
- [ ] Early stopping
- [ ] Learning rate scheduling

**Output:** `src/models/sign_to_text/train.py`

---

### STEP 7: Evaluation (1 SETTIMANA)

**Obiettivo:** Metriche quantitative

```python
# TODO: Implementare
def evaluate_model(model, test_loader):
    predictions = []
    references = []

    for batch in test_loader:
        pred = model.generate(batch['landmarks'])
        predictions.append(pred)
        references.append(batch['caption'])

    bleu = compute_bleu(predictions, references)
    wer = compute_wer(predictions, references)

    return {'bleu': bleu, 'wer': wer}
```

**Tasks:**

- [ ] BLEU-1, BLEU-2, BLEU-3, BLEU-4
- [ ] WER (Word Error Rate)
- [ ] CER (Character Error Rate)
- [ ] Qualitative analysis
- [ ] Error analysis per categoria

**Output:** `src/models/sign_to_text/evaluate.py`

---

### STEP 8: Integrazione ViViT (1 SETTIMANA)

**Obiettivo:** Pipeline end-to-end

```python
# TODO: Implementare
class EndToEndPipeline:
    def __init__(self):
        self.landmark_extractor = MediaPipeLandmarksModel()
        self.sign_to_text = SignToTextModel.load_checkpoint(...)
        self.emotion_classifier = ViViTModel.load_pretrained(...)

    def process(self, video_path):
        # 1. Extract landmarks
        landmarks = self.landmark_extractor.extract(video_path)

        # 2. Generate caption
        caption = self.sign_to_text.translate(landmarks)

        # 3. Predict emotion
        emotion = self.emotion_classifier.predict(video_path)

        return {
            'caption': caption,
            'emotion': emotion,
            'confidence': ...
        }
```

**Tasks:**

- [ ] Pipeline class implementation
- [ ] End-to-end testing
- [ ] Performance benchmarking
- [ ] API REST (FastAPI)
- [ ] Demo Gradio

**Output:** `src/api/sign_language_pipeline.py`

---

## 📅 TIMELINE DETTAGLIATA

| Settimana      | Tasks                                    | Deliverable                                |
| -------------- | ---------------------------------------- | ------------------------------------------ |
| **1** (Questa) | Analisi dataset, Data cleaning           | `docs/DATASET_ANALYSIS.md`                 |
| **2**          | Estrazione landmarks completa, Tokenizer | `data/processed/sign_language_embeddings/` |
| **3-4**        | Dataset loader, Data augmentation        | `src/models/sign_to_text/dataset.py`       |
| **5-7**        | Modello Seq2Seq, Test architettura       | `src/models/sign_to_text/model.py`         |
| **8-9**        | Training pipeline, MLflow setup          | `src/models/sign_to_text/train.py`         |
| **10-11**      | Training, Hyperparameter tuning          | Best model checkpoint                      |
| **12**         | Evaluation, Metrics                      | `src/models/sign_to_text/evaluate.py`      |
| **13**         | Integrazione ViViT, API                  | `src/api/sign_language_pipeline.py`        |
| **14-15**      | Optimization, Documentation              | Sistema completo                           |

**Totale:** 15 settimane (~3.5 mesi)

---

## 🛠️ TOOLS DA INSTALLARE (PROSSIMAMENTE)

```bash
# Quando necessario, installare:

# Per tokenizer
pip install tokenizers

# Per evaluation
pip install sacrebleu
pip install jiwer

# Per API (opzionale)
pip install fastapi uvicorn
pip install gradio
```

---

## 📖 LETTURE CONSIGLIATE (QUESTA SETTIMANA)

### Paper da Leggere

1. **"Sign Language Transformers"** (CVPR 2020)

   - https://arxiv.org/abs/2003.13830
   - Focus: Architettura seq2seq per sign language

2. **"Neural Sign Language Translation"** (CVPR 2018)

   - https://arxiv.org/abs/1801.05704
   - Focus: Baseline approach

3. **"Attention Is All You Need"** (NIPS 2017)
   - https://arxiv.org/abs/1706.03762
   - Focus: Transformer architecture

### Code da Studiare

1. **Hugging Face Transformers**

   - https://github.com/huggingface/transformers
   - Esempio: `BertForSeq2Seq`, `T5ForConditionalGeneration`

2. **sign-language-translator source**
   - https://github.com/sign-language-translator/sign-language-translator
   - File: `models/text_to_sign/concatenative_synthesis.py`

---

## 🎯 OBIETTIVI SETTIMANALI

### Questa Settimana (1-7 Nov)

- [x] Installare sign-language-translator ✅
- [x] Testare feature extraction ✅
- [x] Validare output landmarks ✅
- [ ] Analizzare dataset caption
- [ ] Creare split train/val/test
- [ ] Leggere 1-2 paper
- [ ] Discussione con supervisore

### Prossima Settimana (8-14 Nov)

- [ ] Estrarre landmarks da tutti i 200 video
- [ ] Implementare tokenizer
- [ ] Iniziare dataset loader
- [ ] Test preliminary model architecture

---

## 🆘 PROBLEMI NOTI & SOLUZIONI

### ⚠️ Conflitto Versioni Torch

```
torchaudio 2.9.0 requires torch==2.9.0, but you have torch 2.2.2
torchvision 0.24.0 requires torch==2.9.0, but you have torch 2.2.2
```

**Soluzione (se necessario):**

```bash
# Opzione 1: Upgrade torch (rischio compatibilità ViViT)
pip install torch==2.9.0

# Opzione 2: Downgrade torchaudio/torchvision
pip install torchaudio==2.2.0 torchvision==0.19.0

# Opzione 3: Ignore (funziona per ora per feature extraction)
# Decidere quando arriveremo al training
```

**Status:** ⏸️ Posticipato - funziona per ora

---

## 📞 CONTATTI & RISORSE

### Documentazione

- Roadmap completa: `docs/VIDEO_TO_TEXT_PIPELINE_ROADMAP.md`
- Quick start: `docs/QUICKSTART_SIGN_TO_TEXT.md`
- Decisione libreria: `docs/SIGN_LANGUAGE_TRANSLATOR_DECISION.md`

### Script Utili

- Test: `test_sign_language_extraction.py`
- Future: `extract_all_landmarks.py` (da creare)
- Future: `analyze_dataset.py` (da creare)

### Links

- sign-language-translator: https://github.com/sign-language-translator/sign-language-translator
- MediaPipe: https://google.github.io/mediapipe/solutions/holistic
- Hugging Face: https://huggingface.co/docs/transformers

---

## ✅ CHECKLIST RAPIDA

Prima di passare allo step successivo, verifica:

**Ora (Setup completato):**

- [x] sign-language-translator installato
- [x] MediaPipe funzionante
- [x] Test su video singolo OK
- [x] Test su batch OK
- [x] Landmarks salvati correttamente

**Questa settimana (Analisi):**

- [ ] Analisi caption completata
- [ ] Dataset split creato
- [ ] Statistiche documentate
- [ ] Paper letti

**Prossima settimana (Feature extraction):**

- [ ] Tutti i 200 video processati
- [ ] Tokenizer implementato
- [ ] HDF5 dataset creato

---

**Ultimo aggiornamento:** 1 Novembre 2025, ore 11:40  
**Status generale:** ✅ Setup completato, pronti per Fase 1  
**Prossimo milestone:** Analisi dataset completa

---

## 🎉 CONGRATULAZIONI!

Hai completato con successo il setup iniziale! La libreria funziona perfettamente e i landmarks vengono estratti correttamente.

**Ora sei pronto per:**

1. Analizzare il dataset in profondità
2. Capire i pattern nei dati
3. Progettare il modello in modo informato

**Vai avanti con fiducia!** 🚀
