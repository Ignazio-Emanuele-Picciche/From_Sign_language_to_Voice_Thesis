# Video-to-Text Pipeline Roadmap

## Pipeline di Trascrizione ASL (American Sign Language)

**Data creazione:** 1 Novembre 2025  
**Obiettivo:** Costruire una pipeline che trascrivi video in lingua dei segni (ASL) in testo, con ground truth esistenti (video-testo-emozione)

---

## 📊 Analisi del Contesto Attuale

### Dataset Disponibili

Il progetto dispone già di dataset annotati con:

- **Video**: File `.mp4` di sign language
- **Caption/Testo**: Trascrizione del contenuto del video
- **Emozione**: Etichetta sentiment (Positive/Negative/Neutral)

Esempio dal file `golden_label_sentiment.csv`:

```csv
video_name,caption,emotion
83664512.mp4,"I was like, "Oh, wow, that's fine." He wanted to transfer...",Positive
25328.mp4,There are so many things that are wrong with the system.,Negative
```

### Struttura Attuale del Progetto

```
src/
├── models/
│   └── two_classes/vivit/     # Modelli per classificazione emozioni da video
├── tts/                        # Text-to-Speech per audio generation
├── data_pipeline/             # Pipeline di processamento dati
├── features/                   # Feature extraction
└── utils/                      # Utilities varie
```

---

## 🎯 Obiettivi della Pipeline

1. **Video → Text Translation**: Trascrizione automatica di video ASL in testo
2. **Validazione con Ground Truth**: Confronto con le caption già annotate
3. **Integrazione con modello esistente**: Collegamento con il modello ViViT per analisi emozioni
4. **Pipeline End-to-End**: Video → Testo → Emozione

---

## 📚 Analisi della Libreria sign-language-translator

### ✅ Caratteristiche Principali

**Punti di Forza:**

1. **Supporto Text-to-Sign (Consolidato)**

   - Modello rule-based `ConcatenativeSynthesis` già funzionante
   - Concatena video clips per ogni parola
   - Supporta Urdu, English, Hindi sign languages

2. **Feature Extraction Video**

   - `MediaPipeLandmarksModel`: estrae pose landmarks 3D/2D
   - Basato su MediaPipe (CNN-based)
   - Output: `(n_frames, n_landmarks * 5)` tensor

3. **Framework Modulare**
   - Classi base astratte per estensibilità
   - Supporto per video e landmarks come formati
   - Utilities per dataset processing

### ⚠️ Limitazioni Critiche

**Sign-to-Text NON È ANCORA DISPONIBILE:**

```python
# Dal README della libreria:
# # Load sign-to-text model (pytorch) (COMING SOON!)
# translation_model = slt.get_model(slt.ModelCodes.Gesture)
# text = translation_model.translate(embedding)
# print(text)
```

**Roadmap della libreria (v0.9+):**

- v0.9.2: sign to text with custom seq2seq transformer (FUTURO)
- v1.0.1: video to text model (FUTURO)

**Conseguenze per il nostro progetto:**

- ❌ Non possiamo usare direttamente la libreria per Video→Text
- ✅ Possiamo usare feature extraction (MediaPipe landmarks)
- ✅ Possiamo studiare l'architettura per implementare il nostro modello

---

## 🗺️ ROADMAP COMPLETA

### **FASE 1: Setup e Analisi** (Settimana 1-2)

#### 1.1 Installazione e Familiarizzazione

- [ ] Installare `sign-language-translator`
  ```bash
  pip install "sign-language-translator[all]"
  ```
- [ ] Testare feature extraction su video di esempio
- [ ] Documentare formato landmarks estratti
- [ ] Analizzare compatibilità con i nostri video ASL

#### 1.2 Analisi Dataset

- [ ] Verificare qualità delle caption esistenti
- [ ] Statistiche su lunghezza video vs lunghezza testo
- [ ] Analisi distribuzione vocabulary nelle caption
- [ ] Identificare subset per training/validation/test

**Deliverable:** `docs/sign_language_dataset_analysis.md`

---

### **FASE 2: Feature Extraction Pipeline** (Settimana 3-4)

#### 2.1 Estrazione Landmarks

- [ ] Creare script per processare batch di video
- [ ] Estrarre MediaPipe landmarks da tutti i video

  ```python
  import sign_language_translator as slt

  model = slt.models.MediaPipeLandmarksModel()
  embedding = model.embed(video.iter_frames(), landmark_type="world")
  # Output: (n_frames, 75, 5) -> pose + hands 3D coordinates
  ```

- [ ] Salvare embeddings in formato efficiente (HDF5 o pickle)
- [ ] Normalizzazione e preprocessing landmarks

#### 2.2 Integrazione con Dataset

- [ ] Creare nuova struttura dati:
  ```python
  {
      'video_name': str,
      'landmarks': np.array,  # (n_frames, 75, 5)
      'caption': str,
      'emotion': str,
      'video_duration': float,
      'n_frames': int
  }
  ```
- [ ] Salvare in `data/processed/sign_language_embeddings/`

**Deliverable:**

- `src/features/sign_language_embeddings.py`
- Dataset embeddings processato

---

### **FASE 3: Modello Sign-to-Text (Core)** (Settimana 5-8)

#### 3.1 Scelta Architettura

**Opzioni da valutare:**

**A. Seq2Seq Transformer (CONSIGLIATO)**

- Encoder: Processa sequenza landmarks
- Decoder: Genera sequenza text tokens
- Architettura simile a quella usata da sign-language-translator roadmap
- Pro: SOTA per sequence translation
- Contro: Richiede training da zero

**B. Fine-tuning modello pre-trained**

- Base: Whisper (audio→text) o simili
- Adattamento: landmarks → text
- Pro: Transfer learning
- Contro: Domain gap (audio vs landmarks)

**C. Hybrid approach**

- LSTM/GRU Encoder per landmarks
- Transformer Decoder con attention
- Pro: Bilanciamento efficienza/performance
- Contro: Complessità implementativa

#### 3.2 Implementazione Modello

**Struttura proposta (Seq2Seq Transformer):**

```python
class SignToTextModel(nn.Module):
    def __init__(self,
                 landmark_dim=375,      # 75 landmarks * 5 coords
                 d_model=512,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 vocab_size=10000):

        # Landmark encoder
        self.landmark_projection = nn.Linear(landmark_dim, d_model)
        self.temporal_encoder = nn.TransformerEncoder(...)

        # Text decoder
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder = nn.TransformerDecoder(...)
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, landmarks, text_tokens):
        # Encode landmarks sequence
        landmark_features = self.landmark_projection(landmarks)
        encoded = self.temporal_encoder(landmark_features)

        # Decode to text
        text_embeds = self.token_embedding(text_tokens)
        decoded = self.decoder(text_embeds, encoded)
        logits = self.output_projection(decoded)

        return logits
```

#### 3.3 Training Pipeline

- [ ] Tokenizer per caption (BPE o WordPiece)
- [ ] Data augmentation per landmarks
  - Temporal subsampling
  - Spatial rotation/scaling
  - Noise injection
- [ ] Loss function: CrossEntropy con label smoothing
- [ ] Metrics: BLEU, METEOR, WER (Word Error Rate)
- [ ] Integrazione con MLflow per tracking

**Deliverable:**

- `src/models/sign_to_text/`
  - `model.py`
  - `train.py`
  - `dataset.py`
  - `tokenizer.py`

---

### **FASE 4: Training e Validazione** (Settimana 9-11)

#### 4.1 Training Strategy

- [ ] Split dataset: 70% train, 15% val, 15% test
- [ ] Curriculum learning:
  - Prima: caption corte (< 10 parole)
  - Poi: caption medie (10-20 parole)
  - Infine: caption lunghe (> 20 parole)
- [ ] Early stopping su validation BLEU
- [ ] Gradient clipping per stabilità
- [ ] Learning rate scheduling (warm-up + cosine decay)

#### 4.2 Hyperparameter Tuning

- [ ] Grid search / Optuna per:
  - Learning rate: [1e-4, 5e-4, 1e-3]
  - Batch size: [8, 16, 32]
  - d_model: [256, 512]
  - num_layers: [4, 6, 8]
- [ ] Registrazione risultati in MLflow

#### 4.3 Evaluation

- [ ] Metriche quantitative:
  - BLEU-1, BLEU-2, BLEU-3, BLEU-4
  - METEOR
  - WER (Word Error Rate)
  - CER (Character Error Rate)
- [ ] Analisi qualitativa:
  - Esempi best/worst case
  - Error analysis per categoria
  - Confusion tra parole simili

**Deliverable:**

- `results/sign_to_text/`
  - Training logs
  - Best model checkpoint
  - Evaluation metrics
- `docs/sign_to_text_training_report.md`

---

### **FASE 5: Pipeline End-to-End** (Settimana 12-13)

#### 5.1 Integrazione Video→Text→Emotion

```python
class EndToEndPipeline:
    def __init__(self):
        # Feature extraction
        self.landmark_extractor = MediaPipeLandmarksModel()

        # Sign-to-text
        self.sign_to_text = SignToTextModel.load_checkpoint(...)

        # Emotion classifier (già esistente)
        self.emotion_classifier = ViViTModel.load_pretrained(...)

    def process(self, video_path):
        # 1. Extract landmarks
        landmarks = self.landmark_extractor.embed(video)

        # 2. Generate text
        text = self.sign_to_text.translate(landmarks)

        # 3. Predict emotion
        emotion = self.emotion_classifier.predict(video)

        return {
            'text': text,
            'emotion': emotion,
            'landmarks': landmarks
        }
```

#### 5.2 API e Interfaccia

- [ ] REST API con FastAPI
- [ ] Endpoint: POST /translate (video → text)
- [ ] Endpoint: POST /analyze (video → text + emotion)
- [ ] Gradio/Streamlit demo per testing visuale

**Deliverable:**

- `src/api/sign_language_api.py`
- `src/demo/sign_language_demo.py`

---

### **FASE 6: Ottimizzazione e Deployment** (Settimana 14-15)

#### 6.1 Ottimizzazioni

- [ ] Model quantization (FP16/INT8) per inference veloce
- [ ] Batching dinamico per throughput
- [ ] Caching landmarks pre-estratti
- [ ] ONNX export per portabilità

#### 6.2 Testing Produzione

- [ ] Test su golden labels set completo
- [ ] Benchmark latency/throughput
- [ ] Stress testing con video lunghi
- [ ] Edge cases e fallback strategies

#### 6.3 Documentazione

- [ ] User guide per la pipeline
- [ ] API documentation
- [ ] Model cards per reproducibility
- [ ] Jupyter notebooks con esempi

**Deliverable:**

- `docs/sign_language_pipeline_guide.md`
- `notebooks/sign_to_text_examples.ipynb`

---

## 🔄 Alternative e Plan B

### Opzione 1: Transfer Learning da ASR Models

Se il training da zero non converge:

- Fine-tune Whisper encoder sostituendo mel-spectrogram con landmarks
- Congelare decoder, adattare solo encoder
- Pro: Sfrutta conoscenze seq2seq già apprese
- Contro: Richiede adattamento architetturale

### Opzione 2: Utilizzo di modelli esterni

- **Considerare:** [Hugging Face sign-language models](https://huggingface.co/models?other=sign-language)
- Alcuni modelli pre-trained potrebbero esistere per ASL
- Validare su nostri ground truth

### Opzione 3: Rule-based approach iniziale

- Usare `sign-language-translator` al contrario:
  - Estrarre landmarks
  - Match con dizionario pre-esistente
  - Ricostruire frasi con regole grammaticali
- Pro: Baseline veloce
- Contro: Limitato a vocabolario noto

---

## 📊 Metriche di Successo

### MVP (Minimum Viable Product)

- [ ] BLEU-4 > 0.30 su test set
- [ ] WER < 40%
- [ ] Inference time < 5 secondi per video (30 sec durata)
- [ ] Accuracy emozione > 70% con testo generato

### Target Finale

- [ ] BLEU-4 > 0.45
- [ ] WER < 25%
- [ ] Inference time < 2 secondi
- [ ] Accuracy emozione comparabile a ground truth captions

---

## 🛠️ Stack Tecnologico Consigliato

### Core Libraries

```python
# Feature extraction
sign-language-translator[all]  # v0.8.1
mediapipe                      # Pose/Hand detection

# Deep Learning
torch>=2.0.0
transformers>=4.30.0
tokenizers

# Evaluation
sacrebleu                      # BLEU scores
evaluate                       # Hugging Face metrics
jiwer                         # WER/CER

# MLOps
mlflow                        # Già usato nel progetto
wandb (opzionale)            # Alternative tracking

# API/Demo
fastapi
gradio
streamlit
```

### Struttura Cartelle Proposta

```
src/
├── features/
│   ├── sign_language_embeddings.py    # NEW
│   └── landmark_preprocessing.py      # NEW
├── models/
│   ├── sign_to_text/                  # NEW
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── dataset.py
│   │   ├── tokenizer.py
│   │   └── evaluate.py
│   └── two_classes/vivit/             # EXISTING
├── api/
│   └── sign_language_api.py           # NEW
└── demo/
    └── sign_language_demo.py          # NEW

data/
├── processed/
│   ├── sign_language_embeddings/      # NEW
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── golden_label_sentiment.csv     # EXISTING

docs/
├── VIDEO_TO_TEXT_PIPELINE_ROADMAP.md  # THIS FILE
├── sign_language_dataset_analysis.md  # NEW
├── sign_to_text_training_report.md    # NEW
└── sign_language_pipeline_guide.md    # NEW
```

---

## ⚠️ Rischi e Mitigazioni

| Rischio                        | Probabilità | Impatto | Mitigazione                                          |
| ------------------------------ | ----------- | ------- | ---------------------------------------------------- |
| Dataset troppo piccolo per DL  | Media       | Alto    | Data augmentation aggressiva, Transfer learning      |
| Variabilità performers ASL     | Alta        | Alto    | Multi-person normalization, style-invariant features |
| Caption ground truth rumorose  | Media       | Medio   | Data cleaning, human validation subset               |
| Convergenza training difficile | Media       | Alto    | Curriculum learning, pre-training componenti         |
| Latency inference troppo alta  | Bassa       | Medio   | Model distillation, quantization                     |

---

## 📅 Timeline Riassuntiva

| Fase               | Durata           | Settimane     | Output Chiave           |
| ------------------ | ---------------- | ------------- | ----------------------- |
| Setup & Analisi    | 2 settimane      | 1-2           | Dataset analysis        |
| Feature Extraction | 2 settimane      | 3-4           | Landmarks embeddings    |
| Modello Core       | 4 settimane      | 5-8           | Sign-to-text model      |
| Training           | 3 settimane      | 9-11          | Trained model + metrics |
| Integration        | 2 settimane      | 12-13         | End-to-end pipeline     |
| Optimization       | 2 settimane      | 14-15         | Production-ready system |
| **TOTALE**         | **15 settimane** | **~3.5 mesi** | **Complete pipeline**   |

---

## 🎓 Collegamento con Tesi

### Contributi Scientifici

1. **Multimodal Emotion Recognition**

   - Video (ViViT) + Text (from sign) + Audio (TTS)
   - Analisi consistenza emozioni cross-modal

2. **Sign Language NLP**

   - Primo sistema integrato ASL→Text→Emotion
   - Benchmark su dataset annotato custom

3. **Explainability**
   - Attention visualization: quali segni → quali parole
   - Feature importance: landmark regions più significativi

### Capitoli Tesi Possibili

1. **Background:** Sign Language Recognition & Translation
2. **Methodology:** Seq2Seq Architecture for ASL-to-Text
3. **Integration:** Multi-modal Emotion Analysis
4. **Experiments:** Benchmark su golden labels
5. **Results:** Ablation studies & qualitative analysis
6. **Discussion:** Limitations & future work

---

## 📖 Risorse e Riferimenti

### Paper Rilevanti

- **Sign Language Recognition:**

  - "Sign Language Transformers: Joint End-to-end Sign Language Recognition and Translation" (CVPR 2020)
  - "Continuous Sign Language Recognition with Temporal Graph" (ICCV 2021)

- **Seq2Seq for SL:**
  - "Neural Sign Language Translation" (CVPR 2018)
  - "Progressive Transformers for End-to-End Sign Language Production" (ECCV 2020)

### Dataset Pubblici (per reference)

- **PHOENIX-2014T**: German Sign Language with translations
- **CSL-Daily**: Chinese Sign Language dataset
- **WLASL**: Large-scale ASL video dataset (2000+ glosses)

### Community & Tools

- [sign-language-translator GitHub](https://github.com/sign-language-translator/sign-language-translator)
- [Hugging Face Sign Language Hub](https://huggingface.co/models?other=sign-language)
- [MediaPipe Holistic](https://google.github.io/mediapipe/solutions/holistic)

---

## ✅ Next Immediate Steps

### Da fare OGGI/QUESTA SETTIMANA:

1. [ ] Installare `sign-language-translator[all]`
2. [ ] Testare su 1-2 video di esempio l'estrazione landmarks
3. [ ] Analizzare qualità caption in `golden_label_sentiment.csv`
4. [ ] Cercare paper su ASL-to-text translation per SOTA
5. [ ] Definire architettura modello preliminare

### Script di Test Rapido:

```python
# test_sign_extraction.py
import sign_language_translator as slt

# Load sample video
video_path = "data/raw/ASLLRP/videos/83664512.mp4"
video = slt.Video(video_path)

# Extract landmarks
model = slt.models.MediaPipeLandmarksModel()
landmarks = model.embed(video.iter_frames(), landmark_type="world")

print(f"Video frames: {len(video)}")
print(f"Landmarks shape: {landmarks.shape}")
print(f"Expected caption: 'I was like, Oh, wow, that's fine...'")

# Visualize
slt.Landmarks(landmarks.reshape((-1, 75, 5)),
              connections="mediapipe-world").show()
```

---

## 🤝 Conclusioni e Raccomandazioni

### ✅ Conviene usare sign-language-translator?

**SÌ, ma con cautela:**

- ✅ **Feature extraction**: MediaPipe landmarks funziona bene
- ✅ **Framework di riferimento**: Architettura modulare da studiare
- ✅ **Utilities**: Dataset processing, visualization tools
- ❌ **Sign-to-text diretto**: NON disponibile, dobbiamo implementarlo noi

### Strategia Consigliata:

1. **Short-term (Fase 1-2)**: Usare SLT per feature extraction
2. **Mid-term (Fase 3-4)**: Implementare nostro modello Seq2Seq
3. **Long-term (Fase 5-6)**: Pipeline completa integrata con ViViT

### Alternative se SLT non soddisfa:

- MediaPipe direttamente (senza SLT wrapper)
- OpenPose per landmarks più dettagliati
- Modelli pre-trained da Hugging Face (se esistenti per ASL)

---

**Autore:** GitHub Copilot  
**Versione:** 1.0  
**Ultimo aggiornamento:** 1 Novembre 2025  
**Status:** DRAFT - Da validare con supervisore tesi
