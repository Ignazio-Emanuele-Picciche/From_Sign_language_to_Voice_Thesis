# Quick Start: Video-to-Text Pipeline

## 🎯 Obiettivo

Costruire una pipeline che traduce video ASL (American Sign Language) in testo, usando i ground truth esistenti (video-testo-emozione) per training e validazione.

---

## 📋 Risposta alla tua domanda: "Ha senso usare sign-language-translator?"

### ✅ **SÌ, con alcune precisazioni importanti**

**COSA POSSIAMO USARE:**

- ✅ **Feature extraction** (MediaPipe landmarks) - FUNZIONA BENE
- ✅ **Video processing utilities** - UTILE
- ✅ **Framework di riferimento** - ARCHITETTURA DA STUDIARE

**COSA NON POSSIAMO USARE:**

- ❌ **Sign-to-Text diretto** - NON ANCORA IMPLEMENTATO nella libreria
- ❌ **Modelli pre-trained ASL→Text** - NON DISPONIBILI

### 🔍 Analisi della libreria

Dal README ufficiale:

```python
# # Load sign-to-text model (pytorch) (COMING SOON!)
# translation_model = slt.get_model(slt.ModelCodes.Gesture)
# text = translation_model.translate(embedding)
```

**Roadmap futura della libreria:**

- v0.9.2: sign-to-text con seq2seq transformer ⏰ FUTURO
- v1.0.1: video-to-text model ⏰ FUTURO

**Quello che è DISPONIBILE ORA:**

- Text → Sign Language ✅ (opposto di quello che ci serve)
- Feature extraction (MediaPipe) ✅
- Video utilities ✅

---

## 🗺️ Strategia Consigliata

### 1. **Usare sign-language-translator per Feature Extraction**

```python
import sign_language_translator as slt

# Estrarre landmarks 3D dai video
model = slt.models.MediaPipeLandmarksModel()
landmarks = model.embed(video.iter_frames(), landmark_type="world")
# Output: (n_frames, 75 landmarks * 5 coordinates)
```

### 2. **Implementare NOSTRO modello Sign-to-Text**

La libreria non ce lo fornisce, quindi dobbiamo costruirlo noi:

- Architettura: **Seq2Seq Transformer**
- Input: Landmarks estratti
- Output: Testo (caption)
- Training: I tuoi ground truth (video-caption-emotion)

### 3. **Integrare con il modello esistente (ViViT)**

Pipeline completa:

```
Video → Landmarks → Testo → Emozione
         (SLT)      (NOSTRO)  (ViViT)
```

---

## 🚀 Primi Passi IMMEDIATI

### Passo 1: Installare la libreria

```bash
cd "/Users/ignazioemanuelepicciche/Documents/TESI Magistrale UCBM/Improved_EmoSign_Thesis"
source .venv/bin/activate
pip install "sign-language-translator[all]"
```

### Passo 2: Testare Feature Extraction

```bash
# Test su un singolo video
python test_sign_language_extraction.py \
    --mode single \
    --video_path data/raw/ASLLRP/videos/83664512.mp4

# Test su batch di 5 video
python test_sign_language_extraction.py \
    --mode batch \
    --n_samples 5
```

### Passo 3: Verificare Output

Controlla:

- `results/sign_language_test/*.npz` - Landmarks estratti
- Console output - Statistiche e shape

---

## 📊 I tuoi Dati Attuali

Dal file `golden_label_sentiment.csv`:

```csv
video_name,caption,emotion
83664512.mp4,"I was like, 'Oh, wow, that's fine.'...",Positive
25328.mp4,There are so many things that are wrong...,Negative
```

**Hai 202 video annotati con:**

- ✅ Video file (.mp4)
- ✅ Caption (testo trascritto)
- ✅ Emotion (Positive/Negative)

**Perfetto per:**

- Training modello Seq2Seq (landmarks → text)
- Validazione con BLEU/WER metrics
- Testing pipeline end-to-end

---

## 🏗️ Architettura Proposta

### Componenti da Costruire

```python
# 1. Feature Extractor (USARE sign-language-translator)
class LandmarkExtractor:
    def __init__(self):
        self.model = slt.models.MediaPipeLandmarksModel()

    def extract(self, video_path):
        video = slt.Video(video_path)
        return self.model.embed(video.iter_frames())

# 2. Sign-to-Text Model (DA IMPLEMENTARE)
class SignToTextModel(nn.Module):
    def __init__(self, landmark_dim=375, vocab_size=10000):
        # Transformer Encoder per landmarks
        self.encoder = TransformerEncoder(...)

        # Transformer Decoder per testo
        self.decoder = TransformerDecoder(...)

    def forward(self, landmarks, text_tokens):
        # Encode landmarks → Decode text
        ...

# 3. Pipeline Completa
class SignLanguagePipeline:
    def __init__(self):
        self.landmark_extractor = LandmarkExtractor()
        self.sign_to_text = SignToTextModel.load_checkpoint(...)
        self.emotion_classifier = ViViTModel.load_pretrained(...)

    def process(self, video_path):
        landmarks = self.landmark_extractor.extract(video_path)
        text = self.sign_to_text.translate(landmarks)
        emotion = self.emotion_classifier.predict(video_path)
        return {'text': text, 'emotion': emotion}
```

---

## 📅 Timeline Semplificata

| Settimana | Task                            | Output                |
| --------- | ------------------------------- | --------------------- |
| **1-2**   | Setup + Test feature extraction | Landmarks dataset     |
| **3-4**   | Dataset preparation             | Train/val/test splits |
| **5-8**   | Implementare Seq2Seq model      | Sign-to-text model    |
| **9-11**  | Training + tuning               | Trained checkpoint    |
| **12-13** | Integrazione con ViViT          | Pipeline E2E          |
| **14-15** | Testing + ottimizzazione        | Final system          |

**Totale: ~3.5 mesi**

---

## 📚 Prossime Letture

### Paper da Leggere (in ordine di priorità)

1. **"Sign Language Transformers"** (CVPR 2020)
   - Architettura seq2seq per sign language
2. **"Neural Sign Language Translation"** (CVPR 2018)
   - Baseline approach
3. **"Continuous Sign Language Recognition"** (ICCV 2021)
   - Temporal modeling

### Codice da Studiare

1. Architettura di `sign-language-translator`:
   - `sign_language_translator/models/text_to_sign/`
   - Capire come funziona il reverse (text→sign)
2. MediaPipe Holistic:
   - Come vengono estratti i landmarks
   - Normalizzazione e preprocessing

---

## ⚠️ Considerazioni Importanti

### Vantaggi di questa Strategia

✅ Flessibilità totale sull'architettura del modello  
✅ Controllo completo su training e optimization  
✅ Adattamento perfetto ai tuoi dati ASL  
✅ Contributo originale per la tesi

### Svantaggi/Sfide

⚠️ Più lavoro implementativo (no modello pre-trained)  
⚠️ Necessità di tuning hyperparameters  
⚠️ Dataset potrebbe essere piccolo (202 video)  
⚠️ Richiede competenze deep learning seq2seq

### Mitigazioni

- **Dataset piccolo**: Data augmentation + transfer learning
- **Implementazione**: Usare Hugging Face Transformers library
- **Baseline**: Implementare rule-based prima del DL

---

## 🎯 Decisione Finale

**RACCOMANDAZIONE:**

1. ✅ **USA sign-language-translator** per:

   - Estrarre landmarks MediaPipe
   - Studiare architettura e best practices
   - Utilities di preprocessing

2. ✅ **IMPLEMENTA TUO modello** per:

   - Sign-to-text translation (core contribution)
   - Fine su tuoi dati ASL specifici
   - Integrazione con pipeline esistente

3. ✅ **INTEGRA con ViViT** per:
   - Pipeline completa Video → Text → Emotion
   - Analisi multi-modale
   - Contributo scientifico originale

---

## 🔧 Tools & Libraries Stack

```python
# Feature extraction
sign-language-translator[all]  # MediaPipe wrapper

# Deep learning
torch>=2.0.0
transformers                   # Pre-built Transformer blocks
tokenizers                     # Fast BPE tokenizer

# Evaluation
sacrebleu                      # BLEU metric
jiwer                          # WER/CER metrics

# Existing in your project
mlflow                         # Experiment tracking
```

---

## 💡 Quick Commands

```bash
# Install library
pip install "sign-language-translator[all]"

# Test single video
python test_sign_language_extraction.py \
    --mode single \
    --video_path data/raw/ASLLRP/videos/83664512.mp4

# Test batch
python test_sign_language_extraction.py \
    --mode batch \
    --n_samples 10

# Check results
ls -lh results/sign_language_test/
```

---

## 📖 Documentazione Completa

Per roadmap dettagliata: `docs/VIDEO_TO_TEXT_PIPELINE_ROADMAP.md`

Per domande/dubbi:

- Sign-language-translator docs: https://slt.readthedocs.io
- GitHub repo: https://github.com/sign-language-translator/sign-language-translator
- MediaPipe: https://google.github.io/mediapipe/solutions/holistic

---

**Conclusione:** SÌ, ha senso usare `sign-language-translator`, ma come **tool di feature extraction**, non come soluzione completa. Il modello Sign-to-Text lo implementiamo noi. 🎯
