# Improved_EmoSign_Thesis

Sistema completo per traduzione da Lingua dei Segni Americana (ASL) a testo con sintesi vocale emotiva.

## 📋 Moduli Principali

### 1. 🤟 Sign-to-Text Translation

#### **Seq2Seq Transformer** (Landmarks-based)

- 📂 `src/sign_to_text/`
- 📝 Documentazione: [`src/sign_to_text/README.md`](src/sign_to_text/README.md)
- Input: OpenPose landmarks (411 features)
- Performance: BLEU-4 ~25-30% (target)
- Vantaggi: Interpretabile, veloce (35 fps), leggero (4GB VRAM)

#### **SSVP-SLT** (Video-based, SOTA) ⭐ NEW

- 📂 `src/sign_to_text_ssvp/`
- 📝 Documentazione: [`src/sign_to_text_ssvp/README.md`](src/sign_to_text_ssvp/README.md)
- 🚀 Quick Start: [`src/sign_to_text_ssvp/docs/QUICKSTART.md`](src/sign_to_text_ssvp/docs/QUICKSTART.md)
- 📊 Integration Guide: [`docs/SSVP_SLT_INTEGRATION.md`](docs/SSVP_SLT_INTEGRATION.md)
- Input: Video frames RGB
- Performance: BLEU-4 ~38-40% (SOTA)
- Vantaggi: State-of-the-art, pretraining self-supervised

### 2. 🎭 Emotion Analysis

- 📂 `src/emotion_analysis/`
- Analisi emozioni da testo tradotto

### 3. 🔊 Text-to-Speech (TTS)

- 📂 `src/tts/`
- 📝 Documentazione: [`docs/BARK_TTS_PIPELINE.md`](docs/BARK_TTS_PIPELINE.md)
- Sintesi vocale con prosody emotiva (Bark TTS)

## 🚀 Quick Start

### Setup Sign-to-Text (Seq2Seq)

```bash
# Training modello Seq2Seq Transformer
python src/sign_to_text/train_how2sign.py --epochs 30
```

### Setup SSVP-SLT (NEW)

```bash
cd src/sign_to_text_ssvp

# 1. Installazione
bash scripts/install_ssvp.sh

# 2. Download pretrained model
python download_pretrained.py --model base

# 3. Preparazione dataset
bash scripts/prepare_all_splits.sh

# 4. Fine-tuning
python finetune_how2sign.py --config configs/finetune_base.yaml
```

## 📊 Model Comparison

| Feature              | Seq2Seq (Ours) | SSVP-SLT (SOTA) |
| -------------------- | -------------- | --------------- |
| **BLEU-4**           | 25-30%         | **38-40%** ✅   |
| **Speed**            | **35 fps** ✅  | 12 fps          |
| **Memory**           | **4GB** ✅     | 8-16GB          |
| **Interpretability** | **High** ✅    | Low             |
| **Robustness**       | Medium         | **High** ✅     |

## 📚 Documentation

- **Sign-to-Text Seq2Seq**: [`src/sign_to_text/README.md`](src/sign_to_text/README.md)
- **SSVP-SLT Integration**: [`docs/SSVP_SLT_INTEGRATION.md`](docs/SSVP_SLT_INTEGRATION.md)
- **SSVP-SLT Quick Start**: [`src/sign_to_text_ssvp/docs/QUICKSTART.md`](src/sign_to_text_ssvp/docs/QUICKSTART.md)
- **TTS Pipeline**: [`docs/BARK_TTS_PIPELINE.md`](docs/BARK_TTS_PIPELINE.md)
- **How2Sign Setup**: [`docs/HOW2SIGN_SETUP_COMPLETE.md`](docs/HOW2SIGN_SETUP_COMPLETE.md)

## 🎯 Project Structure

```
Improved_EmoSign_Thesis/
├── src/
│   ├── sign_to_text/           # Seq2Seq Transformer (landmarks)
│   ├── sign_to_text_ssvp/      # SSVP-SLT (video) ⭐ NEW
│   ├── emotion_analysis/       # Emotion detection
│   └── tts/                    # Text-to-Speech
├── data/
│   ├── raw/how2sign/          # How2Sign dataset
│   └── how2sign_ssvp/         # How2Sign formato SSVP-SLT
├── docs/                       # Documentazione
├── results/                    # Output training
└── models/                     # Checkpoints modelli
```

## 🔬 Research Contributions

1. **Sign-to-Text Translation**

   - Seq2Seq Transformer con landmarks OpenPose
   - SSVP-SLT integration per benchmark SOTA
   - Comparison landmarks vs video approaches

2. **Emotion-Aware TTS**

   - Prosody optimization con Bark
   - Emotional tag system

3. **End-to-End Pipeline**
   - ASL Video → Text → Emotion → Voice

## 📄 License

- **Tesi**: Ignazio Emanuele Picciche
- **SSVP-SLT**: CC-BY-NC 4.0 (Facebook Research)
