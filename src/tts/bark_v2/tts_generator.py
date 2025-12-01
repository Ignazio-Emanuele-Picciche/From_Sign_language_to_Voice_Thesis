"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           TTS GENERATOR - MOTORE DI SINTESI VOCALE EMOTIVA (PROD)            ║
╚══════════════════════════════════════════════════════════════════════════════╝

📋 DESCRIZIONE:
    Modulo principale per la generazione massiva (Batch Processing) di audio espressivo
    utilizzando il modello generativo Bark. Questo script rappresenta l'ultimo stadio
    della pipeline multimodale "EmoSign".

    Il generatore funge da orchestratore tra le predizioni del Meta-Learner e la
    sintesi vocale, implementando logiche avanzate di gestione delle risorse hardware
    per massimizzare la velocità su GPU A100.

🔄 FLUSSO DI LAVORO (PIPELINE):
    1. CARICAMENTO DATI ROBUSTO:
       - Legge il file delle predizioni (`final_predictions_with_captions.csv`).
       - Verifica l'integrità dei dati e la presenza delle caption.

    2. OTTIMIZZAZIONE HARDWARE (A100 MODE):
       - Configura PyTorch per utilizzare TF32 (TensorFloat-32) per calcoli matriciali veloci.
       - Disabilita l'offload su CPU per mantenere l'intero modello (Text, Coarse, Fine)
         nella VRAM da 80GB, eliminando i colli di bottiglia del trasferimento dati.

    3. LOGICA DI SINTESI CONTEXT-AWARE:
       - Per ogni video, invoca i moduli ausiliari (`emotion_mapper`, `emotion_tag_optimizer`)
         per determinare speaker, prosodia e tag emotivi ottimali.
       - Gestisce il troncamento intelligente dei testi troppo lunghi per evitare
         allucinazioni del modello Bark.

    4. GESTIONE MEMORIA:
       - Monitora l'uso della VRAM e forza la Garbage Collection ciclica per prevenire
         memory leak durante la generazione di migliaia di file.

📂 INPUT:
    - File CSV pre-processato con colonne: video_name, predicted_label, confidence, caption.

📂 OUTPUT:
    - File .wav salvati in `output_audio/`, nominati univocamente per video ed emozione.

⚠️ NOTE TECNICHE:
    - Richiede GPU con >24GB VRAM per la modalità "High Performance".
    - Utilizza modelli Bark "Large" per la massima fedeltà acustica ed emotiva.
"""

import os
import torch
import gc

# --- 1. OTTIMIZZAZIONE GPU A100 ---
# Disattiviamo l'offload per tenere tutto in VRAM (più veloce)
os.environ["SUNO_OFFLOAD_CPU"] = "False"
os.environ["SUNO_USE_SMALL_MODELS"] = "False"

import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm

# Patch PyTorch
try:
    from . import pytorch_patch
except ImportError:
    try:
        import pytorch_patch
    except ImportError:
        pass

# Import Bark
try:
    from bark import SAMPLE_RATE, generate_audio, preload_models
    from scipy.io.wavfile import write as write_wav

    BARK_AVAILABLE = True
except ImportError:
    BARK_AVAILABLE = False
    warnings.warn("Bark non installato.")

try:
    from .emotion_mapper import (
        map_emotion_to_bark_prompt,
        get_bark_speaker,
        get_emotional_tag,
    )
    from .emotion_tag_optimizer import optimize_emotional_text
except ImportError:
    from emotion_mapper import (
        map_emotion_to_bark_prompt,
        get_bark_speaker,
        get_emotional_tag,
    )
    from emotion_tag_optimizer import optimize_emotional_text

import sys
from pathlib import Path

parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from text_templates import get_tts_text

# --- PATHS ---

# PUNTA AL NUOVO FILE GENERATO DA MERGE_CAPTIONS
PREDICTIONS_FILE = os.path.join(
    # BASE_DIR,
    # "src",
    # "models",
    # "three_classes",
    # "text_plus_video_metalearner_to_sentiment",
    # "results",
    "final_predictions_with_captions.csv",  # <--- FILE CORRETTO
)

OUTPUT_AUDIO_DIR = os.path.join(
    # BASE_DIR,
    # "src",
    # "tts",
    # "bark",
    "output_audio",
)

MODELS_PRELOADED = False


def setup_optimizations():
    """Configura PyTorch per massime prestazioni su A100."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("🚀 A100 Optimization: TF32 Enabled")


def preload_bark_models():
    global MODELS_PRELOADED
    if not BARK_AVAILABLE:
        return
    if not MODELS_PRELOADED:
        print("📥 Caricamento modelli Bark HQ in VRAM...")
        preload_models(
            text_use_gpu=True,
            text_use_small=False,
            coarse_use_gpu=True,
            coarse_use_small=False,
            fine_use_gpu=True,
            fine_use_small=False,
            codec_use_gpu=True,
            force_reload=False,
        )
        MODELS_PRELOADED = True
        print("✅ Modelli pronti in VRAM (No Offload)")


def load_predictions_data():
    if not os.path.exists(PREDICTIONS_FILE):
        raise FileNotFoundError(
            f"❌ File non trovato: {PREDICTIONS_FILE}\nEsegui prima src/utils/merge_captions.py!"
        )

    df = pd.read_csv(PREDICTIONS_FILE)
    print(f"📄 Dataset caricato: {len(df)} righe")

    # Check rapido
    missing_caps = df["caption"].isna().sum()
    if missing_caps > 0:
        print(
            f"⚠️ Attenzione: {missing_caps} caption sono ancora vuote (useranno template fallback)."
        )

    return df


def generate_emotional_audio(emotion, confidence, video_name, output_dir, caption=None):
    if not BARK_AVAILABLE:
        return None

    # Setup parametri
    history_prompt = get_bark_speaker(emotion, video_name=video_name)
    bark_config = map_emotion_to_bark_prompt(emotion, use_emotional_tags=True)
    emotional_tag = get_emotional_tag(emotion, confidence=confidence)

    # Testo
    if isinstance(caption, str) and len(caption) > 3:
        text = caption
        # A100 gestisce testi lunghi, ma Bark può allucinare se > 250 token
        if len(text) > 250:
            text = text[:250] + "..."
    else:
        text = get_tts_text(emotion, confidence, video_name)

    # Ottimizzazione
    text = optimize_emotional_text(
        text, emotion, use_tags=True, custom_tag=emotional_tag, confidence=confidence
    )

    print(f"🎙️ {video_name} | {emotion} ({confidence:.2f})")

    try:
        audio_array = generate_audio(
            text,
            history_prompt=history_prompt,
            text_temp=bark_config["temperature"],
            waveform_temp=bark_config["temperature"],
            silent=True,
        )

        safe_name = str(video_name).replace("/", "_").replace("\\", "_")
        filename = f"{safe_name}_{emotion.lower()}.wav"
        output_path = os.path.join(output_dir, filename)

        write_wav(output_path, SAMPLE_RATE, audio_array)

        # Su A100 non puliamo la cache ad ogni file per velocità
        return output_path

    except Exception as e:
        print(f"❌ Errore: {e}")
        gc.collect()
        torch.cuda.empty_cache()
        return None


def generate_from_csv(limit: int = None):
    setup_optimizations()

    try:
        df = load_predictions_data()
    except Exception as e:
        print(e)
        return

    df = df.dropna(subset=["predicted_label"])
    if limit:
        df = df.head(limit)

    os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)
    preload_bark_models()

    successful = 0
    pbar = tqdm(df.iterrows(), total=len(df), desc="A100 Generating")

    for i, (_, row) in enumerate(pbar):
        video_name = row["video_name"]
        emotion = str(row["predicted_label"]).capitalize()
        if emotion not in ["Positive", "Negative", "Neutral"]:
            emotion = "Neutral"

        # Gestione caption che potrebbero essere NaN
        caption = row["caption"] if pd.notna(row["caption"]) else None

        path = generate_emotional_audio(
            emotion,
            float(row["confidence"]),
            video_name,
            OUTPUT_AUDIO_DIR,
            caption,
        )

        if path:
            successful += 1

        # Pulizia leggera ogni 50 file
        if i % 50 == 0:
            gc.collect()

    print(f"✅ Fatto. {successful}/{len(df)} generati in: {OUTPUT_AUDIO_DIR}")


if __name__ == "__main__":
    generate_from_csv(limit=None)
