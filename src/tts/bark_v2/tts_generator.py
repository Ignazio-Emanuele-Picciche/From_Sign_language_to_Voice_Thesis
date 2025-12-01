"""
╔══════════════════════════════════════════════════════════════════════════════╗
║      TTS GENERATOR - A100 STABLE EDITION (NO COMPILER, NO OFFLOAD)           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import torch
import gc
import pandas as pd
import numpy as np
import warnings
import sys
import contextlib
from tqdm import tqdm
from scipy.io.wavfile import write as write_wav

# ==============================================================================
# 1. FIX PYTORCH 2.6+ (Load Patch)
# ==============================================================================
_original_torch_load = torch.load


def patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = patched_torch_load
print("🛡️ Patch PyTorch applicato: Compatibilità Bark ripristinata.")

# ==============================================================================
# 2. CONFIGURAZIONE "RAW POWER" PER A100
# ==============================================================================
# DISABILITIAMO l'Offload: Teniamo tutto in VRAM (80GB sono sufficienti).
# Questo è il vero acceleratore: evita il caricamento continuo dalla RAM.
os.environ["SUNO_OFFLOAD_CPU"] = "False"
os.environ["SUNO_USE_SMALL_MODELS"] = "False"


# DISABILITIAMO Autocast/Mixed Precision: A100 soffre di NaN con Bark in FP16/BF16.
# Forziamo tutto in Float32 puro. Sarà un po' più lento del teorico, ma NON CRASHA.
@contextlib.contextmanager
def _mock_autocast(*args, **kwargs):
    yield


torch.cuda.amp.autocast = _mock_autocast
if hasattr(torch, "amp"):
    torch.amp.autocast = _mock_autocast

# --- IMPORTS ---
try:
    from bark import SAMPLE_RATE, generate_audio, preload_models

    BARK_AVAILABLE = True
except ImportError:
    BARK_AVAILABLE = False
    warnings.warn("Bark non installato.")

try:
    from emotion_mapper import (
        map_emotion_to_bark_prompt,
        get_bark_speaker,
        get_emotional_tag,
    )
    from emotion_tag_optimizer import optimize_emotional_text
except ImportError:
    try:
        from src.tts.bark.emotion_mapper import (
            map_emotion_to_bark_prompt,
            get_bark_speaker,
            get_emotional_tag,
        )
        from src.tts.bark.emotion_tag_optimizer import optimize_emotional_text
    except:
        pass

# --- PATHS ---
INPUT_FILE = "golden_test_set.csv"
OUTPUT_AUDIO_DIR = "output_audio_emosign"

MODELS_PRELOADED = False


def setup_optimizations():
    """Configura PyTorch per stabilità assoluta."""
    if torch.cuda.is_available():
        # Disabilitiamo TF32. È un'ottimizzazione che spesso causa NaN su Bark/A100.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print("🛡️ A100 Stability Mode: TF32 Disabled, Offload Disabled.")


def preload_bark_models():
    global MODELS_PRELOADED
    if not BARK_AVAILABLE:
        return
    if not MODELS_PRELOADED:
        print("📥 Caricamento modelli Bark HQ in VRAM (Full Precision)...")
        # NOTA: text_use_gpu=True ecc. mantengono i modelli fissi in GPU
        preload_models(
            text_use_gpu=True,
            text_use_small=False,
            coarse_use_gpu=True,
            coarse_use_small=False,
            fine_use_gpu=True,
            fine_use_small=False,
            codec_use_gpu=True,
            force_reload=True,  # Forziamo reload per pulire eventuali residui
        )
        MODELS_PRELOADED = True
        print("✅ Modelli caricati e pronti.")


def load_data():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"❌ File non trovato: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    if "Sentiment" not in df.columns:
        df["Sentiment"] = 0
    return df


def generate_emotional_audio(emotion, sentiment_score, video_name, output_dir, caption):
    if not BARK_AVAILABLE:
        return None

    # Setup
    history_prompt = get_bark_speaker(emotion, video_name=video_name)
    bark_config = map_emotion_to_bark_prompt(emotion)
    emotional_tag = get_emotional_tag(emotion, sentiment_score=sentiment_score)

    # Testo
    text = str(caption)
    if len(text) > 250:
        text = text[:250] + "..."

    # Ottimizzazione
    text = optimize_emotional_text(
        text,
        emotion,
        use_tags=True,
        custom_tag=emotional_tag,
        sentiment_score=sentiment_score,
    )

    if text.strip() == emotional_tag.strip():
        text = f"{emotional_tag} ..."

    # Path
    safe_name = str(video_name).replace("/", "_").replace(".mp4", "")
    filename = f"{safe_name}_{emotion.lower()}_score{sentiment_score}.wav"
    output_path = os.path.join(output_dir, filename)

    # --- RESUME LOGIC ---
    if os.path.exists(output_path):
        return "EXISTING"

    try:
        # Generazione (Senza Autocast grazie al fix sopra)
        audio_array = generate_audio(
            text,
            history_prompt=history_prompt,
            text_temp=bark_config["temperature"],
            waveform_temp=bark_config["temperature"],
            silent=True,
        )
        write_wav(output_path, SAMPLE_RATE, audio_array)
        return output_path

    except Exception as e:
        print(f"❌ Errore {video_name}: {e}")
        # Se la GPU muore, usciamo puliti per permettere restart manuale
        if "illegal memory access" in str(e) or "CUDAGraphs" in str(e):
            print("🚨 ERRORE FATALE GPU: È necessario 'Disconnect and Delete Runtime'.")
            sys.exit(1)
        return None


def generate_from_csv(limit: int = None):
    setup_optimizations()
    try:
        df = load_data()
    except Exception as e:
        print(e)
        return

    if limit:
        df = df.head(limit)
    os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)
    preload_bark_models()

    successful = 0
    skipped = 0

    # SENZA GC NEL LOOP: Massimizziamo la velocità.
    # Con 80GB VRAM e 170GB RAM non serve pulire ogni secondo.
    pbar = tqdm(df.iterrows(), total=len(df), desc="🚀 A100 Generation")

    for i, (_, row) in enumerate(pbar):
        video_name = row["video_name"]
        try:
            sentiment_score = int(row["Sentiment"])
        except:
            sentiment_score = 0

        if sentiment_score > 0:
            emotion = "Positive"
        elif sentiment_score < 0:
            emotion = "Negative"
        else:
            emotion = "Neutral"

        caption = row["caption"] if pd.notna(row["caption"]) else ""
        if not caption:
            continue

        path = generate_emotional_audio(
            emotion, sentiment_score, video_name, OUTPUT_AUDIO_DIR, caption
        )

        if path == "EXISTING":
            skipped += 1
        elif path:
            successful += 1

    print(f"✅ Fatto. Generati: {successful}, Saltati (Esistenti): {skipped}")


if __name__ == "__main__":
    generate_from_csv()
