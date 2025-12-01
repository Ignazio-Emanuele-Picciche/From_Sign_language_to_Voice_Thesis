"""
╔══════════════════════════════════════════════════════════════════════════════╗
║      TTS GENERATOR - VERSIONE IBRIDA STABILE (RESUME + A100 FIX)             ║
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
# 1. FIX PYTORCH 2.6+ (INCORPORATO)
# ==============================================================================
_original_torch_load = torch.load


def patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = patched_torch_load
print("🛡️ Patch PyTorch applicato: weights_only=False per Bark.")

# ==============================================================================
# 2. CONFIGURAZIONE GPU (A100 SAFETY)
# ==============================================================================
# IMPORTANTE: Mettiamo Offload TRUE per evitare 'illegal memory access'
# Scarica la RAM tra una generazione e l'altra. È più sicuro.
os.environ["SUNO_OFFLOAD_CPU"] = "True"
os.environ["SUNO_USE_SMALL_MODELS"] = "False"


# Fix per Autocast su A100 (evita NaN)
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
    # Fallback se i file sono in una sottocartella
    try:
        from src.tts.bark.emotion_mapper import (
            map_emotion_to_bark_prompt,
            get_bark_speaker,
            get_emotional_tag,
        )
        from src.tts.bark.emotion_tag_optimizer import optimize_emotional_text
    except:
        print(
            "⚠️ Moduli emotion_mapper non trovati. Assicurati che siano nella stessa cartella."
        )

# --- PATHS ---
INPUT_FILE = "golden_test_set.csv"
OUTPUT_AUDIO_DIR = "output_audio_emosign"

MODELS_PRELOADED = False


def setup_optimizations():
    """Configura PyTorch per stabilità su A100."""
    if torch.cuda.is_available():
        # Disabilitiamo TF32 per evitare crash improvvisi
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print("🛡️ A100 Stability: TF32 Disabled (Safety First)")


def preload_bark_models():
    global MODELS_PRELOADED
    if not BARK_AVAILABLE:
        return
    if not MODELS_PRELOADED:
        print("📥 Caricamento modelli Bark HQ...")
        preload_models(
            text_use_gpu=True,
            text_use_small=False,
            coarse_use_gpu=True,  # Offload gestisce questo
            coarse_use_small=False,
            fine_use_gpu=True,
            fine_use_small=False,
            codec_use_gpu=True,
            force_reload=False,
        )
        MODELS_PRELOADED = True
        print("✅ Modelli pronti.")


def load_data():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"❌ File non trovato: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    if "Sentiment" not in df.columns:
        print("⚠️ Colonna 'Sentiment' mancante, uso default 0")
        df["Sentiment"] = 0
    print(f"📄 Dataset caricato: {len(df)} righe")
    return df


def generate_emotional_audio(emotion, sentiment_score, video_name, output_dir, caption):
    if not BARK_AVAILABLE:
        return None

    # Setup parametri
    history_prompt = get_bark_speaker(emotion, video_name=video_name)
    bark_config = map_emotion_to_bark_prompt(emotion)
    emotional_tag = get_emotional_tag(emotion, sentiment_score=sentiment_score)

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

    # Fix per testi vuoti o solo tag
    if text.strip() == emotional_tag.strip():
        text = f"{emotional_tag} ..."

    # NOME FILE SICURO
    safe_name = str(video_name).replace("/", "_").replace(".mp4", "")
    filename = f"{safe_name}_{emotion.lower()}_score{sentiment_score}.wav"
    output_path = os.path.join(output_dir, filename)

    # --- RESUME LOGIC ---
    # Se il file esiste già, lo saltiamo e risparmiamo GPU
    if os.path.exists(output_path):
        return "EXISTING"

    print(f"🎙️ [{emotion[:3]}] {video_name} (S:{sentiment_score}) -> '{text}'")

    try:
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
        print(f"❌ Errore generazione: {e}")
        # Se è un errore di memoria critico, usciamo per non bloccare tutto
        if "illegal memory access" in str(e):
            print(
                "🚨 ERRORE CRITICO GPU DETECTED. Riavvia il runtime e rilancia lo script (riprenderà da qui)."
            )
            sys.exit(1)

        gc.collect()
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
    pbar = tqdm(df.iterrows(), total=len(df), desc="Generazione A100")

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
            emotion,
            sentiment_score,
            video_name,
            OUTPUT_AUDIO_DIR,
            caption,
        )

        if path == "EXISTING":
            skipped += 1
        elif path:
            successful += 1

        # Garbage Collection frequente
        if i % 10 == 0:
            gc.collect()

    print(f"✅ Fatto. Generati: {successful}, Saltati (già esistenti): {skipped}")


if __name__ == "__main__":
    generate_from_csv()
