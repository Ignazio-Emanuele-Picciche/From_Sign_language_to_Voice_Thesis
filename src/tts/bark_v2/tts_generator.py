"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           TTS GENERATOR - FIX STABILITÀ A100 & NAN ERROR                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import torch
import gc
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
import sys
from pathlib import Path
from scipy.io.wavfile import write as write_wav

# --- 1. CONFIGURAZIONE GPU ---
# IMPORTANTE: Manteniamo offload False per velocità, ma disabilitiamo TF32 dopo
os.environ["SUNO_OFFLOAD_CPU"] = "False"
os.environ["SUNO_USE_SMALL_MODELS"] = "False"

# --- IMPORTS ---
# Patch PyTorch
try:
    from . import pytorch_patch
except ImportError:
    pass
try:
    from bark import SAMPLE_RATE, generate_audio, preload_models

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

# --- PATHS ---
INPUT_FILE = "golden_test_set.csv"
OUTPUT_AUDIO_DIR = "output_audio_emosign"

MODELS_PRELOADED = False


def setup_optimizations():
    """
    Configurazione Hardware.
    FIX: Disabilitiamo TF32 per evitare errori 'probability tensor contains nan' su A100.
    Bark richiede precisione float32 standard per stabilità.
    """
    if torch.cuda.is_available():
        # DISABILITIAMO TF32 PER STABILITÀ
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print("🛡️ A100 Stability Mode: TF32 Disabled (Prevents NaNs)")


def preload_bark_models():
    global MODELS_PRELOADED
    if not BARK_AVAILABLE:
        return
    if not MODELS_PRELOADED:
        print("📥 Caricamento modelli Bark HQ in VRAM...")
        try:
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
        except Exception as e:
            print(f"Errore preload: {e}")


def load_data():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"❌ File non trovato: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    required_cols = ["video_name", "caption", "Sentiment"]
    # Check permissivo se manca Sentiment
    if "Sentiment" not in df.columns:
        print("⚠️ Colonna 'Sentiment' mancante, uso default 0")
        df["Sentiment"] = 0

    print(f"📄 Dataset EmoSign caricato: {len(df)} campioni")
    return df


def generate_emotional_audio(emotion, sentiment_score, video_name, output_dir, caption):
    if not BARK_AVAILABLE:
        return None

    # 1. Setup
    history_prompt = get_bark_speaker(emotion, video_name=video_name)
    bark_config = map_emotion_to_bark_prompt(emotion)
    emotional_tag = get_emotional_tag(emotion, sentiment_score=sentiment_score)

    # 2. Validazione Testo
    if pd.isna(caption) or str(caption).strip() == "":
        print(f"⚠️ Caption vuota per {video_name}, salto.")
        return None

    text = str(caption)
    if len(text) > 250:
        text = text[:250] + "..."

    # 3. Ottimizzazione
    text = optimize_emotional_text(
        text,
        emotion,
        use_tags=True,
        custom_tag=emotional_tag,
        sentiment_score=sentiment_score,
    )

    # CHECK DI SICUREZZA: Bark crasha se il testo è SOLO un tag (es. "[laughs]")
    # Se il testo ottimizzato è troppo corto o uguale al tag, aggiungiamo un filler silenzioso
    if text.strip() == emotional_tag.strip():
        text = f"{emotional_tag} ..."

    print(f"🎙️ [{emotion[:3]}] {video_name} (S:{sentiment_score}) -> '{text}'")

    try:
        # Generazione Audio
        audio_array = generate_audio(
            text,
            history_prompt=history_prompt,
            text_temp=bark_config["temperature"],
            waveform_temp=bark_config["temperature"],
            silent=True,  # Riduce output console
        )

        safe_name = str(video_name).replace("/", "_").replace(".mp4", "")
        filename = f"{safe_name}_{emotion.lower()}_score{sentiment_score}.wav"
        output_path = os.path.join(output_dir, filename)

        write_wav(output_path, SAMPLE_RATE, audio_array)
        return output_path

    except Exception as e:
        # Gestione specifica errore CUDA
        err_msg = str(e)
        if "device-side assert" in err_msg or "probability tensor" in err_msg:
            print(
                f"💥 CRITICAL ERROR su {video_name}: NaN generati. Riprovo senza tag."
            )
            # Fallback: Riprova senza tag emotivi e temperatura più bassa
            try:
                audio_array = generate_audio(
                    caption,  # Testo pulito originale
                    history_prompt=history_prompt,
                    text_temp=0.5,  # Temp più sicura
                    waveform_temp=0.5,
                    silent=True,
                )
                safe_name = str(video_name).replace("/", "_").replace(".mp4", "")
                filename = f"{safe_name}_{emotion.lower()}_FALLBACK.wav"
                path = os.path.join(output_dir, filename)
                write_wav(path, SAMPLE_RATE, audio_array)
                print(f"✅ Salvato FALLBACK per {video_name}")
                return path
            except:
                print(f"❌ Fallito anche il fallback.")
                return None
        else:
            print(f"❌ Errore generico: {e}")

        # Pulizia memoria aggressiva
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except:
            pass  # Se CUDA è rotto, non possiamo farci nulla
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
    pbar = tqdm(df.iterrows(), total=len(df), desc="Generazione")

    for i, (_, row) in enumerate(pbar):
        video_name = row["video_name"]
        caption = row["caption"]

        try:
            sentiment_score = int(row["Sentiment"])
        except (ValueError, TypeError):
            sentiment_score = 0

        # Derivazione Label
        if sentiment_score > 0:
            emotion = "Positive"
        elif sentiment_score < 0:
            emotion = "Negative"
        else:
            emotion = "Neutral"

        path = generate_emotional_audio(
            emotion,
            sentiment_score,
            video_name,
            OUTPUT_AUDIO_DIR,
            caption,
        )
        if path:
            successful += 1

        if i % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    print(f"✅ Finito. {successful}/{len(df)} file salvati.")


if __name__ == "__main__":
    generate_from_csv()
