"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           TTS GENERATOR - MOTORE DI SINTESI VOCALE EMOTIVA (EMOSIGN)         ║
╚══════════════════════════════════════════════════════════════════════════════╝

📋 DESCRIZIONE:
    Versione adattata per il dataset "Golden Test Set" di EmoSign.
    Genera audio basandosi sui valori di intensità [-3, +3] annotati dai segnanti nativi.

🔄 LOGICA DIROTTATA SUI DATI EMOSIGN:
    1. Input: data/processed/golden_test_set.csv (video_name, caption, Sentiment)
    2. Sentiment Mapping:
       - Valori > 0 -> Positive (Speaker Allegri)
       - Valori < 0 -> Negative (Speaker Tristi)
       - Valore 0   -> Neutral  (Speaker Professionali)
    3. Intensità:
       - Il valore assoluto (1, 2, 3) pilota l'inserimento dei tag (risate, sospiri).

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

# --- 1. OTTIMIZZAZIONE GPU A100 ---
os.environ["SUNO_OFFLOAD_CPU"] = "False"
os.environ["SUNO_USE_SMALL_MODELS"] = "False"

# --- IMPORTS ---
# Patch PyTorch
try:
    from . import pytorch_patch
except ImportError:
    try:
        import pytorch_patch
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


def load_data():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"❌ File non trovato: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    # Pulizia colonne essenziali
    required_cols = ["video_name", "caption", "Sentiment"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Il CSV deve contenere le colonne: {required_cols}")

    print(f"📄 Dataset EmoSign caricato: {len(df)} campioni")
    return df


def generate_emotional_audio(emotion, sentiment_score, video_name, output_dir, caption):
    if not BARK_AVAILABLE:
        return None

    # 1. Selezione Speaker e Parametri Base
    history_prompt = get_bark_speaker(emotion, video_name=video_name)
    bark_config = map_emotion_to_bark_prompt(emotion)  # Recupera temperatura

    # 2. Selezione Tag Emotivo basato su Score [-3, +3]
    emotional_tag = get_emotional_tag(emotion, sentiment_score=sentiment_score)

    # 3. Preparazione Testo
    text = caption
    if len(text) > 250:  # Troncamento di sicurezza
        text = text[:250] + "..."

    # 4. Ottimizzazione Sintattica (Inserimento Tag)
    text = optimize_emotional_text(
        text,
        emotion,
        use_tags=True,
        custom_tag=emotional_tag,
        sentiment_score=sentiment_score,
    )

    print(
        f"🎙️ [{emotion[:3].upper()}] {video_name} (Score: {sentiment_score}) -> '{text}'"
    )

    try:
        audio_array = generate_audio(
            text,
            history_prompt=history_prompt,
            text_temp=bark_config["temperature"],
            waveform_temp=bark_config["temperature"],
            silent=True,
        )

        safe_name = str(video_name).replace("/", "_").replace(".mp4", "")
        # Filename include lo score per facilitare l'analisi successiva
        filename = f"{safe_name}_{emotion.lower()}_score{sentiment_score}.wav"
        output_path = os.path.join(output_dir, filename)

        write_wav(output_path, SAMPLE_RATE, audio_array)
        return output_path

    except Exception as e:
        print(f"❌ Errore generazione {video_name}: {e}")
        gc.collect()
        torch.cuda.empty_cache()
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
    pbar = tqdm(df.iterrows(), total=len(df), desc="Generazione Audio")

    for i, (_, row) in enumerate(pbar):
        video_name = row["video_name"]
        caption = row["caption"]

        # --- LOGICA DI CONVERSIONE EMOSIGN ---
        try:
            sentiment_score = int(row["Sentiment"])
        except (ValueError, TypeError):
            sentiment_score = 0  # Default Neutro se nullo

        # Derivazione Label da Segno
        if sentiment_score > 0:
            emotion = "Positive"
        elif sentiment_score < 0:
            emotion = "Negative"
        else:
            emotion = "Neutral"

        # Generazione
        if pd.notna(caption):
            path = generate_emotional_audio(
                emotion,
                sentiment_score,
                video_name,
                OUTPUT_AUDIO_DIR,
                caption,
            )
            if path:
                successful += 1

        if i % 50 == 0:
            gc.collect()

    print(f"✅ Fatto. {successful}/{len(df)} audio salvati in: {OUTPUT_AUDIO_DIR}")


if __name__ == "__main__":
    generate_from_csv()
