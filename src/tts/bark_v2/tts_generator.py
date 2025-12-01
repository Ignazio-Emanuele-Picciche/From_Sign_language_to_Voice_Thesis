"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           TTS GENERATOR - FIX STABILITÀ A100 (NO MIXED PRECISION)            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import torch
import gc
import pandas as pd
import numpy as np
import warnings
import contextlib
from tqdm import tqdm
import sys
from pathlib import Path
from scipy.io.wavfile import write as write_wav

# ==============================================================================
# 🛑 FIX CRITICI PER A100 E PYTORCH 2.6+
# Da eseguire PRIMA di qualsiasi import di Bark o torch.nn
# ==============================================================================

# 1. FIX PYTORCH 2.6+ (Pickle Error)
_original_torch_load = torch.load


def patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = patched_torch_load


# 2. FIX A100 NAN (Disabilita Mixed Precision / Autocast)
# Bark su A100 crasha con autocast (NaN errors).
# Forziamo un "finto" autocast che non fa nulla, mantenendo tutto in FP32.
@contextlib.contextmanager
def _mock_autocast(*args, **kwargs):
    yield


torch.cuda.amp.autocast = _mock_autocast
if hasattr(torch, "amp"):
    torch.amp.autocast = _mock_autocast

print(
    "🛡️ A100 Safe-Mode attivato: Autocast disabilitato (Force FP32), Patch Load applicato."
)
# ==============================================================================

# --- CONFIGURAZIONE AMBIENTE ---
os.environ["SUNO_OFFLOAD_CPU"] = "False"
os.environ["SUNO_USE_SMALL_MODELS"] = "False"
# Disabilita algoritmi non deterministici che possono causare crash
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# --- IMPORTS ---
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
    Configurazione Hardware Estrema per A100
    """
    if torch.cuda.is_available():
        # Disabilita TF32 (vecchia API)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        # Disabilita TF32 (nuova API PyTorch 2.x)
        if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
            torch.backends.cuda.matmul.fp32_precision = "highest"  # Forza IEEE 754

        print("🛡️ Precisione impostata su FP32 (Highest/IEEE) per evitare NaN.")


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
            print("✅ Modelli caricati (FP32 Mode).")
        except Exception as e:
            print(f"❌ Errore fatale preload: {e}")
            sys.exit(1)


def load_data():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"❌ File non trovato: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    if "Sentiment" not in df.columns:
        print("⚠️ Colonna 'Sentiment' mancante, uso default 0")
        df["Sentiment"] = 0
    print(f"📄 Dataset EmoSign caricato: {len(df)} campioni")
    return df


def generate_emotional_audio(emotion, sentiment_score, video_name, output_dir, caption):
    if not BARK_AVAILABLE:
        return None

    # Setup
    history_prompt = get_bark_speaker(emotion, video_name=video_name)
    bark_config = map_emotion_to_bark_prompt(emotion)
    emotional_tag = get_emotional_tag(emotion, sentiment_score=sentiment_score)

    if pd.isna(caption) or str(caption).strip() == "":
        return None

    text = str(caption)
    if len(text) > 250:
        text = text[:250] + "..."

    # Ottimizzazione testo
    text = optimize_emotional_text(
        text,
        emotion,
        use_tags=True,
        custom_tag=emotional_tag,
        sentiment_score=sentiment_score,
    )

    if text.strip() == emotional_tag.strip():
        text = f"{emotional_tag} ..."

    print(f"🎙️ [{emotion[:3]}] {video_name} (S:{sentiment_score}) -> '{text}'")

    try:
        # Generazione (Ora protetta dal mock autocast)
        audio_array = generate_audio(
            text,
            history_prompt=history_prompt,
            text_temp=bark_config["temperature"],
            waveform_temp=bark_config["temperature"],
            silent=True,
        )

        safe_name = str(video_name).replace("/", "_").replace(".mp4", "")
        filename = f"{safe_name}_{emotion.lower()}_score{sentiment_score}.wav"
        output_path = os.path.join(output_dir, filename)
        write_wav(output_path, SAMPLE_RATE, audio_array)
        return output_path

    except Exception as e:
        # Gestione Errori Aggressiva
        print(f"❌ Errore su {video_name}: {e}")

        # Tentativo Fallback in caso di NaN residui
        if "device-side assert" in str(e) or "probability tensor" in str(e):
            print("💥 Riprovo con parametri conservativi...")
            try:
                audio_array = generate_audio(
                    str(caption),  # No tag
                    history_prompt=history_prompt,
                    text_temp=0.5,
                    waveform_temp=0.5,
                    silent=True,
                )
                safe_name = str(video_name).replace("/", "_").replace(".mp4", "")
                path = os.path.join(output_dir, f"{safe_name}_FALLBACK.wav")
                write_wav(path, SAMPLE_RATE, audio_array)
                print("✅ Fallback salvato.")
                return path
            except:
                pass

        gc.collect()
        # Non chiamiamo empty_cache se c'è un assert failure pendente, peggiora le cose
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

        if sentiment_score > 0:
            emotion = "Positive"
        elif sentiment_score < 0:
            emotion = "Negative"
        else:
            emotion = "Neutral"

        path = generate_emotional_audio(
            emotion, sentiment_score, video_name, OUTPUT_AUDIO_DIR, caption
        )
        if path:
            successful += 1

        # Garbage collection frequente per evitare frammentazione VRAM
        if i % 5 == 0:
            gc.collect()

    print(f"✅ Finito. {successful}/{len(df)} file salvati.")


if __name__ == "__main__":
    generate_from_csv()
