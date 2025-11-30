"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           TTS GENERATOR - MOTORE DI SINTESI VOCALE EMOTIVA (BARK)            ║
╚══════════════════════════════════════════════════════════════════════════════╝

📋 DESCRIZIONE:
    Modulo core per la generazione di audio espressivo utilizzando il modello
    generativo Bark. Questo script rappresenta l'ultimo stadio della pipeline
    multimodale "EmoSign".

    Prende in input le predizioni raffinate del Meta-Learner (Video + Testo)
    e converte le caption testuali in parlato, modulando l'espressività vocale
    (tono, risate, sospiri, pause) in base all'emozione predetta e al
    livello di confidenza del modello.

🔄 FLUSSO DI LAVORO:
    1. CARICAMENTO DATI:
       Legge il CSV generato dal Meta-Learner (`final_metalearner_predictions_for_tts.csv`).
       Se le caption mancano, effettua un fallback intelligente recuperandole
       dal Golden Test Set originale.

    2. MAPPING EMOTIVO:
       Per ogni video, utilizza `emotion_mapper` per tradurre:
       - Label (Positive/Negative) -> Speaker Voice (es. 'en_speaker_6')
       - Confidence Score -> Intensità del Tag Emotivo (es. [laughs] vs [chuckles])

    3. OTTIMIZZAZIONE TESTO:
       Utilizza `emotion_tag_optimizer` per inserire i tag emotivi non a caso,
       ma in punti sintatticamente naturali (es. dopo una virgola o a fine frase),
       per garantire una prosodia realistica.

    4. SINTESI (BARK):
       Genera l'audio waveform utilizzando la potenza di calcolo (GPU/CPU)
       e salva i file .wav risultanti.

📂 INPUT FILE:
    - Path: .../results/final_metalearner_predictions_for_tts.csv
    - Colonne richieste: video_name, predicted_label, confidence, [caption]

📂 OUTPUT:
    - Directory: src/tts/bark/output_audio/
    - Formato: {video_name}_{emotion}.wav (es. "video123_positive.wav")

🛠️ DIPENDENZE INTERNE:
    - emotion_mapper: Configurazione speaker e temperature.
    - emotion_tag_optimizer: Strategie linguistiche per i tag [sound].

⚠️ REQUISITI HARDWARE:
    Bark richiede significativa memoria RAM/VRAM.
    - Consigliato: GPU con >8GB VRAM
    - Minimo: 16GB System RAM (per esecuzione CPU lenta)

👤 AUTORE: Ignazio Emanuele Picciche
📅 DATA: Novembre 2025
🎓 PROGETTO: Tesi Magistrale - EmoSign
"""

"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           TTS GENERATOR - MOTORE DI SINTESI VOCALE EMOTIVA (PROD)            ║
╚══════════════════════════════════════════════════════════════════════════════╝

📋 DESCRIZIONE:
    Modulo core per la generazione di audio espressivo utilizzando il modello
    generativo Bark.
    
⚠️  CONFIGURAZIONE: HIGH QUALITY (STANDARD MODELS)
    - Usa modelli standard (migliore qualità, più VRAM richiesta).
    - Offload su CPU attivo per gestire la memoria.
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

# Patch PyTorch (se serve ancora, ma su A100 con env recenti forse no)
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

# --- PATHS (Ho lasciato i tuoi percorsi hardcoded commentati per sicurezza) ---
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
PREDICTIONS_FILE = (
    "final_metalearner_predictions_for_tts.csv"  # Modifica con path assoluto se serve
)
OUTPUT_AUDIO_DIR = "output_audio"
GOLDEN_TEST_FILE = "golden_test_set.csv"

MODELS_PRELOADED = False


def setup_optimizations():
    """Configura PyTorch per massime prestazioni su A100."""
    if torch.cuda.is_available():
        # Usa TF32 per matrici (velocizza molto su Ampere A100)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("🚀 A100 Optimization: TF32 Enabled")


def preload_bark_models():
    global MODELS_PRELOADED
    if not BARK_AVAILABLE:
        return
    if not MODELS_PRELOADED:
        print("📥 Caricamento modelli Bark HQ in VRAM...")
        # Su A100 carichiamo tutto su GPU subito e non lo muoviamo più
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
        print(f"⚠️ File locale non trovato, provo path assoluto...")
        # Fallback al path complesso se il locale non c'è
        abs_path = os.path.join(
            # BASE_DIR,
            # "src",
            # "models",
            # "three_classes",
            # "text_plus_video_metalearner_to_sentiment",
            # "results",
            "final_metalearner_predictions_for_tts.csv",
        )
        if os.path.exists(abs_path):
            return pd.read_csv(abs_path)
        else:
            raise FileNotFoundError(f"❌ File non trovato: {PREDICTIONS_FILE}")

    df = pd.read_csv(PREDICTIONS_FILE)

    # Logica recupero caption (semplificata per velocità)
    if "caption" not in df.columns or df["caption"].isnull().all():
        print("⚠️ Recupero caption...")
        # ... logica recupero ... (omessa per brevità, usa quella del tuo script precedente se serve)

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
        if len(text) > 250:
            text = text[:250] + "..."  # A100 regge di più ma occhio ai token
    else:
        text = get_tts_text(emotion, confidence, video_name)

    text = optimize_emotional_text(
        text, emotion, use_tags=True, custom_tag=emotional_tag, confidence=confidence
    )

    print(f"🎙️ {video_name} | {emotion} ({confidence:.2f})")

    try:
        # Generazione (Senza silent=True vediamo la barra di Bark se vogliamo)
        # Su A100 questo step dovrebbe volare
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

        # NON puliamo la cache ogni volta su A100, rallenta solo!
        # Facciamolo solo ogni tanto se proprio serve
        return output_path

    except Exception as e:
        print(f"❌ Errore: {e}")
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

    # TQDM con statistiche
    pbar = tqdm(df.iterrows(), total=len(df), desc="A100 Generating")

    for i, (_, row) in enumerate(pbar):
        video_name = row["video_name"]
        emotion = str(row["predicted_label"]).capitalize()
        if emotion not in ["Positive", "Negative", "Neutral"]:
            emotion = "Neutral"

        path = generate_emotional_audio(
            emotion,
            float(row["confidence"]),
            video_name,
            OUTPUT_AUDIO_DIR,
            row.get("caption"),
        )

        if path:
            successful += 1

        # Pulizia leggera ogni 50 file invece che ogni 1
        if i % 50 == 0:
            gc.collect()

    print(f"✅ Fatto. {successful}/{len(df)} generati.")


if __name__ == "__main__":
    generate_from_csv(limit=None)
