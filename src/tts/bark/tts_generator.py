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
import gc
import torch

# --- 1. APPLICA PATCH PRIMA DI TUTTO ---
try:
    # Se il file è nella stessa cartella
    from . import pytorch_patch
except ImportError:
    # Se lo lanci direttamente come script
    import pytorch_patch

# --- 2. CONFIGURAZIONE ANTI-CRASH ---
# os.environ["SUNO_USE_SMALL_MODELS"] = "True"  <-- (Disattivato per HQ)
os.environ["SUNO_OFFLOAD_CPU"] = "True"

import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm

# Patch e Import Bark
try:
    from . import pytorch_patch
except ImportError:
    try:
        import pytorch_patch
    except ImportError:
        pass

try:
    from bark import SAMPLE_RATE, generate_audio, preload_models
    from scipy.io.wavfile import write as write_wav

    BARK_AVAILABLE = True
except ImportError:
    BARK_AVAILABLE = False
    warnings.warn("Bark non installato.")

# Import Moduli Interni
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

# Import Template Utilità
import sys
from pathlib import Path

parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from text_templates import get_tts_text

# --- CONFIGURAZIONE PATH ---
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
PREDICTIONS_FILE = os.path.join(
    # BASE_DIR,
    # "src",
    # "models",
    # "three_classes",
    # "text_plus_video_metalearner_to_sentiment",
    # "results",
    "final_metalearner_predictions_for_tts.csv",
)
OUTPUT_AUDIO_DIR = os.path.join(
    # BASE_DIR,
    # "src",
    # "tts",
    # "bark",
    "output_audio",
)
GOLDEN_TEST_FILE = os.path.join(
    # BASE_DIR,
    # "data",
    # "processed",
    "golden_test_set.csv",
)

MODELS_PRELOADED = False


def preload_bark_models():
    global MODELS_PRELOADED
    if not BARK_AVAILABLE:
        return
    if not MODELS_PRELOADED:
        print("📥 Caricamento modelli Bark (STANDARD/LARGE)...")
        # Nota: Il primo download sarà di circa 12GB se non li hai mai usati
        preload_models()
        MODELS_PRELOADED = True


def clean_memory():
    """Pulisce la memoria dopo ogni generazione."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_predictions_data():
    if not os.path.exists(PREDICTIONS_FILE):
        raise FileNotFoundError(f"❌ File non trovato: {PREDICTIONS_FILE}")

    df = pd.read_csv(PREDICTIONS_FILE)

    if "caption" not in df.columns or df["caption"].isnull().all():
        print("⚠️  Recupero caption dal Golden Set...")
        if os.path.exists(GOLDEN_TEST_FILE):
            df_golden = pd.read_csv(GOLDEN_TEST_FILE)
            df = pd.merge(
                df,
                df_golden[["video_name", "caption"]],
                on="video_name",
                how="left",
                suffixes=("", "_golden"),
            )
            if "caption" in df.columns:
                df["caption"] = df["caption"].fillna(df["caption_golden"])
            else:
                df["caption"] = df["caption_golden"]
    return df


def generate_emotional_audio(
    emotion: str,
    confidence: float,
    video_name: str,
    output_dir: str,
    caption: str = None,
) -> str:
    """Genera audio singolo con logica avanzata."""
    if not BARK_AVAILABLE:
        return None

    # 1. Configurazione Speaker (Hash sul nome video)
    history_prompt = get_bark_speaker(emotion, video_name=video_name)
    bark_config = map_emotion_to_bark_prompt(emotion, use_emotional_tags=True)

    # 2. Scelta Tag (basata su confidenza)
    emotional_tag = get_emotional_tag(emotion, confidence=confidence)

    # 3. Preparazione Testo
    if isinstance(caption, str) and len(caption) > 3:
        text = caption
        # TRUNCATE SAFEGUARD: I modelli grandi sono più sensibili alla lunghezza
        if len(text) > 180:
            text = text[:180] + "..."
    else:
        text = get_tts_text(emotion, confidence, video_name)

    # 4. Ottimizzazione Posizionamento (Context-Aware)
    text = optimize_emotional_text(
        text, emotion, use_tags=True, custom_tag=emotional_tag, confidence=confidence
    )

    print(
        f"🎙️  Gen: {video_name} | {emotion} (Conf: {confidence:.2f}) | {history_prompt}"
    )

    try:
        audio_array = generate_audio(
            text,
            history_prompt=history_prompt,
            text_temp=bark_config["temperature"],
            waveform_temp=bark_config["temperature"],
            silent=True,
        )

        safe_name = video_name.replace("/", "_").replace("\\", "_")
        filename = f"{safe_name}_{emotion.lower()}.wav"
        output_path = os.path.join(output_dir, filename)

        write_wav(output_path, SAMPLE_RATE, audio_array)
        clean_memory()  # Fondamentale con modelli grandi
        return output_path

    except Exception as e:
        print(f"❌ Errore {video_name}: {e}")
        clean_memory()
        return None


def generate_from_csv(limit: int = None):
    """Funzione principale per generazione Batch."""
    print("=" * 60)
    print("TTS BATCH GENERATION (META-LEARNER - HQ MODELS)")
    print("=" * 60)

    if not BARK_AVAILABLE:
        return

    try:
        df = load_predictions_data()
    except Exception as e:
        print(e)
        return

    df = df.dropna(subset=["predicted_label"])

    if limit:
        df = df.head(limit)
        print(f"⚠️  Limitato ai primi {limit} video.")

    os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)
    preload_bark_models()

    successful, failed = 0, 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generazione"):
        video_name = str(row["video_name"])
        emotion = str(row["predicted_label"]).capitalize()
        if emotion not in ["Positive", "Negative", "Neutral"]:
            emotion = "Neutral"

        confidence = float(row["confidence"])
        caption = row["caption"] if pd.notna(row["caption"]) else None

        path = generate_emotional_audio(
            emotion=emotion,
            confidence=confidence,
            video_name=video_name,
            output_dir=OUTPUT_AUDIO_DIR,
            caption=caption,
        )

        if path:
            successful += 1
        else:
            failed += 1

    print("\n" + "=" * 60)
    print(f"✅ COMPLETATO: {successful} ok, {failed} errori.")
    print("=" * 60)


if __name__ == "__main__":
    # Esegui senza limiti per generare tutto
    generate_from_csv(limit=None)
