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

import os
import pandas as pd
import numpy as np
import warnings
from typing import Optional
from tqdm import tqdm

# Applica patch per compatibilità PyTorch 2.9+
try:
    from . import pytorch_patch
except ImportError:
    try:
        import pytorch_patch
    except ImportError:
        pass

# Bark imports
try:
    from bark import SAMPLE_RATE, generate_audio, preload_models
    from scipy.io.wavfile import write as write_wav

    BARK_AVAILABLE = True
except ImportError:
    BARK_AVAILABLE = False
    warnings.warn("Bark non installato. Audio non verrà generato.")

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

# Importa text_templates dal modulo parent
import sys
from pathlib import Path

parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from text_templates import get_tts_text, get_baseline_text

# --- CONFIGURAZIONE PATH ---
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
PREDICTIONS_FILE = os.path.join(
    BASE_DIR,
    "src",
    "models",
    "three_classes",
    "text_plus_video_metalearner_to_sentiment",
    "results",
    "final_metalearner_predictions_for_tts.csv",
)
OUTPUT_AUDIO_DIR = os.path.join(BASE_DIR, "src", "tts", "bark", "output_audio")
GOLDEN_TEST_FILE = os.path.join(
    BASE_DIR, "data", "processed", "golden_test_set.csv"
)  # Backup per le caption

# Flag per pre-caricare i modelli
MODELS_PRELOADED = False


def preload_bark_models():
    """Pre-carica i modelli Bark in memoria (10GB RAM richiesta)."""
    global MODELS_PRELOADED
    if not BARK_AVAILABLE:
        return
    if not MODELS_PRELOADED:
        print("📥 Caricamento modelli Bark in memoria...")
        preload_models()
        MODELS_PRELOADED = True
        print("✅ Modelli Bark pronti!")


def load_predictions_data():
    """Carica le predizioni del Meta-Learner e fa il merge con le caption se mancano."""
    if not os.path.exists(PREDICTIONS_FILE):
        raise FileNotFoundError(f"❌ File predizioni non trovato: {PREDICTIONS_FILE}")

    df = pd.read_csv(PREDICTIONS_FILE)

    # Se la colonna caption è vuota o manca, proviamo a recuperarla dal golden set originale
    if "caption" not in df.columns or df["caption"].isnull().all():
        print("⚠️  Caption mancanti nel file predizioni. Recupero dal Golden Set...")
        if os.path.exists(GOLDEN_TEST_FILE):
            df_golden = pd.read_csv(GOLDEN_TEST_FILE)
            # Merge per recuperare la caption usando video_name
            df = pd.merge(
                df,
                df_golden[["video_name", "caption"]],
                on="video_name",
                how="left",
                suffixes=("", "_golden"),
            )
            # Se c'era una colonna caption vuota, riempila
            if "caption" in df.columns:
                df["caption"] = df["caption"].fillna(df["caption_golden"])
            else:
                df["caption"] = df["caption_golden"]
            print(f"✅ Caption recuperate per {df['caption'].notnull().sum()} video.")
        else:
            print("❌ File Golden Set non trovato. Useremo template generici.")

    return df


def generate_emotional_audio(
    emotion: str,
    confidence: float,
    video_name: str,
    output_dir: str,
    caption: str = None,
    use_emotional_tags: bool = True,
    optimize_tag_placement: bool = True,
) -> str:
    """
    Genera un singolo file audio basato sui parametri passati.
    """
    if not BARK_AVAILABLE:
        return None

    # 1. Configurazione Bark (Speaker & Temperature)
    bark_config = map_emotion_to_bark_prompt(emotion, use_emotional_tags)
    history_prompt = get_bark_speaker(emotion)  # Default speaker per l'emozione

    # 2. Scelta Tag Emotivo (basata su Confidence)
    emotional_tag = ""
    if use_emotional_tags:
        emotional_tag = get_emotional_tag(emotion, confidence=confidence)

    # 3. Preparazione Testo
    if isinstance(caption, str) and len(caption) > 3:
        text = caption
    else:
        # Fallback se caption manca
        text = get_tts_text(emotion, confidence, video_name)

    # 4. Ottimizzazione Posizionamento Tag
    if use_emotional_tags and emotional_tag:
        if optimize_tag_placement:
            text = optimize_emotional_text(
                text, emotion, use_tags=True, custom_tag=emotional_tag
            )
        else:
            text = f"{emotional_tag} {text}"

    # 5. Generazione Audio
    print(f"🎙️  Generating: {video_name}")
    print(f"   Emotion: {emotion} (Conf: {confidence:.2f}) | Tag: {emotional_tag}")
    print(f'   Text: "{text}"')

    try:
        audio_array = generate_audio(
            text,
            history_prompt=history_prompt,
            text_temp=bark_config["temperature"],
            waveform_temp=bark_config["temperature"],
            silent=True,  # Riduci output console di Bark
        )

        # 6. Salvataggio
        safe_name = video_name.replace("/", "_").replace("\\", "_")
        filename = f"{safe_name}_{emotion.lower()}.wav"
        output_path = os.path.join(output_dir, filename)

        write_wav(output_path, SAMPLE_RATE, audio_array)
        return output_path

    except Exception as e:
        print(f"❌ Errore generazione {video_name}: {e}")
        return None


def generate_from_csv(limit: int = None):
    """
    Funzione principale per generare audio in batch dal CSV del Meta-Learner.
    Args:
        limit (int): Se specificato, genera solo i primi N audio (per test rapidi).
    """
    print("=" * 60)
    print("TTS BATCH GENERATION (META-LEARNER PREDICTIONS)")
    print("=" * 60)

    if not BARK_AVAILABLE:
        print("❌ Bark non installato. Impossibile procedere.")
        return

    # 1. Carica Dati
    try:
        df = load_predictions_data()
    except FileNotFoundError as e:
        print(e)
        return

    # Filtra eventuali righe senza predizione valida
    df = df.dropna(subset=["predicted_label"])

    # Applica limite se richiesto
    if limit:
        df = df.head(limit)
        print(f"⚠️  Limitato ai primi {limit} video per test.")

    print(f"📂 Output Directory: {OUTPUT_AUDIO_DIR}")
    os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)

    # 2. Preload Modelli
    preload_bark_models()

    # 3. Loop Generazione
    successful = 0
    failed = 0

    # Barre di progresso
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generazione Audio"):
        video_name = str(row["video_name"])
        emotion = str(row["predicted_label"])  # L'etichetta predetta dal Meta-Learner
        confidence = float(row["confidence"])  # La confidenza del Meta-Learner
        caption = row["caption"] if pd.notna(row["caption"]) else None

        # Normalizza nome emozione (es. "Positive" -> "Positive")
        emotion = emotion.capitalize()
        if emotion not in ["Positive", "Negative", "Neutral"]:
            # Fallback per etichette strane
            emotion = "Neutral"

        path = generate_emotional_audio(
            emotion=emotion,
            confidence=confidence,
            video_name=video_name,
            output_dir=OUTPUT_AUDIO_DIR,
            caption=caption,
            use_emotional_tags=True,  # Abilita [laughs], [sighs]
            optimize_tag_placement=True,  # Posiziona i tag in modo intelligente
        )

        if path:
            successful += 1
        else:
            failed += 1

    print("\n" + "=" * 60)
    print(f"✅ COMPLETATO")
    print(f"Audio generati correttamente: {successful}")
    print(f"Errori: {failed}")
    print(f"Cartella: {OUTPUT_AUDIO_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    # Esempio: Genera solo i primi 5 audio per testare che tutto funzioni
    # Imposta limit=None per processare tutto il dataset
    generate_from_csv(limit=5)
