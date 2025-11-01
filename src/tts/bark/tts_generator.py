"""
TTS Generator - Genera audio emotivo usando Bark TTS

Bark è un modello transformer-based di Suno AI che supporta:
- Generazione emotiva attraverso speaker prompts
- Tag speciali: [laughs], [sighs], [gasps], [clears throat], etc.
- Controllo fine tramite temperature
- Audio molto naturale e espressivo
"""

import os
import numpy as np
from typing import Optional
import warnings

# Applica patch per compatibilità PyTorch 2.9+
try:
    from . import pytorch_patch
except ImportError:
    pass  # Patch non disponibile, continua comunque

# Bark imports
try:
    from bark import SAMPLE_RATE, generate_audio, preload_models
    from scipy.io.wavfile import write as write_wav

    BARK_AVAILABLE = True
except ImportError:
    BARK_AVAILABLE = False
    warnings.warn(
        "Bark non installato. Installa con: pip install git+https://github.com/suno-ai/bark.git"
    )

from .emotion_mapper import (
    map_emotion_to_bark_prompt,
    get_bark_speaker,
    get_emotional_tag,
    get_alternative_emotional_tags,
)
from .emotion_tag_optimizer import optimize_emotional_text, get_alternative_tags

# Importa text_templates dal modulo parent
import sys
from pathlib import Path

parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from text_templates import get_tts_text, get_baseline_text


# Flag per pre-caricare i modelli (velocizza la generazione)
MODELS_PRELOADED = False


def preload_bark_models():
    """
    Pre-carica i modelli Bark in memoria
    Questo velocizza la generazione successiva ma richiede ~10GB di RAM
    """
    global MODELS_PRELOADED

    if not BARK_AVAILABLE:
        raise RuntimeError("Bark non è installato!")

    if not MODELS_PRELOADED:
        print("Caricamento modelli Bark in memoria...")
        preload_models()
        MODELS_PRELOADED = True
        print("✅ Modelli Bark caricati!")


def generate_emotional_audio(
    emotion: str,
    confidence: float,
    video_name: str,
    output_dir: str,
    caption: str = None,
    use_emotional_tags: bool = True,
    alternative_speaker: int = 0,
    alternative_tag: int = 0,
    confidence_based_tags: bool = True,
    preload: bool = False,
    optimize_tag_placement: bool = True,
) -> str:
    """
    Genera audio emotivo per un video analizzato usando Bark

    Args:
        emotion (str): Emozione predetta ('Positive', 'Negative', 'Neutral')
        confidence (float): Confidenza della predizione (0.0-1.0 o 0-100)
        video_name (str): Nome del video (per naming del file)
        output_dir (str): Directory dove salvare l'audio
        caption (str, optional): Testo originale del sign language da pronunciare
        use_emotional_tags (bool): Se True, usa tag emotivi come [laughs], [sighs]
        alternative_speaker (int): Indice speaker alternativo (0-2)
        alternative_tag (int): 🆕 Indice tag emotivo alternativo (0=default, 1+=alternatives)
        confidence_based_tags (bool): 🆕 Se True, sceglie tag basandosi su confidenza
        preload (bool): Se True, pre-carica i modelli Bark (prima generazione più lenta, successive veloci)
        optimize_tag_placement (bool): Se True, posiziona tag in modo intelligente nel testo

    Returns:
        str: Path completo del file audio generato

    Example:
        >>> # Tag default basato su confidenza
        >>> generate_emotional_audio('Positive', 0.92, 'video_001', 'results/tts_audio')

        >>> # Tag alternativo specifico
        >>> generate_emotional_audio('Positive', 0.92, 'video_001', 'results/tts_audio',
        ...                          alternative_tag=1)  # Usa [chuckles] invece di [laughs]
    """
    if not BARK_AVAILABLE:
        raise RuntimeError(
            "Bark non è installato! Installa con: pip install git+https://github.com/suno-ai/bark.git"
        )

    # Pre-carica modelli se richiesto
    if preload:
        preload_bark_models()

    # Crea directory se non esiste
    os.makedirs(output_dir, exist_ok=True)

    # Ottieni configurazione Bark per l'emozione
    bark_config = map_emotion_to_bark_prompt(emotion, use_emotional_tags)

    # Ottieni speaker (con possibilità di usare alternative)
    history_prompt = get_bark_speaker(emotion, alternative_speaker)

    # 🆕 Determina quale tag emotivo usare
    emotional_tag = ""
    if use_emotional_tags:
        if confidence_based_tags and alternative_tag == 0:
            # Usa tag basato su confidenza
            emotional_tag = get_emotional_tag(emotion, confidence=confidence)
        else:
            # Usa tag alternativo specifico o default
            emotional_tag = get_emotional_tag(emotion, alternative=alternative_tag)

    # Genera testo
    if caption:
        # Usa il caption originale del sign language
        text = caption
    else:
        # Genera testo basato sull'emozione
        text = get_tts_text(emotion, confidence, video_name, caption=None)

    # Applica ottimizzazione posizionamento tag emotivi
    if use_emotional_tags and emotional_tag:
        if optimize_tag_placement:
            # Usa posizionamento intelligente
            text = optimize_emotional_text(
                text=text,
                emotion=emotion,
                use_tags=True,
                custom_tag=emotional_tag,
            )
        else:
            # Metodo tradizionale: tag all'inizio
            text = f"{emotional_tag} {text}"

    # Path output (Bark genera WAV)
    safe_video_name = video_name.replace("/", "_").replace("\\", "_")
    output_filename = f"{safe_video_name}_{emotion.lower()}.wav"
    output_path = os.path.join(output_dir, output_filename)

    # Genera audio con Bark
    print(f"Generazione audio con Bark...")
    print(f"  Testo: {text[:80]}..." if len(text) > 80 else f"  Testo: {text}")
    print(f"  Speaker: {history_prompt}")
    print(f"  Tag emotivo: {emotional_tag if emotional_tag else '(nessuno)'}")
    print(f"  Temperature: {bark_config['temperature']}")

    audio_array = generate_audio(
        text,
        history_prompt=history_prompt,
        text_temp=bark_config["temperature"],
        waveform_temp=bark_config["temperature"],
    )

    # Salva audio come WAV (Bark usa sample rate 24kHz)
    write_wav(output_path, SAMPLE_RATE, audio_array)

    print(f"  ✅ Audio salvato: {output_path}")

    return output_path


def generate_baseline_audio(output_path: str, preload: bool = False) -> str:
    """
    Genera audio baseline neutrale (senza modulazione emotiva)
    Serve come riferimento per comparazioni

    Args:
        output_path (str): Path completo dove salvare il file
        preload (bool): Se True, pre-carica i modelli Bark

    Returns:
        str: Path del file audio generato

    Example:
        >>> generate_baseline_audio('results/tts_audio/baseline/baseline_neutral.wav')
        'results/tts_audio/baseline/baseline_neutral.wav'
    """
    if not BARK_AVAILABLE:
        raise RuntimeError("Bark non è installato!")

    # Pre-carica modelli se richiesto
    if preload:
        preload_bark_models()

    # Crea directory se non esiste
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Usa speaker neutro
    neutral_speaker = "v2/en_speaker_9"

    # Testo baseline
    text = get_baseline_text()

    # Genera audio
    print(f"Generazione audio baseline con Bark...")
    audio_array = generate_audio(
        text,
        history_prompt=neutral_speaker,
        text_temp=0.5,
        waveform_temp=0.5,
    )

    # Salva audio come WAV
    write_wav(output_path, SAMPLE_RATE, audio_array)

    print(f"✅ Baseline salvato: {output_path}")

    return output_path


if __name__ == "__main__":
    # Test del modulo
    print("=" * 70)
    print("TEST TTS GENERATOR - BARK")
    print("=" * 70)

    if not BARK_AVAILABLE:
        print("❌ Bark non installato!")
        print("   Installa con: pip install git+https://github.com/suno-ai/bark.git")
        exit(1)

    # Test 1: Pre-carica modelli
    print("\n1. Pre-caricamento modelli Bark...")
    preload_bark_models()

    # Test 2: Genera baseline
    print("\n2. Generazione audio baseline...")
    baseline_path = generate_baseline_audio("test_baseline_bark.wav", preload=True)

    # Test 3: Genera audio emotivo
    print("\n3. Generazione audio emotivo...")

    positive_path = generate_emotional_audio(
        emotion="Positive",
        confidence=0.92,
        video_name="test_video",
        output_dir=".",
        caption="This is a test of positive emotion with Bark TTS",
        preload=True,
    )
    print(f"   ✅ Audio Positive salvato in: {positive_path}")

    negative_path = generate_emotional_audio(
        emotion="Negative",
        confidence=0.85,
        video_name="test_video",
        output_dir=".",
        caption="This is a test of negative emotion with Bark TTS",
        preload=True,
    )
    print(f"   ✅ Audio Negative salvato in: {negative_path}")

    print("\n✅ Test completato! Ascolta i file .wav generati per verificare.")
    print("   Bark genera audio molto più espressivo di edge-tts!")
