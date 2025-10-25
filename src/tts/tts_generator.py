"""
TTS Generator - Genera audio emotivo usando edge-tts
"""

import edge_tts
import asyncio
import os
from typing import Optional
from .emotion_mapper import map_emotion_to_prosody
from .text_templates import get_tts_text, get_baseline_text


# Voce di default (puoi cambiarla)
DEFAULT_VOICE = "en-US-AriaNeural"  # Voce femminile USA, molto naturale
# Altre opzioni:
# - "en-US-GuyNeural" (maschile USA)
# - "en-GB-SoniaNeural" (femminile UK)
# - "en-GB-RyanNeural" (maschile UK)


async def _synthesize_speech_async(
    text: str, prosody_params: dict, output_path: str, voice: str = DEFAULT_VOICE
) -> str:
    """
    Funzione asincrona per sintetizzare speech con edge-tts

    Args:
        text (str): Testo da sintetizzare
        prosody_params (dict): Parametri prosodici {'rate': '+15%', 'pitch': '+8%', 'volume': '+5%'}
        output_path (str): Path dove salvare il file audio
        voice (str): Nome della voce edge-tts

    Returns:
        str: Path del file audio generato
    """

    # Edge-TTS richiede formati specifici:
    # - rate: ±X% (percentuale)
    # - pitch: ±XHz (Hertz, non percentuale!)
    # - volume: ±X% (percentuale)

    import re

    def convert_rate_volume(param_str: str) -> str:
        """Converte '+14.2%' in '+14%' (solo interi)"""
        match = re.match(r"([+-])(\d+\.?\d*)(%)", param_str)
        if match:
            sign, value, percent = match.groups()
            return f"{sign}{int(float(value))}{percent}"
        return param_str

    def convert_pitch_to_hz(param_str: str) -> str:
        """Converte percentuale in Hz: '+8%' -> '+12Hz' (assumendo baseline ~150Hz)"""
        match = re.match(r"([+-])(\d+\.?\d*)(%)", param_str)
        if match:
            sign, value_str, _ = match.groups()
            percent_value = float(value_str)
            # Conversione: assumiamo pitch medio di 150Hz
            # +10% di 150Hz = +15Hz
            hz_value = int((percent_value / 100.0) * 150.0)
            return f"{sign}{hz_value}Hz"
        return param_str

    rate_param = convert_rate_volume(prosody_params["rate"])
    pitch_param = convert_pitch_to_hz(prosody_params["pitch"])
    volume_param = convert_rate_volume(prosody_params["volume"])

    # Genera audio - passa i parametri prosodici direttamente al costruttore
    # Edge-TTS gestisce internamente la conversione in SSML
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=rate_param,
        pitch=pitch_param,
        volume=volume_param,
    )
    await communicate.save(output_path)

    return output_path


def generate_emotional_audio(
    emotion: str,
    confidence: float,
    video_name: str,
    output_dir: str,
    voice: str = DEFAULT_VOICE,
    use_simple_template: bool = False,
    caption: str = None,
) -> str:
    """
    Genera audio emotivo per un video analizzato

    Args:
        emotion (str): Emozione predetta ('Positive', 'Negative')
        confidence (float): Confidenza della predizione (0.0-1.0 o 0-100)
        video_name (str): Nome del video (per naming del file)
        output_dir (str): Directory dove salvare l'audio
        voice (str): Voce edge-tts da usare
        use_simple_template (bool): Se True, usa solo nome emozione come testo
        caption (str, optional): Testo originale del sign language da pronunciare

    Returns:
        str: Path completo del file audio generato

    Example:
        >>> generate_emotional_audio('Positive', 0.92, 'video_001', 'results/tts_audio', caption='Hello world')
        'results/tts_audio/video_001_positive.mp3'
    """
    # Crea directory se non esiste
    os.makedirs(output_dir, exist_ok=True)

    # Mappa emozione a prosody
    prosody_params = map_emotion_to_prosody(
        emotion, confidence if confidence <= 1.0 else confidence / 100
    )

    # Genera testo
    if use_simple_template:
        from .text_templates import get_simple_tts_text

        text = get_simple_tts_text(emotion)
    else:
        # Passa caption se disponibile
        text = get_tts_text(emotion, confidence, video_name, caption=caption)

    # Path output
    safe_video_name = video_name.replace("/", "_").replace("\\", "_")
    output_filename = f"{safe_video_name}_{emotion.lower()}.mp3"
    output_path = os.path.join(output_dir, output_filename)

    # Genera audio (sincrono wrapper)
    asyncio.run(_synthesize_speech_async(text, prosody_params, output_path, voice))

    return output_path


def generate_baseline_audio(output_path: str, voice: str = DEFAULT_VOICE) -> str:
    """
    Genera audio baseline neutrale (senza modulazione prosodica)
    Serve come riferimento per comparazioni

    Args:
        output_path (str): Path completo dove salvare il file
        voice (str): Voce edge-tts da usare

    Returns:
        str: Path del file audio generato

    Example:
        >>> generate_baseline_audio('results/tts_audio/baseline/baseline_neutral.mp3')
        'results/tts_audio/baseline/baseline_neutral.mp3'
    """
    # Crea directory se non esiste
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Parametri neutrali (nessuna modulazione)
    neutral_prosody = {"rate": "+0%", "pitch": "+0%", "volume": "+0%"}

    # Testo baseline
    text = get_baseline_text()

    # Genera audio
    asyncio.run(_synthesize_speech_async(text, neutral_prosody, output_path, voice))

    return output_path


async def list_available_voices():
    """
    Lista tutte le voci disponibili in edge-tts
    Utile per scegliere la voce migliore
    """
    voices = await edge_tts.list_voices()

    # Filtra solo voci inglesi
    en_voices = [v for v in voices if v["Locale"].startswith("en-")]

    print(f"Voci inglesi disponibili: {len(en_voices)}")
    print("\nAlcune voci consigliate:")
    print("-" * 60)

    recommended = [
        "en-US-AriaNeural",
        "en-US-GuyNeural",
        "en-GB-SoniaNeural",
        "en-GB-RyanNeural",
    ]

    for voice in voices:
        if voice["ShortName"] in recommended:
            print(f"{voice['ShortName']:30} - {voice['Gender']:10} ({voice['Locale']})")


if __name__ == "__main__":
    # Test del modulo
    print("=" * 60)
    print("TEST TTS GENERATOR")
    print("=" * 60)

    # Test 1: Lista voci
    print("\n1. Liste voci disponibili:")
    asyncio.run(list_available_voices())

    # Test 2: Genera baseline
    print("\n2. Generazione audio baseline...")
    baseline_path = generate_baseline_audio("test_baseline.mp3")
    print(f"   ✅ Baseline salvato in: {baseline_path}")

    # Test 3: Genera audio emotivo
    print("\n3. Generazione audio emotivo...")
    positive_path = generate_emotional_audio(
        emotion="Positive", confidence=0.92, video_name="test_video", output_dir="."
    )
    print(f"   ✅ Audio Positive salvato in: {positive_path}")

    negative_path = generate_emotional_audio(
        emotion="Negative", confidence=0.85, video_name="test_video", output_dir="."
    )
    print(f"   ✅ Audio Negative salvato in: {negative_path}")

    print("\n✅ Test completato! Ascolta i file generati per verificare.")
