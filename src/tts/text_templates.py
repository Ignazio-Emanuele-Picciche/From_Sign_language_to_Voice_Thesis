"""
Text Templates - Genera testi per TTS basati su emozione e contesto
"""

import random
import re


def clean_text_for_tts(text: str) -> str:
    """
    Pulisce il testo per renderlo adatto al TTS
    Rimuove caratteri speciali che edge-tts leggerebbe letteralmente

    Args:
        text (str): Testo grezzo da pulire

    Returns:
        str: Testo pulito per TTS
    """
    if not text:
        return text

    # Sostituisci backtick con apostrofo normale
    text = text.replace("`", "'")

    # Sostituisci virgolette curve con virgolette dritte
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(""", "'").replace(""", "'")

    # Rimuovi doppi apici (vengono letti come "quote")
    # Ma mantieni gli apostrofi per contrazioni (don't, it's, etc.)
    # Strategia: rimuovi " all'inizio e alla fine di parole
    text = re.sub(r'\s+"', " ", text)  # " dopo spazio
    text = re.sub(r'"\s+', " ", text)  # " prima di spazio
    text = re.sub(r'^"', "", text)  # " all'inizio
    text = re.sub(r'"$', "", text)  # " alla fine

    # Rimuovi caratteri speciali problematici
    text = text.replace("/", " ")  # slash
    text = text.replace("\\", " ")  # backslash
    text = text.replace("|", " ")  # pipe
    text = text.replace("[", "")  # parentesi quadre
    text = text.replace("]", "")
    text = text.replace("{", "")  # parentesi graffe
    text = text.replace("}", "")
    text = text.replace("<", "")  # tag HTML
    text = text.replace(">", "")
    text = text.replace("_", " ")  # underscore
    text = text.replace("*", "")  # asterisco
    text = text.replace("#", "")  # hash
    text = text.replace("@", " at ")  # at
    text = text.replace("&", " and ")  # ampersand

    # Normalizza spazi multipli
    text = re.sub(r"\s+", " ", text)

    # Trim
    text = text.strip()

    return text


# Template variati per rendere gli audio più distintivi
TEMPLATES_POSITIVE = [
    "This sign expresses a positive emotion with {confidence:.1f}% confidence",
    "Detected positive emotional content, confidence level {confidence:.1f}%",
    "The signer shows a positive emotional state, {confidence:.1f}% certain",
    "Positive emotion recognized with {confidence:.1f}% accuracy",
    "Analysis indicates positive emotional expression, {confidence:.1f}% confidence",
]

TEMPLATES_NEGATIVE = [
    "This sign expresses a negative emotion with {confidence:.1f}% confidence",
    "Detected negative emotional content, confidence level {confidence:.1f}%",
    "The signer shows a negative emotional state, {confidence:.1f}% certain",
    "Negative emotion recognized with {confidence:.1f}% accuracy",
    "Analysis indicates negative emotional expression, {confidence:.1f}% confidence",
]


def get_tts_text(
    emotion: str, confidence: float, video_name: str, caption: str = None
) -> str:
    """
    Genera il testo da utilizzare per il TTS
    Se è disponibile un caption, usa quello (pulito), altrimenti usa un template

    Args:
        emotion (str): Emozione predetta (Positive o Negative)
        confidence (float): Livello di confidenza (0-1)
        video_name (str): Nome del video
        caption (str, optional): Testo della caption originale

    Returns:
        str: Testo da sintetizzare con TTS
    """
    # Se è disponibile un caption, puliscilo e usalo (priorità massima)
    if caption:
        cleaned = clean_text_for_tts(caption)
        return (
            cleaned if cleaned else caption
        )  # Fallback se la pulizia restituisce stringa vuota


def get_simple_tts_text(emotion: str) -> str:
    """
    Genera un testo semplice solo con l'emozione

    Args:
        emotion (str): Emozione predetta

    Returns:
        str: Testo semplice

    Example:
        >>> get_simple_tts_text('Positive')
        'Positive'
    """
    return emotion


def get_detailed_tts_text(
    emotion: str, confidence: float, video_name: str = None
) -> str:
    """
    Genera un testo dettagliato con informazioni aggiuntive

    Args:
        emotion (str): Emozione predetta
        confidence (float): Confidenza della predizione
        video_name (str, optional): Nome del video

    Returns:
        str: Testo dettagliato

    Example:
        >>> get_detailed_tts_text('Positive', 0.92, 'video_001')
        'Analysis of video_001 shows a positive emotional expression...'
    """
    # Normalizza confidence
    if confidence <= 1.0:
        confidence = confidence * 100

    if video_name:
        text = (
            f"Analysis of {video_name} shows a {emotion.lower()} emotional expression "
        )
    else:
        text = f"This analysis shows a {emotion.lower()} emotional expression "

    # Aggiungi livello di confidenza
    if confidence >= 90:
        conf_level = "very high"
    elif confidence >= 70:
        conf_level = "high"
    elif confidence >= 50:
        conf_level = "moderate"
    else:
        conf_level = "low"

    text += f"with {conf_level} confidence at {confidence:.1f}%"

    return text


def get_baseline_text() -> str:
    """
    Testo neutrale per baseline audio

    Returns:
        str: Testo neutrale standard
    """
    return "This is a neutral baseline audio sample for comparison purposes"


if __name__ == "__main__":
    # Test del modulo
    print("=" * 60)
    print("TEST TEXT TEMPLATES")
    print("=" * 60)

    print("\n1. Template standard:")
    print(get_tts_text("Positive", 92.5, "video_001"))

    print("\n2. Template semplice:")
    print(get_simple_tts_text("Negative"))

    print("\n3. Template dettagliato:")
    print(get_detailed_tts_text("Positive", 0.92, "video_001"))
    print(get_detailed_tts_text("Negative", 0.65))

    print("\n4. Template baseline:")
    print(get_baseline_text())
