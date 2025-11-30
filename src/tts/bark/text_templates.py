"""
╔══════════════════════════════════════════════════════════════════════════════╗
║            TEXT TEMPLATES - GENERAZIONE TESTI DI FALLBACK                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

📋 DESCRIZIONE:
    Questo modulo gestisce il testo da passare al TTS.
    La sua funzione principale è `get_tts_text`:
    1. Se esiste una CAPTION originale: la pulisce e la usa (Priorità Massima).
    2. Se manca la CAPTION: genera una frase sintetica basata sull'emozione
       e sulla confidenza (Fallback).

🔧 UTILIZZO:
    Usato da tts_generator.py quando il dataset non contiene trascrizioni.
"""

import random
import re


def clean_text_for_tts(text: str) -> str:
    """
    Pulisce il testo per renderlo leggibile dal TTS.
    Rimuove caratteri speciali spuri ma mantiene la punteggiatura utile per la prosodia.
    """
    if not text:
        return ""

    # Normalizzazione base
    text = str(text)
    text = text.replace("`", "'")
    text = text.replace('"', "")  # Rimuovi doppi apici per evitare confusioni

    # Rimuovi caratteri speciali che non si pronunciano
    # Nota: Manteniamo . , ! ? ; : - perché Bark li usa per le pause
    chars_to_remove = ["|", "[", "]", "{", "}", "<", ">", "_", "*", "#", "\\", "/"]
    for char in chars_to_remove:
        text = text.replace(char, " ")

    # Sostituzioni per leggibilità
    text = text.replace("@", " at ")
    text = text.replace("&", " and ")
    text = text.replace("=", " equals ")
    text = text.replace("+", " plus ")

    # Rimuovi spazi multipli
    text = re.sub(r"\s+", " ", text).strip()

    return text


# --- TEMPLATE DI FALLBACK (Usati solo se manca la caption) ---

TEMPLATES_POSITIVE = [
    "I am feeling really good about this.",
    "This is such a positive outcome!",
    "I am very happy to share this with you.",
    "Everything is looking great and positive.",
    "I'm expressing a joyful emotion right now.",
    "This brings a smile to my face.",
]

TEMPLATES_NEGATIVE = [
    "I am not feeling very good about this.",
    "This is rather disappointing.",
    "I am expressing some concern and negativity.",
    "Unfortunately, this is a negative situation.",
    "I feel quite sad about this.",
    "This is really frustrating.",
]

TEMPLATES_NEUTRAL = [
    "I am simply stating the facts.",
    "This is a neutral observation.",
    "There is nothing specific to report.",
    "I am communicating this calmly.",
    "This is just standard information.",
    "Proceeding with the neutral statement.",
]


def get_tts_text(
    emotion: str, confidence: float, video_name: str, caption: str = None
) -> str:
    """
    Genera il testo finale per il TTS.
    PRIORITÀ: Caption originale > Template generico
    """
    # 1. Se abbiamo la caption originale, usiamo quella!
    if caption and isinstance(caption, str) and len(caption.strip()) > 1:
        return clean_text_for_tts(caption)

    # 2. Fallback: Genera frase basata sull'emozione

    # Assicuriamoci che l'emozione sia valida (capitalize)
    emotion = emotion.capitalize() if emotion else "Neutral"

    if emotion == "Positive":
        text = random.choice(TEMPLATES_POSITIVE)
    elif emotion == "Negative":
        text = random.choice(TEMPLATES_NEGATIVE)
    else:  # Neutral o sconosciuto
        text = random.choice(TEMPLATES_NEUTRAL)

    return text


def get_baseline_text() -> str:
    """Testo standard per calibrazione audio."""
    return "This is a standard audio sample to test the voice synthesizer."


if __name__ == "__main__":
    print("Test generazione testi:")
    print(f"Positivo: {get_tts_text('Positive', 0.9, 'vid1', caption=None)}")
    print(f"Negativo: {get_tts_text('Negative', 0.8, 'vid2', caption=None)}")
    print(f"Neutro:   {get_tts_text('Neutral', 0.5, 'vid3', caption=None)}")
    print(f"Caption:  {get_tts_text('Positive', 0.9, 'vid4', caption='Hello world!')}")
