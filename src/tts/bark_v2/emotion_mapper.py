"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        EMOTION MAPPER - SCALA EMOSIGN [-3, +3]                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

📋 DESCRIZIONE:
    Mapping diretto e deterministico tra i valori di intensità del dataset EmoSign
    e i parametri acustici di Bark.

    SCALE MAPPING:
    +3 (Forte)  -> [laughter] + Temp 0.75
    +2 (Medio)  -> [laughs]   + Temp 0.75
    +1 (Debole) -> [laughs]   + Temp 0.75 (Senza esitazioni)
     0 (Neutro) -> (Nessuno)  + Temp 0.50
    -1 (Debole) -> ... (Pausa) + Temp 0.65
    -2 (Medio)  -> [gasps]    + Temp 0.65
    -3 (Forte)  -> [sighs]    + Temp 0.65
"""

import random

# Mapping Emozione -> Parametri Voce
EMOTION_BARK_MAPPING = {
    "Positive": {
        "speakers": ["v2/en_speaker_6", "v2/en_speaker_5", "v2/en_speaker_8"],
        "temperature": 0.65,
    },
    "Negative": {
        "speakers": ["v2/en_speaker_3", "v2/en_speaker_1", "v2/en_speaker_4"],
        "temperature": 0.45,
    },
    "Neutral": {
        "speakers": ["v2/en_speaker_9", "v2/en_speaker_0"],
        "temperature": 0.30,
    },
}


def map_emotion_to_bark_prompt(emotion: str, use_emotional_tags: bool = True):
    """Ritorna la configurazione base per l'emozione."""
    if emotion not in EMOTION_BARK_MAPPING:
        return EMOTION_BARK_MAPPING["Neutral"]
    return EMOTION_BARK_MAPPING[emotion]


def get_emotional_tag(emotion: str, sentiment_score: int = 0) -> str:
    """
    Traduce lo score [-3, +3] in tag Bark specifici e variati.
    """
    score = abs(int(sentiment_score))

    if emotion == "Positive":
        if score == 3:
            return "[laughter]"  # Intensità Alta: Risata piena
        if score == 2:
            return "[laughs]"  # Intensità Media: Risata normale
        if score == 1:
            return "[chuckles]"  # Intensità Bassa: Ridacchiare (più sottile)
        return ""

    elif emotion == "Negative":
        if score == 3:
            return "[sighs]"  # Intensità Alta: Sospiro profondo (Disperazione)
        if score == 2:
            return "[gasps]"  # Intensità Media: Shock/Sorpresa negativa
        if score == 1:
            return "..."  # Intensità Bassa: Pausa riflessiva/Esitazione
            # Alternativa per -1: return "[clears throat]" (se vuoi imbarazzo)
        return ""

    else:  # Neutral
        return ""


def get_bark_speaker(emotion: str, video_name: str = None) -> str:
    if emotion not in EMOTION_BARK_MAPPING:
        emotion = "Neutral"
    speakers_list = EMOTION_BARK_MAPPING[emotion]["speakers"]

    if video_name:
        random_seed = sum([ord(c) for c in video_name])
        rng = random.Random(random_seed)
        return rng.choice(speakers_list)
    else:
        return speakers_list[0]
