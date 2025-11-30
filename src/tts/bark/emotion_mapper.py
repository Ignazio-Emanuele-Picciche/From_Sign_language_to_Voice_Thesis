"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        EMOTION MAPPER - CONFIGURAZIONE DINAMICA BARK TTS (PROD)              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import random

# Mapping Emozione -> Lista Speaker
EMOTION_BARK_MAPPING = {
    "Positive": {
        "speakers": [
            "v2/en_speaker_6",
            "v2/en_speaker_5",
            "v2/en_speaker_7",
            "v2/en_speaker_8",
        ],
        "description": "Voce allegra, energica e positiva",
        "text_prefix": "[laughs]",
        "temperature": 0.75,
    },
    "Negative": {
        "speakers": ["v2/en_speaker_3", "v2/en_speaker_1", "v2/en_speaker_4"],
        "description": "Voce calma, riflessiva o triste",
        "text_prefix": "[sighs]",
        "temperature": 0.65,
    },
    "Neutral": {
        "speakers": ["v2/en_speaker_9", "v2/en_speaker_0", "v2/en_speaker_2"],
        "description": "Voce neutra e professionale",
        "text_prefix": "",
        "temperature": 0.30,
    },
}

# Configurazione Tag Emotivi basata su Confidenza
EMOTIONAL_TAGS = {
    "Positive": {
        "primary": "[laughs]",
        "alternatives": ["[chuckles]", "[giggles]", "[laughter]"],
        "high_confidence": "[laughs]",  # >85%
        "medium_confidence": "[chuckles]",  # 65-85%
        "low_confidence": "[hesitation]",  # <65%
    },
    "Negative": {
        "primary": "[sighs]",
        "alternatives": ["[sad]", "[gasps]", "... -"],
        "high_confidence": "[sighs]",  # >85%
        "medium_confidence": "...",  # 65-85%
        "low_confidence": "[clears throat]",  # <65%
    },
    "Neutral": {
        "primary": "",
        "alternatives": ["[clears throat]", "[breath]"],
        "high_confidence": "",
        "medium_confidence": "",
        "low_confidence": "[hesitation]",
    },
}


def map_emotion_to_bark_prompt(emotion: str, use_emotional_tags: bool = True) -> dict:
    """Restituisce la config base (temperatura, desc), escluso lo speaker."""
    if emotion not in EMOTION_BARK_MAPPING:
        emotion = "Neutral"

    config = EMOTION_BARK_MAPPING[emotion].copy()

    # Rimuoviamo la lista speakers per evitare errori in Bark (gestito separatamente)
    if "speakers" in config:
        del config["speakers"]

    if not use_emotional_tags:
        config["text_prefix"] = ""

    return config


def get_emotional_tag(
    emotion: str, confidence: float = None, alternative: int = 0
) -> str:
    """Seleziona il tag in base alla confidenza."""
    if emotion not in EMOTIONAL_TAGS:
        return ""

    tags_config = EMOTIONAL_TAGS[emotion]

    # Selezione Manuale
    if alternative > 0:
        alts = tags_config.get("alternatives", [])
        if 0 <= (alternative - 1) < len(alts):
            return alts[alternative - 1]

    # Selezione basata su Confidenza
    if confidence is not None:
        conf = confidence if confidence <= 1.0 else confidence / 100.0

        if conf >= 0.85:
            return tags_config.get("high_confidence", "")
        elif conf >= 0.65:
            return tags_config.get("medium_confidence", "")
        else:
            return tags_config.get("low_confidence", "")

    return tags_config.get("primary", "")


def get_bark_speaker(emotion: str, video_name: str = None) -> str:
    """
    Seleziona uno speaker usando un hash deterministico sul video_name.
    Garantisce varietà nel dataset ma coerenza per lo stesso file.
    """
    if emotion not in EMOTION_BARK_MAPPING:
        emotion = "Neutral"

    speakers_list = EMOTION_BARK_MAPPING[emotion]["speakers"]

    if video_name:
        random_seed = sum([ord(c) for c in video_name])
        rng = random.Random(random_seed)
        return rng.choice(speakers_list)
    else:
        return speakers_list[0]
