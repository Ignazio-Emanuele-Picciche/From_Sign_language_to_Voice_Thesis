"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        EMOTION MAPPER - CALIBRATO SULLA DISTRIBUZIONE DEI DATI               ║
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
        "temperature": 0.50,
    },
}

# Configurazione Tag Emotivi (SOGLIE AGGIORNATE)
EMOTIONAL_TAGS = {
    "Positive": {
        "primary": "[laughs]",
        "alternatives": ["[chuckles]", "[giggles]", "[laughter]", "♪"],
        # SOGLIA ALZATA A 0.92: Il grafico mostra molti dati > 0.90.
        # Riserviamo [laughter] (risata forte) solo ai casi estremi.
        "high_confidence": "[laughter]",
        # Fascia 0.75 - 0.92: Confidenza solida ma non estrema.
        "medium_confidence": "[laughs]",
        # Sotto 0.75: Il modello ha dubbi. Usiamo [chuckles] (sorriso) o niente.
        "low_confidence": "[chuckles]",
    },
    "Negative": {
        "primary": "[sighs]",
        "alternatives": ["[sad]", "[gasps]", "... -"],
        # SOGLIA ALZATA A 0.92
        "high_confidence": "[sighs]",
        # Fascia 0.75 - 0.92
        "medium_confidence": "...",
        # Sotto 0.75: Esitazione
        "low_confidence": "[clears throat]",
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
    if emotion not in EMOTION_BARK_MAPPING:
        emotion = "Neutral"
    config = EMOTION_BARK_MAPPING[emotion].copy()
    if "speakers" in config:
        del config["speakers"]
    if not use_emotional_tags:
        config["text_prefix"] = ""
    return config


def get_emotional_tag(
    emotion: str, confidence: float = None, alternative: int = 0
) -> str:
    if emotion not in EMOTIONAL_TAGS:
        return ""
    tags_config = EMOTIONAL_TAGS[emotion]

    if alternative > 0:
        alts = tags_config.get("alternatives", [])
        if 0 <= (alternative - 1) < len(alts):
            return alts[alternative - 1]

    if confidence is not None:
        conf = confidence if confidence <= 1.0 else confidence / 100.0

        # SOGLIE CALIBRATE SULL'ISTOGRAMMA
        if conf >= 0.92:  # Top tier (Picco estremo)
            return tags_config.get("high_confidence", "")
        elif conf >= 0.75:  # Fascia media solida
            return tags_config.get("medium_confidence", "")
        else:  # Coda bassa e incerta
            return tags_config.get("low_confidence", "")

    return tags_config.get("primary", "")


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
