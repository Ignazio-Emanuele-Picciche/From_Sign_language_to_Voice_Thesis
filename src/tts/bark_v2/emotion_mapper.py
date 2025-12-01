"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        EMOTION MAPPER - CALIBRAZIONE ADATTIVA DELLE EMOZIONI                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

📋 DESCRIZIONE:
    Questo modulo definisce le regole di mappatura tra le predizioni numeriche del
    classificatore (Label + Confidence Score) e i parametri acustici di Bark.

    La logica è stata calibrata empiricamente analizzando la distribuzione delle
    confidenze del modello sul Test Set (vedere analisi in `analyze_tts_results.py`).

🔧 FUNZIONALITÀ CHIAVE:
    1. SELEZIONE SPEAKER DETERMINISTICA:
       - Implementa una rotazione degli speaker basata sull'hash del `video_name`.
       - Garantisce varietà nel dataset (non tutti i video "Positive" hanno la stessa voce)
         ma coerenza assoluta (lo stesso video avrà sempre la stessa voce se rigenerato).

    2. SOGLIE DI CONFIDENZA ASIMMETRICHE:
       - Le soglie per attivare i tag emotivi (es. [laughs], [sighs]) sono diverse
         per ogni emozione, riflettendo il bias del modello:
         * Positive: Soglia alta (0.90) perché il modello è molto sicuro sui positivi.
         * Negative: Soglia bassa (0.65) per compensare la cautela del modello sui negativi.

    3. TAGGING SEMANTICO:
       - Mappa livelli di certezza a risposte acustiche coerenti:
         * Alta Certezza -> Emozione esplicita (Risata, Sospiro profondo).
         * Bassa Certezza -> Segnali di dubbio (Esitazioni, Schiarimenti di voce "uhm...").

Struttura Dati:
    - EMOTION_BARK_MAPPING: Definisce il "timbro" (Speaker ID, Temperatura).
    - EMOTIONAL_TAGS: Definisce la "prosodia" (Token non verbali).
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

# Configurazione Tag Emotivi basata su Confidenza (CORRETTA)
EMOTIONAL_TAGS = {
    "Positive": {
        "primary": "[laughs]",
        "alternatives": ["[chuckles]", "[giggles]", "[laughter]", "♪"],
        "high_confidence": "[laughter]",  # >92%
        "medium_confidence": "[laughs]",  # 75-92%
        "low_confidence": "[clears throat]",  # <75% (Sicuro)
    },
    "Negative": {
        "primary": "[sighs]",
        "alternatives": ["[sad]", "[gasps]", "...", "—"],
        "high_confidence": "[sighs]",  # >65% (Ricalibrato sui tuoi dati)
        "medium_confidence": "[gasps]",  # 55-65%
        "low_confidence": "...",  # <55% (Pausa lunga è meglio di [hesitation])
    },
    "Neutral": {
        "primary": "",
        "alternatives": ["[clears throat]", "uhm...", "—"],
        "high_confidence": "",
        "medium_confidence": "[clears throat]",
        "low_confidence": "uhm...",  # <75% (Uhm... è il vero suono di esitazione)
    },
}


def get_emotional_tag(
    emotion: str, sentiment_score: int = 0, alternative: int = 0
) -> str:
    """
    Restituisce il tag Bark basato sul valore intero [-3, +3] di EmoSign.
    """
    if emotion not in EMOTIONAL_TAGS:
        return ""

    # Normalizziamo il segno per sicurezza (es. se arriva -3 ma emotion è Negative)
    score = abs(int(sentiment_score))

    tags_config = EMOTIONAL_TAGS[emotion]

    # --- LOGICA DIRETTA VALORE -> TAG ---
    if emotion == "Positive":
        if score == 3:
            return "[laughter]"  # Forte: Risata piena
        if score == 2:
            return "[laughs]"  # Medio: Risatina
        if score == 1:
            return "[laughs]"  # Debole: Tono allegro (meglio di clears throat)
        return ""  # 0: Neutro

    elif emotion == "Negative":
        if score == 3:
            return "[sighs]"  # Forte: Sospiro profondo
        if score == 2:
            return "[gasps]"  # Medio: Gasp/Shock
        if score == 1:
            return "..."  # Debole: Pausa riflessiva
        return ""  # 0: Neutro

    else:  # Neutral
        if score >= 2:
            return "[clears throat]"  # Solo se molto enfatizzato
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
