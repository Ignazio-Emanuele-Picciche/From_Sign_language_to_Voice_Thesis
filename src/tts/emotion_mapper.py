"""
Emotion Mapper - Mappa emozioni a parametri prosodici
"""

# Definizione parametri prosodici per ogni emozione
PROSODY_MAPPING = {
    "Positive": {
        "rate": "+22%",  # Velocità di eloquio, prima era +15%
        "pitch": "+12%",  # Altezza del tono, prima era +8%
        "volume": "+8%",  # Volume della voce, prima era +5%
        "description": "Voce energica, allegra, veloce",
    },
    "Negative": {
        "rate": "-18%",  # Prima era -12%
        "pitch": "-9%",  # Prima era -6%
        "volume": "-5%",  # Prima era -3%
        "description": "Voce lenta, triste, contenuta",
    },
    "Neutral": {
        "rate": "+0%",
        "pitch": "+0%",
        "volume": "+0%",
        "description": "Voce neutra senza modulazione",
    },
}


def map_emotion_to_prosody(emotion: str, confidence: float = 1.0) -> dict:
    """
    Mappa un'emozione a parametri prosodici per TTS

    Args:
        emotion (str): Emozione predetta ('Positive', 'Negative', 'Neutral')
        confidence (float): Confidenza della predizione (0.0-1.0)
                           Scala l'intensità della modulazione

    Returns:
        dict: Dizionario con parametri prosodici
              {'rate': str, 'pitch': str, 'volume': str, 'description': str}

    Example:
        >>> map_emotion_to_prosody('Positive', 0.9)
        {'rate': '+13.5%', 'pitch': '+7.2%', 'volume': '+4.5%', ...}
    """
    if emotion not in PROSODY_MAPPING:
        raise ValueError(
            f"Emozione '{emotion}' non riconosciuta. Usa: {list(PROSODY_MAPPING.keys())}"
        )

    base_prosody = PROSODY_MAPPING[emotion].copy()

    # Se confidence < 1.0, scala i parametri proporzionalmente
    if confidence < 1.0:
        for param in ["rate", "pitch", "volume"]:
            value_str = base_prosody[param]
            # Estrai il valore numerico (es: '+15%' -> 15)
            sign = value_str[0]  # '+' o '-'
            value = float(value_str[1:-1])  # Rimuovi segno e %

            # Scala per confidence
            scaled_value = value * confidence

            # Ricostruisci la stringa
            if scaled_value >= 0:
                base_prosody[param] = f"+{scaled_value:.1f}%"
            else:
                base_prosody[param] = f"{scaled_value:.1f}%"

    return base_prosody


def get_prosody_values(emotion: str, confidence: float = 1.0) -> tuple:
    """
    Ottiene i valori numerici dei parametri prosodici

    Returns:
        tuple: (rate_value, pitch_value, volume_value) come float
               Positivo = aumento, Negativo = diminuzione

    Example:
        >>> get_prosody_values('Positive', 1.0)
        (15.0, 8.0, 5.0)
    """
    prosody = map_emotion_to_prosody(emotion, confidence)

    def parse_value(value_str: str) -> float:
        """Converte '+15%' -> 15.0, '-12%' -> -12.0"""
        return float(value_str.replace("%", ""))

    rate_val = parse_value(prosody["rate"])
    pitch_val = parse_value(prosody["pitch"])
    volume_val = parse_value(prosody["volume"])

    return rate_val, pitch_val, volume_val


if __name__ == "__main__":
    # Test del modulo
    print("=" * 60)
    print("TEST EMOTION MAPPER")
    print("=" * 60)

    emotions = ["Positive", "Negative", "Neutral"]
    confidences = [1.0, 0.8, 0.5]

    for emotion in emotions:
        print(f"\nEmozione: {emotion}")
        for conf in confidences:
            prosody = map_emotion_to_prosody(emotion, conf)
            print(
                f"  Confidence {conf:.1f}: rate={prosody['rate']}, "
                f"pitch={prosody['pitch']}, volume={prosody['volume']}"
            )
