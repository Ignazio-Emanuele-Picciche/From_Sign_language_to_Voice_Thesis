"""
╔══════════════════════════════════════════════════════════════════════════════╗
║      EMOTION TAG OPTIMIZER - SOGLIE AGGIORNATE SU DISTRIBUZIONE DATI         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import re
from typing import List


def find_natural_breaks(text: str) -> List[int]:
    breaks = []
    for match in re.finditer(r"[.,!?;]\s+", text):
        breaks.append(match.end())
    for match in re.finditer(r"\b(and|but|so|because|however)\s+", text, re.IGNORECASE):
        breaks.append(match.end())
    return sorted(breaks)


def insert_tag(text: str, tag: str, pos: int) -> str:
    return f"{text[:pos].rstrip()} {tag} {text[pos:].lstrip()}"


def optimize_emotional_text(
    text: str,
    emotion: str,
    use_tags: bool = True,
    custom_tag: str = None,
    confidence: float = 1.0,
) -> str:
    if not use_tags:
        return text

    conf = confidence if confidence <= 1.0 else confidence / 100.0

    default_tags = {
        "Positive": "[laughs]",
        "Negative": "[sighs]",
        "Neutral": "[clears throat]",
    }
    tag = custom_tag or default_tags.get(emotion, "")
    if not tag:
        return text

    breaks = find_natural_breaks(text)
    text_len = len(text)

    # --- STRATEGIA BASSA CONFIDENZA (< 0.75) ---
    # Il grafico mostrava un cluster tra 0.60 e 0.70.
    # Trattiamo questa fascia come "incerta" -> Esitazione invece di emozione.
    if conf < 0.75:
        if breaks:
            return insert_tag(text, "... [hesitation]", breaks[0])
        else:
            return f"[hesitation] {text}"

    # --- STRATEGIA ALTA CONFIDENZA ---

    if text_len < 40:
        return f"{tag} {text}"

    if emotion == "Positive":
        result = f"{tag} {text}"
        # Solo se > 0.95 (certezza quasi assoluta) raddoppiamo la risata
        if conf > 0.95 and breaks:
            mid_break = breaks[len(breaks) // 2]
            result = insert_tag(result, "[laughter]", mid_break + len(tag) + 1)
        return result

    elif emotion == "Negative":
        if breaks:
            return insert_tag(text, tag, breaks[0])
        else:
            return f"{tag} {text}"

    else:  # Neutral
        return f"{tag} {text}" if text_len > 100 else text
