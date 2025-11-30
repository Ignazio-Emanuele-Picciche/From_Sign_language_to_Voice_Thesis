"""
╔══════════════════════════════════════════════════════════════════════════════╗
║      EMOTION TAG OPTIMIZER - POSIZIONAMENTO CONTEXT-AWARE (PROD)             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import re
from typing import List


def find_natural_breaks(text: str) -> List[int]:
    """Trova posizioni naturali (punteggiatura, congiunzioni)."""
    breaks = []
    for match in re.finditer(r"[.,!?;]\s+", text):
        breaks.append(match.end())
    for match in re.finditer(r"\b(and|but|so|because|however)\s+", text, re.IGNORECASE):
        breaks.append(match.end())
    return sorted(breaks)


def insert_tag(text: str, tag: str, pos: int) -> str:
    """Inserisce tag gestendo gli spazi."""
    return f"{text[:pos].rstrip()} {tag} {text[pos:].lstrip()}"


def optimize_emotional_text(
    text: str,
    emotion: str,
    use_tags: bool = True,
    custom_tag: str = None,
    confidence: float = 1.0,
) -> str:
    """
    Ottimizza il testo inserendo tag in base a emozione e confidenza.
    """
    if not use_tags:
        return text

    # Normalizza confidenza
    conf = confidence if confidence <= 1.0 else confidence / 100.0

    # Tag base di fallback
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

    # --- STRATEGIA BASSA CONFIDENZA (< 0.60) ---
    # Inserisce esitazioni invece di emozioni forti
    if conf < 0.60:
        if breaks:
            return insert_tag(text, "... [hesitation]", breaks[0])
        else:
            return f"[hesitation] {text}"

    # --- STRATEGIA STANDARD ---

    # 1. Testo Corto
    if text_len < 40:
        return f"{tag} {text}"

    # 2. Testo Medio/Lungo
    if emotion == "Positive":
        result = f"{tag} {text}"
        # Se ALTA CONFIDENZA (>0.90) e testo lungo: raddoppia l'emozione
        if conf > 0.90 and breaks:
            mid_break = breaks[len(breaks) // 2]
            result = insert_tag(result, "[chuckles]", mid_break + len(tag) + 1)
        return result

    elif emotion == "Negative":
        # Negative: preferisce interrompere la frase (più drammatico)
        if breaks:
            return insert_tag(text, tag, breaks[0])
        else:
            return f"{tag} {text}"

    else:  # Neutral
        return f"{tag} {text}" if text_len > 100 else text
