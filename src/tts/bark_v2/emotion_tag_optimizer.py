"""
╔══════════════════════════════════════════════════════════════════════════════╗
║      EMOTION TAG OPTIMIZER - POSIZIONAMENTO SINTATTICO DEI TAG               ║
╚══════════════════════════════════════════════════════════════════════════════╝

📋 DESCRIZIONE:
    Gestisce l'inserimento dei tag nel testo.
    Implementa logiche differenziate per intensità deboli (±1) vs forti (±3).
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
    sentiment_score: int = 0,
) -> str:
    if not use_tags or not custom_tag:
        return text

    score = abs(int(sentiment_score))
    tag = custom_tag
    breaks = find_natural_breaks(text)
    text_len = len(text)

    # --- 1. GESTIONE BASSA INTENSITÀ (Score ±1) ---

    # Negative (-1): Pausa esitante, ma non invasiva
    if emotion == "Negative" and score == 1:
        # Preferiamo mettere la pausa all'inizio o alla prima virgola
        if breaks:
            return insert_tag(text, "...", breaks[0])
        else:
            return f"... {text}"

    # Positive (+1): Solo tag all'inizio per settare il tono, mai in mezzo
    if emotion == "Positive" and score == 1:
        return f"{tag} {text}"

    # --- 2. GESTIONE ALTA INTENSITÀ (Score ±2, ±3) ---

    # Testi brevi (< 40 char): Tag sempre all'inizio per impatto immediato
    if text_len < 40:
        return f"{tag} {text}"

    # Logica specifica per Intensità Massima (+3)
    if emotion == "Positive" and score == 3 and breaks:
        # Inseriamo risata anche a metà frase
        result = f"{tag} {text}"
        # Ricalcoliamo breaks sul nuovo testo se necessario, o usiamo euristica semplice
        # Qui semplifichiamo mettendo il tag iniziale + il testo originale
        # Se volessimo essere precisi dovremmo ricalcolare gli indici,
        # ma per Bark spesso basta all'inizio.
        return f"{tag} {text}"

    # Default per medi/alti: Alla prima pausa naturale se esiste, altrimenti inizio
    if breaks:
        return insert_tag(text, tag, breaks[0])
    else:
        return f"{tag} {text}"
