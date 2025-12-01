"""
╔══════════════════════════════════════════════════════════════════════════════╗
║      EMOTION TAG OPTIMIZER - POSIZIONAMENTO SINTATTICO DEI TAG               ║
╚══════════════════════════════════════════════════════════════════════════════╝

📋 DESCRIZIONE:
    Modulo responsabile dell'iniezione intelligente dei tag emotivi (es. [laughs])
    all'interno del testo. Invece di preporre ciecamente il tag all'inizio della frase,
    questo script analizza la struttura sintattica per posizionarli in modo naturale.

🧠 LOGICA ALGORITMICA:
    1. ANALISI DELLE PAUSE NATURALI:
       - Scansiona il testo cercando punti di respiro naturali: punteggiatura (. , ! ?)
         e congiunzioni avversative o coordinanti (but, and, so, however).

    2. STRATEGIE DI INIEZIONE (Context-Aware):
       - Testi Brevi: Tag all'inizio per impostare subito il tono.
       - Testi Lunghi + Alta Confidenza: Inserimento "a tenaglia" (inizio e metà frase)
         per mantenere l'emozione viva durante tutta l'esecuzione.
       - Emozioni Negative: Preferisce inserire i sospiri (`[sighs]`) in corrispondenza
         delle pause intermedie, simulando un parlato rotto o drammatico.

    3. GESTIONE DELL'INCERTEZZA (Low Confidence Handling):
       - Se la confidenza è bassa (< 0.75), converte i tag di emozione forte in
         marcatori di esitazione (`uhm...`, `...`), istruendo Bark a generare
         un parlato più incerto e meno assertivo.

🔧 UTILIZZO:
    Chiamato da `tts_generator.py` subito prima della sintesi audio.
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

    # Tag base (fallback)
    default_tags = {
        "Positive": "[laughs]",
        "Negative": "[sighs]",
        "Neutral": "[clears throat]",
    }
    tag = custom_tag or default_tags.get(emotion, "")

    # Se il tag passato era il vecchio [hesitation], lo convertiamo al volo
    if tag == "[hesitation]":
        tag = "uhm..."

    if not tag:
        return text

    breaks = find_natural_breaks(text)
    text_len = len(text)

    # --- STRATEGIA BASSA CONFIDENZA (< 0.75) ---
    # Usiamo "uhm..." o "..." per simulare incertezza acustica reale
    if conf < 0.75:
        hesitation_sound = "uhm..."  # Molto efficace in Bark
        if breaks:
            # Inserisci un'esitazione alla prima pausa
            return insert_tag(text, f"... {hesitation_sound}", breaks[0])
        else:
            # Inizio frase
            return f"{hesitation_sound} {text}"

    # --- STRATEGIA ALTA CONFIDENZA ---
    # ... (Il resto rimane uguale) ...

    if text_len < 40:
        return f"{tag} {text}"

    if emotion == "Positive":
        result = f"{tag} {text}"
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
