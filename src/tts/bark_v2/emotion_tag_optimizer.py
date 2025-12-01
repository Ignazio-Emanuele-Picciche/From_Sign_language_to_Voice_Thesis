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
    sentiment_score: int = 0,  # <-- Rinominato da confidence
) -> str:
    if not use_tags:
        return text

    score = abs(int(sentiment_score))
    tag = custom_tag

    # Se non c'è tag (es. score 0), ritorna testo pulito
    if not tag:
        return text

    breaks = find_natural_breaks(text)
    text_len = len(text)

    # --- GESTIONE EMOZIONI DEBOLI (-1 / +1) ---

    # CASO 1: Negativo Debole (-1) -> Inseriamo pausa/esitazione
    if emotion == "Negative" and score == 1:
        hesitation = "..."
        if breaks:
            return insert_tag(text, hesitation, breaks[0])  # Pausa alla prima virgola
        else:
            return f"{hesitation} {text}"  # Pausa all'inizio

    # CASO 2: Positivo Debole (+1) -> Solo tag all'inizio, niente esitazioni
    if emotion == "Positive" and score == 1:
        return f"{tag} {text}"

    # --- GESTIONE ALTA INTENSITÀ (2, 3) ---

    # Per testi corti, mettiamo sempre il tag all'inizio
    if text_len < 40:
        return f"{tag} {text}"

    # Logica per testi lunghi e alta intensità (+3)
    if emotion == "Positive" and score == 3 and breaks:
        # Inseriamo risata anche a metà frase per enfatizzare il +3
        result = f"{tag} {text}"
        mid_break = breaks[len(breaks) // 2]
        result = insert_tag(result, "[laughter]", mid_break + len(tag) + 1)
        return result

    # Default: tag all'inizio o alla prima pausa
    if breaks:
        return insert_tag(text, tag, breaks[0])
    else:
        return f"{tag} {text}"
