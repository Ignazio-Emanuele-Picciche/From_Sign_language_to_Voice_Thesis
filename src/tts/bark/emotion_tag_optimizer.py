"""
Emotion Tag Optimizer - Posizionamento intelligente dei tag emotivi in Bark TTS

Questo modulo ottimizza DOVE inserire i tag emotivi ([laughs], [sighs], etc.)
nel testo per massimizzare l'espressività dell'audio generato.

Strategia:
- Positive: Risate all'inizio o dopo pause naturali (più spontaneo)
- Negative: Sospiri a metà o fine frase (più drammatico)
- Neutral: Schiarimento voce solo se necessario
"""

import re
from typing import List, Tuple


def find_natural_breaks(text: str) -> List[int]:
    """
    Trova posizioni naturali per inserire tag emotivi

    Cerca:
    - Dopo punteggiatura (. , ! ? ;)
    - Dopo congiunzioni (and, but, so, because)
    - A metà di frasi lunghe

    Args:
        text (str): Testo da analizzare

    Returns:
        List[int]: Indici delle posizioni dove inserire tag
    """
    breaks = []

    # Dopo punteggiatura seguita da spazio
    for match in re.finditer(r"[.,!?;]\s+", text):
        breaks.append(match.end())

    # Dopo congiunzioni comuni
    for match in re.finditer(
        r"\b(and|but|so|because|however|therefore)\s+", text, re.IGNORECASE
    ):
        breaks.append(match.end())

    # Se frase molto lunga (>100 char) senza punteggiatura, aggiungi break a metà
    if len(text) > 100 and not breaks:
        words = text.split()
        mid_point = len(words) // 2
        # Trova posizione del mid_point-esimo spazio
        word_count = 0
        for i, char in enumerate(text):
            if char.isspace():
                word_count += 1
                if word_count == mid_point:
                    breaks.append(i + 1)
                    break

    return sorted(breaks)


def insert_tag_at_position(text: str, tag: str, position: int) -> str:
    """
    Inserisce un tag emotivo in una posizione specifica

    Args:
        text (str): Testo originale
        tag (str): Tag da inserire (es. "[laughs]")
        position (int): Indice dove inserire

    Returns:
        str: Testo con tag inserito
    """
    # Assicurati che ci sia uno spazio prima e dopo
    before = text[:position].rstrip()
    after = text[position:].lstrip()

    return f"{before} {tag} {after}"


def optimize_positive_tags(text: str, base_tag: str = "[laughs]") -> str:
    """
    Ottimizza il posizionamento di tag positivi (risate)

    Strategia:
    - Testo corto (<40 char): tag all'inizio (spontaneo)
    - Testo medio (40-100 char): tag dopo prima pausa naturale
    - Testo lungo (>100 char): tag all'inizio + dopo metà frase

    Args:
        text (str): Testo originale
        base_tag (str): Tag emotivo da usare

    Returns:
        str: Testo ottimizzato con tag
    """
    text_len = len(text)

    # Caso 1: Testo corto - tag all'inizio
    if text_len < 40:
        return f"{base_tag} {text}"

    # Caso 2: Testo medio - tag dopo prima pausa naturale
    elif text_len < 100:
        breaks = find_natural_breaks(text)
        if breaks:
            # Inserisci dopo la prima pausa
            return insert_tag_at_position(text, base_tag, breaks[0])
        else:
            # Nessuna pausa - metti all'inizio
            return f"{base_tag} {text}"

    # Caso 3: Testo lungo - tag all'inizio + eventualmente a metà
    else:
        breaks = find_natural_breaks(text)

        # Inizia con tag all'inizio
        result = f"{base_tag} {text}"

        # Se ci sono molte pause (>2), aggiungi un secondo tag a metà
        if len(breaks) > 2:
            mid_break = breaks[len(breaks) // 2]
            result = insert_tag_at_position(
                result, base_tag, mid_break + len(base_tag) + 2
            )

        return result


def optimize_negative_tags(text: str, base_tag: str = "[sighs]") -> str:
    """
    Ottimizza il posizionamento di tag negativi (sospiri)

    Strategia:
    - Testo corto: tag all'inizio
    - Testo medio: tag a metà frase (più drammatico)
    - Testo lungo: tag all'inizio + prima della fine

    Args:
        text (str): Testo originale
        base_tag (str): Tag emotivo da usare

    Returns:
        str: Testo ottimizzato con tag
    """
    text_len = len(text)

    # Caso 1: Testo corto - tag all'inizio
    if text_len < 40:
        return f"{base_tag} {text}"

    # Caso 2: Testo medio - tag a metà (più impatto emotivo)
    elif text_len < 100:
        breaks = find_natural_breaks(text)
        if breaks and len(breaks) > 1:
            # Inserisci a metà delle pause
            mid_idx = len(breaks) // 2
            return insert_tag_at_position(text, base_tag, breaks[mid_idx])
        elif breaks:
            # Una sola pausa - usala
            return insert_tag_at_position(text, base_tag, breaks[0])
        else:
            # Nessuna pausa - metti all'inizio
            return f"{base_tag} {text}"

    # Caso 3: Testo lungo - tag all'inizio + verso la fine
    else:
        breaks = find_natural_breaks(text)

        # Inizia con tag all'inizio
        result = f"{base_tag} {text}"

        # Aggiungi un secondo tag verso la fine (75% del testo)
        if len(breaks) > 2:
            # Trova la pausa più vicina al 75% del testo
            target_pos = int(len(text) * 0.75)
            closest_break = min(breaks, key=lambda x: abs(x - target_pos))
            result = insert_tag_at_position(
                result, base_tag, closest_break + len(base_tag) + 2
            )

        return result


def optimize_neutral_tags(text: str, base_tag: str = "[clears throat]") -> str:
    """
    Ottimizza il posizionamento di tag neutrali

    Strategia:
    - Usa tag solo se testo lungo (>80 char)
    - Inserisci solo all'inizio
    - Non esagerare con i tag per mantenere neutralità

    Args:
        text (str): Testo originale
        base_tag (str): Tag emotivo da usare

    Returns:
        str: Testo ottimizzato con tag (o invariato)
    """
    # Per neutral, usa tag solo se testo molto lungo
    if len(text) > 80:
        return f"{base_tag} {text}"
    else:
        # Testo corto/medio - nessun tag (mantieni neutralità)
        return text


def optimize_emotional_text(
    text: str, emotion: str, use_tags: bool = True, custom_tag: str = None
) -> str:
    """
    Funzione principale: ottimizza il testo con tag emotivi posizionati strategicamente

    Args:
        text (str): Testo originale
        emotion (str): Emozione ('Positive', 'Negative', 'Neutral')
        use_tags (bool): Se False, ritorna testo invariato
        custom_tag (str, optional): Tag personalizzato invece di quello di default

    Returns:
        str: Testo ottimizzato

    Example:
        >>> optimize_emotional_text("Hello world", "Positive")
        '[laughs] Hello world'

        >>> optimize_emotional_text("This is a longer sentence with more content to analyze", "Negative")
        '[sighs] This is a longer sentence with [sighs] more content to analyze'
    """
    if not use_tags:
        return text

    # Default tags per emozione
    default_tags = {
        "Positive": "[laughs]",
        "Negative": "[sighs]",
        "Neutral": "[clears throat]",
    }

    # Usa custom tag o default
    tag = custom_tag or default_tags.get(emotion, "")

    if not tag:
        return text

    # Applica strategia ottimale per emozione
    if emotion == "Positive":
        return optimize_positive_tags(text, tag)
    elif emotion == "Negative":
        return optimize_negative_tags(text, tag)
    elif emotion == "Neutral":
        return optimize_neutral_tags(text, tag)
    else:
        # Emozione sconosciuta - tag all'inizio
        return f"{tag} {text}"


def get_alternative_tags(emotion: str) -> List[str]:
    """
    Ottieni tag emotivi alternativi per variare l'espressività

    Args:
        emotion (str): Emozione

    Returns:
        List[str]: Lista di tag alternativi
    """
    tags = {
        "Positive": ["[laughs]", "[giggles]", "[chuckles]"],
        "Negative": ["[sighs]", "[gasps]", "[sad]"],
        "Neutral": ["[clears throat]", ""],  # Vuoto per evitare tag
    }

    return tags.get(emotion, [""])


if __name__ == "__main__":
    # Test del modulo
    print("=" * 80)
    print("TEST EMOTION TAG OPTIMIZER")
    print("=" * 80)

    # Test 1: Testi corti
    print("\n1. TESTI CORTI (<40 char)")
    print("-" * 80)

    short_text = "Hello world"
    print(f"Originale: {short_text}")
    print(f"Positive:  {optimize_emotional_text(short_text, 'Positive')}")
    print(f"Negative:  {optimize_emotional_text(short_text, 'Negative')}")
    print(f"Neutral:   {optimize_emotional_text(short_text, 'Neutral')}")

    # Test 2: Testi medi
    print("\n2. TESTI MEDI (40-100 char)")
    print("-" * 80)

    medium_text = "This is a medium length sentence, with some punctuation and pauses."
    print(f"Originale: {medium_text}")
    print(f"Positive:  {optimize_emotional_text(medium_text, 'Positive')}")
    print(f"Negative:  {optimize_emotional_text(medium_text, 'Negative')}")
    print(f"Neutral:   {optimize_emotional_text(medium_text, 'Neutral')}")

    # Test 3: Testi lunghi
    print("\n3. TESTI LUNGHI (>100 char)")
    print("-" * 80)

    long_text = (
        "This is a much longer sentence that contains multiple clauses, "
        "and it has several natural breaks where emotional tags could be inserted. "
        "The optimizer should find the best positions for maximum expressiveness."
    )
    print(f"Originale: {long_text}")
    print(f"Positive:  {optimize_emotional_text(long_text, 'Positive')}")
    print(f"Negative:  {optimize_emotional_text(long_text, 'Negative')}")
    print(f"Neutral:   {optimize_emotional_text(long_text, 'Neutral')}")

    # Test 4: Natural breaks detection
    print("\n4. NATURAL BREAKS DETECTION")
    print("-" * 80)

    test_sentences = [
        "Hello, world! How are you?",
        "This sentence and that sentence but also this one",
        "VeryLongSentenceWithoutAnyPunctuationOrBreaksToTestEdgeCases" * 3,
    ]

    for sent in test_sentences:
        breaks = find_natural_breaks(sent)
        print(f"\nText: {sent[:60]}...")
        print(f"Breaks at positions: {breaks}")
        if breaks:
            for b in breaks:
                print(
                    f"  Position {b}: ...{sent[max(0,b-10):b]}|{sent[b:min(len(sent),b+10)]}..."
                )

    # Test 5: Tag alternativi
    print("\n5. TAG ALTERNATIVI")
    print("-" * 80)

    for emotion in ["Positive", "Negative", "Neutral"]:
        tags = get_alternative_tags(emotion)
        print(f"{emotion}: {tags}")

    print("\n" + "=" * 80)
    print("✅ Test completati!")
    print("=" * 80)
