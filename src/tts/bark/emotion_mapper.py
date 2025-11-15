"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                  EMOTION MAPPER - CONFIGURAZIONE BARK TTS                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

📋 DESCRIZIONE:
    Modulo di mappatura intelligente tra emozioni predette dal modello ViViT
    e configurazioni Bark TTS (speaker prompts, tag emotivi, temperature).
    Agisce come traduttore tra lo spazio delle emozioni e lo spazio dei
    parametri di Bark.

🎯 SCOPO PRINCIPALE:
    Convertire una predizione emotiva (es: "Positive" con confidence 0.92)
    in una configurazione Bark completa che include:
    - Speaker voice prompt (quale voce usare)
    - Tag emotivi da inserire nel testo ([laughs], [sighs], etc.)
    - Temperature per controllare la variabilità della generazione
    - Descrizioni leggibili per logging/debug

🏗️ COMPONENTI CHIAVE:

    1. EMOTION_BARK_MAPPING (dict)
       └─> Mapping principale: emozione → configurazione Bark di default
           Struttura: {
               "Positive": {
                   "history_prompt": "v2/en_speaker_6",  # Voce energica
                   "text_prefix": "[laughs]",             # Tag di default
                   "temperature": 0.7,                    # Variabilità
                   "description": "Voce allegra..."
               },
               ...
           }

    2. EMOTIONAL_TAGS (dict)
       └─> Sistema multi-tag con selezione basata su confidenza
           - primary: Tag di default per l'emozione
           - alternatives: Tag alternativi per varietà ([chuckles], [giggles])
           - high/medium/low_confidence: Tag scelti in base a confidence score

           INNOVAZIONE: Adapta l'intensità del tag alla certezza della predizione
           - Confidence >90%: tag forte ([laughs])
           - Confidence 70-90%: tag moderato ([chuckles])
           - Confidence <70%: nessun tag (troppo incerto)

    3. ALTERNATIVE_SPEAKERS (dict)
       └─> Pool di speaker alternativi per ogni emozione
           Bark ha 10 speaker per lingua (v2/en_speaker_0 a _9)
           Questo permette varietà vocale mantenendo coerenza emotiva

📊 MAPPING DETTAGLIATO:

    POSITIVE:
    - Speakers: 6 (energico), 5 (allegro), 7 (vivace)
    - Tags: [laughs] (genuino), [chuckles] (contenuto), [giggles] (leggero)
    - Temperature: 0.7 (alta espressività)
    - Uso: Gioia, eccitazione, soddisfazione

    NEGATIVE:
    - Speakers: 3 (calmo), 1 (riflessivo), 4 (serio)
    - Tags: [sighs] (sospiro), [gasps] (shock), [sad] (tristezza), [clears throat]
    - Temperature: 0.6 (più controllato)
    - Uso: Tristezza, frustrazione, delusione

    NEUTRAL:
    - Speakers: 9 (professionale), 0 (narratore), 2 (standard)
    - Tags: nessuno o [clears throat] (minimalista)
    - Temperature: 0.5 (bassa variabilità)
    - Uso: Informazioni fattuali, neutralità

🔧 FUNZIONI PRINCIPALI:

    map_emotion_to_bark_prompt(emotion, use_emotional_tags)
    └─> Restituisce configurazione Bark completa per un'emozione

    get_emotional_tag(emotion, confidence, alternative)
    └─> Seleziona tag emotivo ottimale basandosi su confidenza
        INNOVATIVO: Adapta l'intensità emotiva alla certezza

    get_alternative_emotional_tags(emotion)
    └─> Lista tutti i tag disponibili per un'emozione (per sperimentazione)

    get_bark_speaker(emotion, alternative)
    └─> Restituisce speaker prompt con possibilità di varianti

💡 STRATEGIA DI SELEZIONE TAG:

    La selezione del tag emotivo segue una logica a cascata:

    1. Se specificato alternative index → usa tag dalla lista alternatives
    2. Se specificata confidence → usa tag basato su soglie:
       - confidence ≥ 0.9 → high_confidence tag (più forte)
       - 0.7 ≤ confidence < 0.9 → medium_confidence tag
       - confidence < 0.7 → low_confidence tag (più neutro/assente)
    3. Altrimenti → usa primary tag (default)

    Questo permette:
    - Evitare tag inappropriati quando predizione è incerta
    - Variare l'espressività in base alla certezza del modello
    - Sperimentare con tag alternativi manualmente

🎨 PERSONALIZZAZIONE:

    Il sistema è facilmente estensibile:
    - Aggiungi nuove emozioni in EMOTION_BARK_MAPPING
    - Sperimenta con speaker diversi (Bark ha 10 per lingua)
    - Crea nuove combinazioni di tag emotivi
    - Regola temperature per controllare espressività

📈 VALIDAZIONE:

    Il modulo include un blocco __main__ per testing:
    - Verifica mapping per tutte le emozioni
    - Testa selezione tag basata su confidenza
    - Mostra speaker alternativi disponibili
    - Esporta tutti i tag per debugging

    Esegui: python emotion_mapper.py

🔗 INTEGRAZIONE:

    Usato da:
    - tts_generator.py: per ottenere configurazione prima della generazione
    - Script di testing: per verificare mapping e sperimentare
    - Pipeline principale: come primo step dopo predizione emozione

📚 RIFERIMENTI:
    - Bark tag emotivi supportati: docs/BARK_EMOTIONAL_TAGS.md
    - Speaker caratteristiche: README.md sezione "Speaker Mapping"
    - Temperature tuning: docs/HYPERPARAMETER_TUNING_GUIDE.md

⚠️ NOTE IMPORTANTI:
    - Gli speaker Bark hanno caratteristiche intrinseche (genere, età, stile)
      che influenzano la percezione emotiva
    - Non tutti i tag funzionano bene con tutti gli speaker
    - La temperature influenza sia prosodia che qualità audio
    - Tag multipli nello stesso testo possono creare effetti interessanti
      ma vanno posizionati con cura (vedi emotion_tag_optimizer.py)

👤 AUTORE: Ignazio Emanuele Picciche
📅 DATA: Novembre 2025
🎓 PROGETTO: Tesi Magistrale - EmoSign con Bark TTS
"""

# Mapping da emozione a speaker prompts di Bark
# Bark usa "history prompts" predefiniti che determinano lo stile vocale
# Formato: "v2/{language}_speaker_{number}" dove ogni speaker ha caratteristiche emotive diverse
EMOTION_BARK_MAPPING = {
    "Positive": {
        "history_prompt": "v2/en_speaker_6",  # Speaker energico e positivo
        "description": "Voce allegra, energica e positiva (upbeat female voice)",
        "text_prefix": "[laughs]",  # Tag di default
        "temperature": 0.7,  # Controllo della variabilità (0.1-1.0)
    },
    "Negative": {
        "history_prompt": "v2/en_speaker_3",  # Speaker più calmo e riflessivo
        "description": "Voce calma, riflessiva e contenuta (calm male voice)",
        "text_prefix": "[sighs]",  # Tag di default
        "temperature": 0.6,
    },
    "Neutral": {
        "history_prompt": "v2/en_speaker_9",  # Speaker neutro
        "description": "Voce neutra e professionale (neutral narrator)",
        "text_prefix": "",  # Nessun tag di default
        "temperature": 0.5,
    },
}

# 🆕 Tag emotivi alternativi per varietà e sfumature diverse
# Usati per variare l'espressività o basarsi sulla confidenza
EMOTIONAL_TAGS = {
    "Positive": {
        "primary": "[laughs]",  # Risata genuina (default)
        "alternatives": [
            "[chuckles]",  # Risata contenuta, più professionale
            "[giggles]",  # Risata leggera, giocosa
            "[laughter]",  # Risata (variante)
        ],
        "high_confidence": "[laughs]",  # >90% confidence
        "medium_confidence": "[chuckles]",  # 70-90% confidence
        "low_confidence": "",  # <70% confidence (nessun tag)
    },
    "Negative": {
        "primary": "[sighs]",  # Sospiro (default)
        "alternatives": [
            "[gasps]",  # Rantolo/shock negativo
            "[sad]",  # Voce triste
            "[clears throat]",  # Disagio/esitazione
        ],
        "high_confidence": "[sighs]",  # >90% confidence
        "medium_confidence": "[sighs]",  # 70-90% confidence
        "low_confidence": "[clears throat]",  # <70% (più neutro)
    },
    "Neutral": {
        "primary": "",  # Nessun tag (default)
        "alternatives": [
            "[clears throat]",  # Solo se necessario
            "[breath]",  # Respiro neutro
        ],
        "high_confidence": "",  # Sempre neutro
        "medium_confidence": "",
        "low_confidence": "[clears throat]",  # Solo se molto lungo
    },
}

# Alternative speaker prompts per maggiore varietà
# Bark ha 10 speaker per lingua (0-9), ognuno con caratteristiche uniche
ALTERNATIVE_SPEAKERS = {
    "Positive": [
        "v2/en_speaker_6",  # Energico
        "v2/en_speaker_5",  # Allegro
        "v2/en_speaker_7",  # Vivace
    ],
    "Negative": [
        "v2/en_speaker_3",  # Calmo
        "v2/en_speaker_1",  # Riflessivo
        "v2/en_speaker_4",  # Serio
    ],
    "Neutral": [
        "v2/en_speaker_9",  # Neutro professionale
        "v2/en_speaker_0",  # Narratore
        "v2/en_speaker_2",  # Standard
    ],
}


def map_emotion_to_bark_prompt(emotion: str, use_emotional_tags: bool = True) -> dict:
    """
    Mappa un'emozione a un Bark speaker prompt

    Args:
        emotion (str): Emozione predetta ('Positive', 'Negative', 'Neutral')
        use_emotional_tags (bool): Se True, aggiunge tag emotivi ([laughs], [sighs], etc.)

    Returns:
        dict: Dizionario con configurazione Bark
              {'history_prompt': str, 'text_prefix': str, 'description': str, 'temperature': float}

    Example:
        >>> map_emotion_to_bark_prompt('Positive')
        {'history_prompt': 'v2/en_speaker_6', 'text_prefix': '[laughs]', ...}
    """
    if emotion not in EMOTION_BARK_MAPPING:
        raise ValueError(
            f"Emozione '{emotion}' non riconosciuta. Usa: {list(EMOTION_BARK_MAPPING.keys())}"
        )

    config = EMOTION_BARK_MAPPING[emotion].copy()

    # Se non vogliamo tag emotivi, rimuovi il prefisso
    if not use_emotional_tags:
        config["text_prefix"] = ""

    return config


def get_emotional_tag(
    emotion: str, confidence: float = None, alternative: int = 0
) -> str:
    """
    🆕 Ottiene il tag emotivo ottimale basato su emozione, confidenza e variante

    Args:
        emotion (str): Emozione predetta ('Positive', 'Negative', 'Neutral')
        confidence (float, optional): Confidenza della predizione (0.0-1.0 o 0-100)
                                       Se None, usa tag primary
        alternative (int): Indice tag alternativo (0 = primary, 1+ = alternatives)

    Returns:
        str: Tag emotivo da usare (es: '[laughs]', '[chuckles]', '')

    Example:
        >>> get_emotional_tag('Positive')  # Default
        '[laughs]'
        >>> get_emotional_tag('Positive', confidence=0.95)  # Alta confidenza
        '[laughs]'
        >>> get_emotional_tag('Positive', confidence=0.65)  # Bassa confidenza
        ''
        >>> get_emotional_tag('Positive', alternative=1)  # Tag alternativo
        '[chuckles]'
    """
    if emotion not in EMOTIONAL_TAGS:
        return ""

    tags_config = EMOTIONAL_TAGS[emotion]

    # Se specificato alternative index, usa quello
    if alternative > 0:
        alternatives = tags_config.get("alternatives", [])
        # alternative=1 → index 0, alternative=2 → index 1, etc.
        idx = alternative - 1
        if 0 <= idx < len(alternatives):
            return alternatives[idx]
        # Fallback a primary se index fuori range
        return tags_config.get("primary", "")

    # Se specificata confidence, usa tag basato su livello di confidenza
    if confidence is not None:
        # Normalizza confidence a 0-1 range
        conf = confidence if confidence <= 1.0 else confidence / 100.0

        if conf >= 0.9:
            return tags_config.get("high_confidence", tags_config.get("primary", ""))
        elif conf >= 0.7:
            return tags_config.get("medium_confidence", tags_config.get("primary", ""))
        else:
            return tags_config.get("low_confidence", "")

    # Default: usa primary tag
    return tags_config.get("primary", "")


def get_alternative_emotional_tags(emotion: str) -> list:
    """
    🆕 Ottiene tutti i tag emotivi disponibili per un'emozione

    Args:
        emotion (str): Emozione predetta

    Returns:
        list: Lista di tutti i tag disponibili (primary + alternatives)

    Example:
        >>> get_alternative_emotional_tags('Positive')
        ['[laughs]', '[chuckles]', '[giggles]', '[laughter]']
    """
    if emotion not in EMOTIONAL_TAGS:
        return []

    tags_config = EMOTIONAL_TAGS[emotion]
    primary = tags_config.get("primary", "")
    alternatives = tags_config.get("alternatives", [])

    # Combina primary + alternatives, rimuovi duplicati e stringhe vuote
    all_tags = [primary] + alternatives
    return [tag for tag in all_tags if tag]


def get_bark_speaker(emotion: str, alternative: int = 0) -> str:
    """
    Ottiene il nome del speaker Bark per un'emozione

    Args:
        emotion (str): Emozione predetta
        alternative (int): Indice speaker alternativo (0 = default, 1-2 = alternative)

    Returns:
        str: Nome del speaker Bark (es: 'v2/en_speaker_6')

    Example:
        >>> get_bark_speaker('Positive')
        'v2/en_speaker_6'
        >>> get_bark_speaker('Positive', alternative=1)
        'v2/en_speaker_5'
    """
    if emotion not in EMOTION_BARK_MAPPING:
        raise ValueError(
            f"Emozione '{emotion}' non riconosciuta. Usa: {list(EMOTION_BARK_MAPPING.keys())}"
        )

    if alternative == 0:
        return EMOTION_BARK_MAPPING[emotion]["history_prompt"]
    else:
        alternatives = ALTERNATIVE_SPEAKERS.get(emotion, [])
        if 0 <= alternative < len(alternatives):
            return alternatives[alternative]
        else:
            # Fallback al default se alternative index è fuori range
            return EMOTION_BARK_MAPPING[emotion]["history_prompt"]


if __name__ == "__main__":
    # Test del modulo
    print("=" * 70)
    print("TEST EMOTION MAPPER - BARK TTS")
    print("=" * 70)

    emotions = ["Positive", "Negative", "Neutral"]

    print("\n1. Mapping emozioni a Bark prompts:")
    for emotion in emotions:
        print(f"\nEmozione: {emotion}")
        config = map_emotion_to_bark_prompt(emotion)
        print(f"  Speaker: {config['history_prompt']}")
        print(f"  Descrizione: {config['description']}")
        print(f"  Tag emotivo: {config['text_prefix']}")
        print(f"  Temperature: {config['temperature']}")

    print("\n2. Speaker alternativi:")
    for emotion in emotions:
        print(f"\n{emotion}:")
        for i in range(3):
            speaker = get_bark_speaker(emotion, alternative=i)
            print(f"  Alternative {i}: {speaker}")

    print("\n🆕 3. Tag emotivi per confidenza:")
    for emotion in emotions:
        print(f"\n{emotion}:")
        for conf in [0.95, 0.80, 0.60]:
            tag = get_emotional_tag(emotion, confidence=conf)
            print(f"  Confidence {conf:.0%}: {tag if tag else '(nessun tag)'}")

    print("\n🆕 4. Tag emotivi alternativi:")
    for emotion in emotions:
        tags = get_alternative_emotional_tags(emotion)
        print(f"\n{emotion}: {tags}")

    print("\n🆕 5. Test varianti tag (Positive):")
    for i in range(4):
        tag = get_emotional_tag("Positive", alternative=i)
        print(f"  Variante {i}: {tag if tag else '(default)'}")

    print("\n✅ Test completato!")
