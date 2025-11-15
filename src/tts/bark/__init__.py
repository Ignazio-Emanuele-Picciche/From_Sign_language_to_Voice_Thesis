"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   MODULO TTS BARK - TEXT-TO-SPEECH EMOTIVO                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

📋 DESCRIZIONE:
    Package principale per la generazione di audio emotivo da testo utilizzando
    Bark TTS (modello transformer di Suno AI). Questo modulo orchestra l'intera
    pipeline di conversione da emozione predetta a audio espressivo.

🎯 SCOPO:
    Generare audio WAV con prosodia emotiva naturale partendo da:
    - Predizioni di emozioni (Positive/Negative/Neutral)
    - Livelli di confidenza delle predizioni
    - Testo del linguaggio dei segni tradotto

🏗️ ARCHITETTURA:
    Questo package coordina 4 componenti specializzati:

    1. emotion_mapper.py
       └─> Mappa emozioni a configurazioni Bark (speaker + tag emotivi)

    2. emotion_tag_optimizer.py
       └─> Posiziona intelligentemente tag emotivi nel testo per massimizzare
           l'espressività ([laughs], [sighs], etc.)

    3. tts_generator.py
       └─> Core: genera effettivamente l'audio usando Bark TTS

    4. pytorch_patch.py
       └─> Fix di compatibilità per PyTorch 2.9+ (weights_only issue)

🔄 FLUSSO TIPICO:
    Video Sign Language → ViViT (emozione) → emotion_mapper (config Bark)
    → emotion_tag_optimizer (testo ottimizzato) → tts_generator (audio WAV)

📦 EXPORT:
    Questo __init__.py espone le funzioni principali per uso esterno:
    - Mapping emozioni: map_emotion_to_bark_prompt(), get_bark_speaker(), etc.
    - Ottimizzazione tag: optimize_emotional_text(), get_alternative_tags()
    - Generazione audio: generate_emotional_audio(), preload_bark_models()

💡 USO TIPICO:
    from src.tts.bark import generate_emotional_audio

    audio_path = generate_emotional_audio(
        emotion="Positive",
        confidence=0.92,
        video_name="video_001",
        output_dir="results/audio",
        use_emotional_tags=True
    )

📚 DIPENDENZE:
    - bark (Suno AI): Modello TTS transformer-based
    - torch: Backend per inferenza Bark
    - scipy: Salvataggio file WAV
    - numpy: Manipolazione array audio

🔗 RIFERIMENTI:
    - Bark GitHub: https://github.com/suno-ai/bark
    - Documentazione completa: src/tts/bark/README.md
    - Tag emotivi Bark: docs/BARK_EMOTIONAL_TAGS.md

👤 AUTORE: Ignazio Emanuele Picciche
📅 DATA: Novembre 2025
🎓 PROGETTO: Tesi Magistrale - EmoSign con Bark TTS
"""

from .emotion_mapper import (
    map_emotion_to_bark_prompt,
    get_bark_speaker,
    get_emotional_tag,
    get_alternative_emotional_tags,
)
from .emotion_tag_optimizer import optimize_emotional_text, get_alternative_tags
from .tts_generator import (
    generate_emotional_audio,
    generate_baseline_audio,
    preload_bark_models,
)

__all__ = [
    # Emotion mapping
    "map_emotion_to_bark_prompt",
    "get_bark_speaker",
    "get_emotional_tag",
    "get_alternative_emotional_tags",
    # Tag optimization
    "optimize_emotional_text",
    "get_alternative_tags",
    # Audio generation
    "generate_emotional_audio",
    "generate_baseline_audio",
    "preload_bark_models",
]
