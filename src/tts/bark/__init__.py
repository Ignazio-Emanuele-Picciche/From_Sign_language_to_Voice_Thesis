"""
Modulo TTS Bark (Text-to-Speech) Emotivo
Genera audio con modulazione emotiva usando Bark di Suno AI

Componenti:
- emotion_mapper: Mappatura emozioni → speaker prompts Bark + tag emotivi multipli
- emotion_tag_optimizer: Posizionamento intelligente tag emotivi nel testo
- tts_generator: Generazione audio con Bark
- pytorch_patch: Fix compatibilità PyTorch 2.9+
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
