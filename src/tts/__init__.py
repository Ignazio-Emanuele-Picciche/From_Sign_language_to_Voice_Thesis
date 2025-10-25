"""
Modulo TTS (Text-to-Speech) Emotivo
Genera audio con modulazione prosodica basata su emozioni predette
"""

from .emotion_mapper import map_emotion_to_prosody
from .text_templates import get_tts_text
from .tts_generator import generate_emotional_audio, generate_baseline_audio

__all__ = [
    "map_emotion_to_prosody",
    "get_tts_text",
    "generate_emotional_audio",
    "generate_baseline_audio",
]
