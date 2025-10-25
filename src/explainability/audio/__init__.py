"""
Modulo Audio Explainability
Analizza e valida l'audio generato dal TTS emotivo
"""

from .acoustic_analyzer import AcousticAnalyzer
from .prosody_validator import validate_prosody, validate_prosody_batch

__all__ = ["AcousticAnalyzer", "validate_prosody", "validate_prosody_batch"]
