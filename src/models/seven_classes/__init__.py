"""
Seven Classes Emotion Recognition Models

This module contains model architectures and training scripts for 7-class emotion recognition:
- Extremely Negative
- Negative
- Somewhat Negative
- Neutral
- Somewhat Positive
- Positive
- Extremely Positive

Models:
- EmotionLSTM: LSTM-based sequence model
- STGCN: Spatial-Temporal Graph Convolutional Network
- ViViT: Video Vision Transformer (in vivit/ subdirectory)
"""

from .lstm_model import EmotionLSTM
from .stgcn_model import STGCN

__all__ = ["EmotionLSTM", "STGCN"]
