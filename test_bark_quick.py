#!/usr/bin/env python3
"""
Test VELOCE per Bark - senza pre-caricamento modelli
Genera solo un audio di test per verificare che funzioni
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Applica patch PyTorch
from src.tts.bark import pytorch_patch

from src.tts.bark.tts_generator import generate_emotional_audio
from src.tts.bark.emotion_mapper import map_emotion_to_bark_prompt

print("=" * 70)
print("TEST VELOCE BARK TTS")
print("=" * 70)

# Output directory
output_dir = os.path.join(BASE_DIR, "test_output_bark")
os.makedirs(output_dir, exist_ok=True)

# Test: Genera solo un audio Positive
print("\n1. Test mapping emozione Positive:")
config = map_emotion_to_bark_prompt("Positive")
print(f"  Speaker: {config['history_prompt']}")
print(f"  Tag emotivo: {config['text_prefix']}")
print(f"  Temperature: {config['temperature']}")

print("\n2. Generazione audio Positive (senza pre-caricamento)...")
print("   NOTA: Questo può richiedere 1-2 minuti la prima volta")
print("   I modelli Bark verranno scaricati automaticamente (~5GB)")

try:
    audio_path = generate_emotional_audio(
        emotion="Positive",
        confidence=0.95,
        video_name="test_quick",
        output_dir=output_dir,
        caption="This is a quick test of Bark TTS!",
        use_emotional_tags=True,
        preload=False,  # NON pre-caricare per velocità
    )

    print(f"\n✅ SUCCESS! Audio generato: {audio_path}")

    if os.path.exists(audio_path):
        size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        print(f"   Dimensione: {size_mb:.2f} MB")
        print(f"\n🎵 Ascolta il file per verificare la qualità!")

except Exception as e:
    print(f"\n❌ ERRORE: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 70)
