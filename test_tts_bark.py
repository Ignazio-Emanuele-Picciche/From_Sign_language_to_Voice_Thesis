#!/usr/bin/env python3
"""
Test rapido per verificare il TTS con Bark
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Applica patch PyTorch per compatibilità con Bark
from src.tts.bark import pytorch_patch

from src.tts.bark.tts_generator import (
    generate_emotional_audio,
    generate_baseline_audio,
    preload_bark_models,
)
from src.tts.bark.emotion_mapper import map_emotion_to_bark_prompt

print("=" * 70)
print("TEST TTS CON BARK")
print("=" * 70)

# Directory di output per test
output_dir = os.path.join(BASE_DIR, "test_output_bark")
os.makedirs(output_dir, exist_ok=True)

# Test 1: Verifica mapping emozione -> Bark prompt
print("\n1. Test mapping emozioni a Bark prompts:")
for emotion in ["Positive", "Negative", "Neutral"]:
    config = map_emotion_to_bark_prompt(emotion)
    print(f"  {emotion:8} -> Speaker: {config['history_prompt']}")
    print(
        f"             Tag: {config['text_prefix'] if config['text_prefix'] else '(none)'}, Temp: {config['temperature']}"
    )
    print(f"             {config['description']}")

# Test 2: Pre-carica modelli
print("\n2. Pre-caricamento modelli Bark...")
print("   (Questo può richiedere alcuni minuti la prima volta)")
try:
    preload_bark_models()
    print("   ✅ Modelli caricati!")
except Exception as e:
    print(f"   ❌ Errore: {e}")
    print("   Continuo senza pre-caricamento (generazione più lenta)")

# Test 3: Genera audio baseline
print("\n3. Generazione audio baseline (neutro)...")
baseline_path = os.path.join(output_dir, "baseline_neutral_bark.wav")
try:
    result = generate_baseline_audio(baseline_path, preload=True)
    print(f"  ✅ Baseline generato: {result}")
    size_mb = os.path.getsize(result) / (1024 * 1024)
    print(f"     Dimensione: {size_mb:.2f} MB")
except Exception as e:
    print(f"  ❌ Errore: {e}")
    import traceback

    traceback.print_exc()

# Test 4: Genera audio emotivo
print("\n4. Generazione audio emotivo con Bark...")

emotions_to_test = [
    ("Positive", 0.95, "This sign language conveys happiness and joy!"),
    ("Negative", 0.87, "This gesture expresses sadness and disappointment."),
]

for emotion, confidence, caption in emotions_to_test:
    print(f"\n  Generazione audio per '{emotion}' (confidence: {confidence:.0%})...")
    print(f"  Caption: {caption}")
    try:
        audio_path = generate_emotional_audio(
            emotion=emotion,
            confidence=confidence,
            video_name=f"test_{emotion.lower()}",
            output_dir=output_dir,
            caption=caption,
            use_emotional_tags=True,
            preload=True,
        )
        print(f"  ✅ Audio salvato: {os.path.basename(audio_path)}")

        # Verifica che il file esista
        if os.path.exists(audio_path):
            size_mb = os.path.getsize(audio_path) / (1024 * 1024)
            print(f"     Dimensione: {size_mb:.2f} MB")
        else:
            print(f"  ⚠️  File non trovato!")

    except Exception as e:
        print(f"  ❌ Errore: {e}")
        import traceback

        traceback.print_exc()

print("\n" + "=" * 70)
print("TEST COMPLETATO!")
print(f"File audio salvati in: {output_dir}")
print("Ascolta i file .wav per verificare la qualità di Bark")
print("\nNOTA: Bark genera audio molto espressivo con:")
print("  - Tag emotivi: [laughs], [sighs], [gasps]")
print("  - Speaker con caratteristiche emotive diverse")
print("  - Modulazione naturale di pitch, rate e volume")
print("=" * 70)
