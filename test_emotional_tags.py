"""
Test Tag Emotivi Multipli - Confronto audio con tag diversi
Questo test genera audio usando tag emotivi DIVERSI per la stessa emozione
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

print("=" * 80)
print("TEST TAG EMOTIVI MULTIPLI - Bark TTS")
print("=" * 80)

# Test 1: Verifica tag disponibili
from src.tts.bark.emotion_mapper import (
    get_emotional_tag,
    get_alternative_emotional_tags,
    EMOTIONAL_TAGS,
)

print("\n📋 TAG EMOTIVI DISPONIBILI:")
print("-" * 80)

for emotion in ["Positive", "Negative", "Neutral"]:
    all_tags = get_alternative_emotional_tags(emotion)
    print(f"\n{emotion}:")
    print(f"  Tutti i tag: {all_tags}")

    # Mostra tag per confidenza
    print(f"  Per confidenza:")
    for conf, label in [
        (0.95, "Alta (>90%)"),
        (0.80, "Media (70-90%)"),
        (0.60, "Bassa (<70%)"),
    ]:
        tag = get_emotional_tag(emotion, confidence=conf)
        print(f"    {label:20s}: {tag if tag else '(nessun tag)'}")

# Test 2: Esempi tag alternativi
print("\n\n🎭 ESEMPI TAG ALTERNATIVI:")
print("-" * 80)

print("\nPositive (4 varianti):")
for i in range(4):
    tag = get_emotional_tag("Positive", alternative=i)
    print(f"  Alternative {i}: {tag if tag else '(default)'}")

print("\nNegative (4 varianti):")
for i in range(4):
    tag = get_emotional_tag("Negative", alternative=i)
    print(f"  Alternative {i}: {tag if tag else '(default)'}")

# Test 3: Strategia confidenza
print("\n\n📊 STRATEGIA BASATA SU CONFIDENZA:")
print("-" * 80)

test_confidences = [0.98, 0.85, 0.72, 0.55]

for emotion in ["Positive", "Negative"]:
    print(f"\n{emotion}:")
    for conf in test_confidences:
        tag = get_emotional_tag(emotion, confidence=conf)
        print(f"  Confidence {conf:.0%}: {tag:20s} {'(no tag)' if not tag else ''}")

# Test 4: Configurazione completa
print("\n\n🔧 CONFIGURAZIONE COMPLETA:")
print("-" * 80)

for emotion, config in EMOTIONAL_TAGS.items():
    print(f"\n{emotion}:")
    print(f"  Primary: {config.get('primary', '(none)')}")
    print(f"  Alternatives: {config.get('alternatives', [])}")
    print(f"  High conf:   {config.get('high_confidence', '(none)')}")
    print(f"  Medium conf: {config.get('medium_confidence', '(none)')}")
    print(f"  Low conf:    {config.get('low_confidence', '(none)')}")

print("\n" + "=" * 80)
print("✅ Test tag emotivi completato!")
print("=" * 80)

print("\n💡 COME USARE:")
print("-" * 80)
print("\n1. Tag basato su CONFIDENZA (automatico):")
print("   generate_emotional_audio(emotion='Positive', confidence=0.92)")
print("   → Sceglie automaticamente il tag migliore per 92% confidence")

print("\n2. Tag ALTERNATIVO specifico:")
print("   generate_emotional_audio(emotion='Positive', alternative_tag=1)")
print("   → Usa [chuckles] invece di [laughs]")

print("\n3. DISABILITA tag basati su confidenza:")
print("   generate_emotional_audio(emotion='Positive', confidence_based_tags=False)")
print("   → Usa sempre il tag primary")

print("\n🎧 Per generare AUDIO con tag diversi:")
print("   python test_bark_multiple_tags.py")
