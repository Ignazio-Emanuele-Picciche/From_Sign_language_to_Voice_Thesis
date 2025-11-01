"""
Test Audio con Tag Emotivi Multipli

Genera audio usando TAG DIVERSI per confrontare l'espressività:
- [laughs] vs [chuckles] vs [giggles] (Positive)
- [sighs] vs [gasps] vs [sad] (Negative)
"""

import sys
from pathlib import Path
import os

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.tts.bark import pytorch_patch  # Applica patch
from src.tts.bark.tts_generator import generate_emotional_audio, preload_bark_models
from src.tts.bark.emotion_mapper import get_alternative_emotional_tags

print("=" * 80)
print("🎭 TEST AUDIO CON TAG EMOTIVI MULTIPLI")
print("=" * 80)

# Pre-carica modelli
print("\n🔄 Pre-caricamento modelli Bark...")
try:
    preload_bark_models()
    print("✅ Modelli caricati!\n")
except Exception as e:
    print(f"⚠️  Errore: {e}")
    print("   Continuo senza pre-caricamento\n")

# Test cases
test_caption = "I am so happy to see you today!"

# Output directory
output_dir = "outputs/bark_tags_comparison"
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("TEST 1: POSITIVE - Confronto Tag Diversi")
print("=" * 80)

positive_tags = get_alternative_emotional_tags("Positive")
print(f"\nTag disponibili per Positive: {positive_tags}")
print(f"Testo: {test_caption}\n")

for i, tag in enumerate(positive_tags):
    print(f"\n{'─' * 80}")
    print(f"Generazione {i+1}/{len(positive_tags)}: {tag}")
    print(f"{'─' * 80}")

    try:
        audio_path = generate_emotional_audio(
            emotion="Positive",
            confidence=0.90,
            video_name=f"positive_tag_{i}",
            output_dir=output_dir,
            caption=test_caption,
            use_emotional_tags=True,
            alternative_tag=i,  # Usa tag specifico
            confidence_based_tags=False,  # Disabilita auto-selezione
            optimize_tag_placement=True,
            preload=True,
        )
        print(f"✅ Salvato: {audio_path}")
    except Exception as e:
        print(f"❌ Errore: {e}")

print("\n\n" + "=" * 80)
print("TEST 2: NEGATIVE - Confronto Tag Diversi")
print("=" * 80)

test_caption_negative = "This is very disappointing and sad news."

negative_tags = get_alternative_emotional_tags("Negative")
print(f"\nTag disponibili per Negative: {negative_tags}")
print(f"Testo: {test_caption_negative}\n")

for i, tag in enumerate(negative_tags):
    print(f"\n{'─' * 80}")
    print(f"Generazione {i+1}/{len(negative_tags)}: {tag}")
    print(f"{'─' * 80}")

    try:
        audio_path = generate_emotional_audio(
            emotion="Negative",
            confidence=0.90,
            video_name=f"negative_tag_{i}",
            output_dir=output_dir,
            caption=test_caption_negative,
            use_emotional_tags=True,
            alternative_tag=i,
            confidence_based_tags=False,
            optimize_tag_placement=True,
            preload=True,
        )
        print(f"✅ Salvato: {audio_path}")
    except Exception as e:
        print(f"❌ Errore: {e}")

print("\n\n" + "=" * 80)
print("TEST 3: CONFIDENCE-BASED - Tag Automatico per Confidenza")
print("=" * 80)

test_confidences = [
    (0.98, "Alta confidenza"),
    (0.82, "Media confidenza"),
    (0.62, "Bassa confidenza"),
]

test_caption_conf = "This is wonderful news!"

for conf, label in test_confidences:
    print(f"\n{'─' * 80}")
    print(f"{label} ({conf:.0%})")
    print(f"{'─' * 80}")

    try:
        audio_path = generate_emotional_audio(
            emotion="Positive",
            confidence=conf,
            video_name=f"positive_conf_{int(conf*100)}",
            output_dir=output_dir,
            caption=test_caption_conf,
            use_emotional_tags=True,
            confidence_based_tags=True,  # ✅ Abilita selezione automatica
            optimize_tag_placement=True,
            preload=True,
        )
        print(f"✅ Salvato: {audio_path}")
    except Exception as e:
        print(f"❌ Errore: {e}")

print("\n\n" + "=" * 80)
print("✅ TEST COMPLETATO!")
print("=" * 80)

print(f"\n📁 Audio generati in: {output_dir}/")

print("\n🎧 COME CONFRONTARE:")
print("-" * 80)
print("\n1. POSITIVE - Tag Diversi:")
for i, tag in enumerate(positive_tags):
    print(f"   {tag:15s} → positive_tag_{i}_positive.wav")

print("\n2. NEGATIVE - Tag Diversi:")
for i, tag in enumerate(negative_tags):
    print(f"   {tag:15s} → negative_tag_{i}_negative.wav")

print("\n3. CONFIDENCE - Tag Automatici:")
print("   98% conf → positive_conf_98_positive.wav  ([laughs])")
print("   82% conf → positive_conf_82_positive.wav  ([chuckles])")
print("   62% conf → positive_conf_62_positive.wav  (no tag)")

print("\n📊 ASPETTATIVE:")
print("-" * 80)
print("\nPOSITIVE:")
print("  [laughs]   - Risata forte, genuina")
print("  [chuckles] - Risata contenuta, professionale")
print("  [giggles]  - Risata leggera, giocosa")
print("  [laughter] - Risata (variante)")

print("\nNEGATIVE:")
print("  [sighs]         - Sospiro di tristezza/frustrazione")
print("  [gasps]         - Shock/sorpresa negativa")
print("  [sad]           - Voce triste")
print("  [clears throat] - Disagio/esitazione")

print("\n💡 QUALE TAG SCEGLIERE?")
print("-" * 80)
print("Dopo aver ascoltato gli audio, scegli il tag che:")
print("  ✓ Suona più naturale per il contesto")
print("  ✓ Ha miglior timing emotivo")
print("  ✓ Non è troppo eccessivo o sottotono")
