"""
Test comparativo: Tag posizionati ALL'INIZIO vs OTTIMIZZATI

Genera 2 versioni di ogni audio per confrontare la qualità:
1. Metodo tradizionale: tag sempre all'inizio
2. Metodo ottimizzato: tag posizionati strategicamente
"""

import sys
from pathlib import Path
import os

# Aggiungi src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.tts.bark import pytorch_patch  # Applica patch PyTorch
from src.tts.bark.tts_generator import generate_emotional_audio, preload_bark_models

print("=" * 80)
print("TEST COMPARATIVO: Posizionamento Tag Ottimizzato vs Tradizionale")
print("=" * 80)

# Test cases con testi di diversa lunghezza
test_cases = [
    {
        "emotion": "Positive",
        "caption": "Hello",
        "description": "Testo molto corto",
    },
    {
        "emotion": "Positive",
        "caption": "I am so happy to see you today, and I hope you're doing well!",
        "description": "Testo medio con pause",
    },
    {
        "emotion": "Negative",
        "caption": "I feel sad",
        "description": "Testo corto negativo",
    },
    {
        "emotion": "Negative",
        "caption": "This is very disappointing news for everyone involved, and we must address it carefully.",
        "description": "Testo lungo negativo",
    },
    {
        "emotion": "Neutral",
        "caption": "The meeting starts at three",
        "description": "Testo neutro breve",
    },
]

# Directory output
output_dir_traditional = "outputs/bark_comparison/traditional"
output_dir_optimized = "outputs/bark_comparison/optimized"

os.makedirs(output_dir_traditional, exist_ok=True)
os.makedirs(output_dir_optimized, exist_ok=True)

# Pre-carica modelli (velocizza generazione)
print("\n🔄 Pre-caricamento modelli Bark...")
try:
    preload_bark_models()
    print("✅ Modelli caricati!\n")
except Exception as e:
    print(f"⚠️  Errore pre-caricamento: {e}")
    print("   Continuo senza pre-caricamento (sarà più lento)\n")

# Genera audio per ogni test case
for i, case in enumerate(test_cases, 1):
    emotion = case["emotion"]
    caption = case["caption"]
    description = case["description"]

    print(f"\n{'=' * 80}")
    print(f"TEST {i}/{len(test_cases)}: {description}")
    print(f"Emozione: {emotion}")
    print(f"Caption: {caption}")
    print("-" * 80)

    try:
        # Versione 1: Tag tradizionale (sempre all'inizio)
        print(f"\n1️⃣  Generazione TRADIZIONALE (tag all'inizio)...")
        traditional_path = generate_emotional_audio(
            emotion=emotion,
            confidence=0.9,
            video_name=f"test_{i:02d}_traditional",
            output_dir=output_dir_traditional,
            caption=caption,
            use_emotional_tags=True,
            optimize_tag_placement=False,  # ❌ NO ottimizzazione
            preload=True,
        )
        print(f"   ✅ Salvato: {traditional_path}")

        # Versione 2: Tag ottimizzato (posizionato strategicamente)
        print(f"\n2️⃣  Generazione OTTIMIZZATA (posizionamento intelligente)...")
        optimized_path = generate_emotional_audio(
            emotion=emotion,
            confidence=0.9,
            video_name=f"test_{i:02d}_optimized",
            output_dir=output_dir_optimized,
            caption=caption,
            use_emotional_tags=True,
            optimize_tag_placement=True,  # ✅ SI ottimizzazione
            preload=True,
        )
        print(f"   ✅ Salvato: {optimized_path}")

    except Exception as e:
        print(f"\n❌ Errore durante generazione: {e}")
        import traceback

        traceback.print_exc()
        continue

print("\n" + "=" * 80)
print("✅ TEST COMPLETATO!")
print("=" * 80)

print("\n📁 Audio generati in:")
print(f"   Tradizionale: {output_dir_traditional}/")
print(f"   Ottimizzato:  {output_dir_optimized}/")

print("\n🎧 COME CONFRONTARE:")
print("-" * 80)
print("1. Apri i file audio in un player (QuickTime, VLC, etc.)")
print("2. Per ogni coppia (traditional vs optimized):")
print("   - Ascolta la versione TRADIZIONALE (tag sempre all'inizio)")
print("   - Ascolta la versione OTTIMIZZATA (tag posizionato strategicamente)")
print("3. Nota le differenze in:")
print("   ✓ Naturalezza dell'audio")
print("   ✓ Posizionamento delle risate/sospiri")
print("   ✓ Espressività emotiva complessiva")

print("\n📊 ASPETTATIVE:")
print("-" * 80)
print("TESTI CORTI:")
print("  → Poca differenza (entrambi mettono tag all'inizio)")
print("\nTESTI MEDI:")
print("  → Ottimizzato dovrebbe suonare più naturale")
print("  → Tag dopo pause naturali invece che sempre all'inizio")
print("\nTESTI LUNGHI:")
print("  → Ottimizzato dovrebbe avere tag distribuiti meglio")
print("  → Positive: risate all'inizio + a metà")
print("  → Negative: sospiri all'inizio + verso la fine")

print("\n" + "=" * 80)
