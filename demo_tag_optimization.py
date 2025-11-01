"""
Demo Interattiva: Mostra come il testo viene ottimizzato con i tag emotivi
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.tts.bark.emotion_tag_optimizer import (
    optimize_emotional_text,
    find_natural_breaks,
)


def visualize_optimization(text: str, emotion: str):
    """Mostra visivamente come il testo viene ottimizzato"""

    print(f"\n{'=' * 80}")
    print(f"EMOZIONE: {emotion}")
    print(f"{'=' * 80}")

    # Testo originale
    print(f"\n📝 TESTO ORIGINALE ({len(text)} caratteri):")
    print(f"   {text}")

    # Trova pause naturali
    breaks = find_natural_breaks(text)
    if breaks:
        print(f"\n🔍 PAUSE NATURALI rilevate ({len(breaks)}):")
        for i, pos in enumerate(breaks[:5], 1):
            before = text[max(0, pos - 15) : pos].strip()
            after = text[pos : min(len(text), pos + 15)].strip()
            print(f"   {i}. Posizione {pos:3d}: ...{before} | {after}...")

    # Versione tradizionale (tag all'inizio)
    from src.tts.bark.emotion_mapper import EMOTION_BARK_MAPPING

    tag = EMOTION_BARK_MAPPING.get(emotion, {}).get("text_prefix", "")
    traditional = f"{tag} {text}" if tag else text

    print(f"\n❌ TRADIZIONALE (tag all'inizio):")
    print(f"   {traditional}")

    # Versione ottimizzata
    optimized = optimize_emotional_text(text, emotion, use_tags=True)

    print(f"\n✅ OTTIMIZZATO (posizionamento intelligente):")
    print(f"   {optimized}")

    # Evidenzia differenze
    if traditional != optimized:
        print(f"\n💡 DIFFERENZA:")
        print(f"   Tag spostato in posizione più naturale!")
    else:
        print(f"\n💡 NOTA:")
        print(f"   Stesso risultato (testo corto o strategia simile)")


# Test cases rappresentativi
test_cases = [
    # Testi corti
    ("Hello world", "Positive"),
    ("I feel sad", "Negative"),
    # Testi medi con pause
    (
        "I am so happy to see you today, and I hope you're doing well!",
        "Positive",
    ),
    (
        "This is very disappointing news, and we must address it carefully.",
        "Negative",
    ),
    # Testi lunghi
    (
        "The sign language video shows clear emotional expression. "
        "The analysis indicates strong positive sentiment with high confidence, "
        "and the results are very encouraging for our research.",
        "Positive",
    ),
    (
        "Unfortunately, the results are not what we expected. "
        "Despite our best efforts, we were unable to achieve the desired outcome. "
        "This is a setback that requires careful consideration.",
        "Negative",
    ),
    # Neutrale
    ("The meeting starts at three", "Neutral"),
    (
        "The meeting starts at three and will cover project updates, "
        "budget review, quarterly results, and next steps for the team.",
        "Neutral",
    ),
]

print("=" * 80)
print("🎭 DEMO: OTTIMIZZAZIONE POSIZIONAMENTO TAG EMOTIVI")
print("=" * 80)
print("\nQuesta demo mostra come i tag emotivi ([laughs], [sighs]) vengono")
print("posizionati in modo INTELLIGENTE invece che sempre all'inizio.\n")

for i, (text, emotion) in enumerate(test_cases, 1):
    if i > 1:
        input("\n👉 Premi INVIO per vedere il prossimo esempio...")
    visualize_optimization(text, emotion)

print("\n" + "=" * 80)
print("✅ DEMO COMPLETATA!")
print("=" * 80)

print("\n📊 RIASSUNTO:")
print("-" * 80)
print("POSITIVE:")
print("  • Testo corto:  Tag all'inizio")
print("  • Testo medio:  Tag dopo prima pausa (più spontaneo)")
print("  • Testo lungo:  Tag all'inizio + a metà")
print("\nNEGATIVE:")
print("  • Testo corto:  Tag all'inizio")
print("  • Testo medio:  Tag a metà (più drammatico)")
print("  • Testo lungo:  Tag all'inizio + verso la fine (75%)")
print("\nNEUTRAL:")
print("  • Testo breve:  Nessun tag")
print("  • Testo lungo:  Tag solo se >80 caratteri")

print("\n🎧 Per ascoltare la differenza, esegui:")
print("   python test_bark_optimization_comparison.py")
