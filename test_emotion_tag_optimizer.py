"""
Test Emotion Tag Optimizer - Verifica il posizionamento intelligente dei tag emotivi
"""

import sys
from pathlib import Path

# Aggiungi src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.tts.bark.emotion_tag_optimizer import (
    optimize_emotional_text,
    find_natural_breaks,
    get_alternative_tags,
    optimize_positive_tags,
    optimize_negative_tags,
    optimize_neutral_tags,
)

print("=" * 80)
print("TEST EMOTION TAG OPTIMIZER")
print("=" * 80)

# Test 1: Testi di diversa lunghezza
print("\n📝 TEST 1: Ottimizzazione per lunghezza testo")
print("-" * 80)

test_cases = [
    ("Hello world", "Testo corto"),
    (
        "This is a medium length sentence, with some punctuation and pauses.",
        "Testo medio",
    ),
    (
        "This is a much longer sentence that contains multiple clauses, "
        "and it has several natural breaks where emotional tags could be inserted. "
        "The optimizer should find the best positions for maximum expressiveness.",
        "Testo lungo",
    ),
]

for text, description in test_cases:
    print(f"\n{description} ({len(text)} char): {text[:50]}...")
    print(f"  Positive: {optimize_emotional_text(text, 'Positive')[:100]}...")
    print(f"  Negative: {optimize_emotional_text(text, 'Negative')[:100]}...")
    print(f"  Neutral:  {optimize_emotional_text(text, 'Neutral')[:100]}...")

# Test 2: Rilevamento pause naturali
print("\n\n🔍 TEST 2: Rilevamento pause naturali")
print("-" * 80)

sentences_with_breaks = [
    "Hello, world! How are you?",
    "This is good, and that is better, but this one is the best.",
    "I went to the store because I needed milk, but they were closed.",
    "The analysis shows positive results. However, more testing is needed.",
]

for sent in sentences_with_breaks:
    breaks = find_natural_breaks(sent)
    print(f"\nText: {sent}")
    print(f"Breaks: {len(breaks)} trovate alle posizioni {breaks}")
    if breaks:
        for i, b in enumerate(breaks[:3]):  # Mostra max 3
            before = sent[max(0, b - 15) : b]
            after = sent[b : min(len(sent), b + 15)]
            print(f"  Break {i+1}: ...{before}|{after}...")

# Test 3: Confronto strategia Positive vs Negative
print("\n\n⚡ TEST 3: Confronto strategie Positive vs Negative")
print("-" * 80)

comparison_text = (
    "The sign language video shows clear emotional expression. "
    "The analysis indicates strong sentiment with high confidence."
)

print(f"\nTesto originale:\n{comparison_text}\n")
print(f"Strategia POSITIVE:")
print(f"{optimize_positive_tags(comparison_text)}\n")
print(f"Strategia NEGATIVE:")
print(f"{optimize_negative_tags(comparison_text)}\n")
print(f"Strategia NEUTRAL:")
print(f"{optimize_neutral_tags(comparison_text)}\n")

# Test 4: Tag alternativi
print("\n🎭 TEST 4: Tag emotivi alternativi")
print("-" * 80)

for emotion in ["Positive", "Negative", "Neutral"]:
    alts = get_alternative_tags(emotion)
    print(f"{emotion}: {alts}")

# Test 5: Esempi realistici con caption
print("\n\n📹 TEST 5: Caption realistiche (sign language)")
print("-" * 80)

realistic_captions = [
    ("Hello", "Positive"),
    ("I am happy to see you today", "Positive"),
    ("The weather is beautiful and sunny", "Positive"),
    ("Goodbye", "Negative"),
    ("I feel sad about this situation", "Negative"),
    ("This is disappointing news for everyone", "Negative"),
    ("The meeting starts at three", "Neutral"),
]

for caption, emotion in realistic_captions:
    optimized = optimize_emotional_text(caption, emotion, use_tags=True)
    print(f"\n{emotion:8s} | {caption:40s}")
    print(f"         → {optimized}")

# Test 6: Disabilitazione tag
print("\n\n🚫 TEST 6: Disabilitazione tag emotivi")
print("-" * 80)

test_text = "This is a test sentence with emotional content."
print(f"Testo originale: {test_text}")
print(
    f"Con tag:         {optimize_emotional_text(test_text, 'Positive', use_tags=True)}"
)
print(
    f"Senza tag:       {optimize_emotional_text(test_text, 'Positive', use_tags=False)}"
)

# Test 7: Tag personalizzati
print("\n\n🎨 TEST 7: Tag emotivi personalizzati")
print("-" * 80)

custom_text = "This is a test with a custom emotional tag."
print(f"Testo originale: {custom_text}")
print(
    f"Tag default:     {optimize_emotional_text(custom_text, 'Positive', use_tags=True)}"
)
print(
    f"Tag custom:      {optimize_emotional_text(custom_text, 'Positive', use_tags=True, custom_tag='[giggles]')}"
)

print("\n" + "=" * 80)
print("✅ Test completati!")
print("=" * 80)

# Summary delle strategie
print("\n📊 SUMMARY STRATEGIE OTTIMIZZAZIONE:")
print("-" * 80)
print("\nPOSITIVE ([laughs]):")
print("  • Testo corto (<40):  Tag all'inizio (spontaneo)")
print("  • Testo medio (40-100): Tag dopo prima pausa naturale")
print("  • Testo lungo (>100): Tag all'inizio + a metà frase")
print("\nNEGATIVE ([sighs]):")
print("  • Testo corto (<40):  Tag all'inizio")
print("  • Testo medio (40-100): Tag a metà (più drammatico)")
print("  • Testo lungo (>100): Tag all'inizio + verso la fine (75%)")
print("\nNEUTRAL ([clears throat]):")
print("  • Testo corto/medio: Nessun tag (mantieni neutralità)")
print("  • Testo lungo (>80): Tag solo all'inizio")
print("\nPAUSE NATURALI rilevate dopo:")
print("  • Punteggiatura: . , ! ? ;")
print("  • Congiunzioni: and, but, so, because, however, therefore")
print("  • Metà frase (se >100 char senza punteggiatura)")
