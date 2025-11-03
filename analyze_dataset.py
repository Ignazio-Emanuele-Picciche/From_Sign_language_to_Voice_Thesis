"""
Analisi Dataset Sign Language
==============================

Questo script analizza il dataset di caption per capire:
- Vocabulary size e distribuzione parole
- Lunghezza caption (min/max/media)
- Differenze Positive vs Negative
- Statistiche utili per design tokenizer e modello

Usage:
    python analyze_dataset.py --csv_path data/processed/golden_label_sentiment.csv
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
import re


def clean_text(text):
    """Pulisce il testo per analisi."""
    # Lowercase
    text = text.lower()
    # Remove special chars, mantieni solo lettere e spazi
    text = re.sub(r"[^a-z\s]", "", text)
    # Multiple spaces -> single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_simple(text):
    """Tokenizzazione semplice per analisi."""
    return clean_text(text).split()


def analyze_captions(csv_path, output_dir="results/dataset_analysis"):
    """
    Analizza le caption del dataset.

    Args:
        csv_path: Path al CSV con annotazioni
        output_dir: Directory output per grafici e statistiche
    """
    # Crea output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("📊 ANALISI DATASET SIGN LANGUAGE")
    print("=" * 70)

    # 1. Carica dataset
    print(f"\n1️⃣  Caricamento dataset...")
    df = pd.read_csv(csv_path)
    print(f"   ✓ Caricati {len(df)} campioni")
    print(f"   ✓ Colonne: {df.columns.tolist()}")

    # 2. Info generali
    print(f"\n2️⃣  Statistiche generali:")
    print(f"   - Total videos: {len(df)}")
    print(f"   - Emotions: {df['emotion'].value_counts().to_dict()}")
    print(f"   - Missing captions: {df['caption'].isna().sum()}")

    # 3. Analisi lunghezza caption
    print(f"\n3️⃣  Analisi lunghezza caption:")

    # Conta parole per caption
    df["num_words"] = df["caption"].apply(lambda x: len(tokenize_simple(x)))
    df["num_chars"] = df["caption"].apply(len)

    print(f"\n   📏 Lunghezza in PAROLE:")
    print(f"      - Min:    {df['num_words'].min()}")
    print(f"      - Max:    {df['num_words'].max()}")
    print(f"      - Mean:   {df['num_words'].mean():.2f}")
    print(f"      - Median: {df['num_words'].median():.0f}")
    print(f"      - Std:    {df['num_words'].std():.2f}")

    print(f"\n   📏 Lunghezza in CARATTERI:")
    print(f"      - Min:    {df['num_chars'].min()}")
    print(f"      - Max:    {df['num_chars'].max()}")
    print(f"      - Mean:   {df['num_chars'].mean():.2f}")
    print(f"      - Median: {df['num_chars'].median():.0f}")

    # 4. Vocabulary analysis
    print(f"\n4️⃣  Analisi vocabulary:")

    all_words = []
    for caption in df["caption"]:
        all_words.extend(tokenize_simple(caption))

    word_counts = Counter(all_words)

    print(f"   📚 Vocabulary:")
    print(f"      - Unique words: {len(word_counts)}")
    print(f"      - Total words:  {len(all_words)}")
    print(f"      - Avg frequency: {len(all_words) / len(word_counts):.2f}")

    print(f"\n   🔝 Top 20 parole più frequenti:")
    for word, count in word_counts.most_common(20):
        print(f"      - '{word}': {count} ({count/len(all_words)*100:.1f}%)")

    # 5. Analisi per emozione
    print(f"\n5️⃣  Analisi per EMOZIONE:")

    for emotion in df["emotion"].unique():
        emotion_df = df[df["emotion"] == emotion]
        emotion_words = []
        for caption in emotion_df["caption"]:
            emotion_words.extend(tokenize_simple(caption))

        print(f"\n   {'🙂' if emotion == 'Positive' else '😞'} {emotion}:")
        print(f"      - Samples: {len(emotion_df)}")
        print(f"      - Avg words/caption: {emotion_df['num_words'].mean():.2f}")
        print(f"      - Unique words: {len(set(emotion_words))}")
        print(f"      - Total words: {len(emotion_words)}")

    # 6. Distribuzione lunghezze
    print(f"\n6️⃣  Distribuzione lunghezze:")

    # Bins per lunghezza
    bins = [0, 5, 10, 15, 20, 25, 100]
    labels = ["0-5", "6-10", "11-15", "16-20", "21-25", "25+"]
    df["length_bin"] = pd.cut(df["num_words"], bins=bins, labels=labels)

    print(f"\n   📊 Caption per lunghezza:")
    for label in labels:
        count = (df["length_bin"] == label).sum()
        print(f"      - {label} words: {count} ({count/len(df)*100:.1f}%)")

    # 7. Save statistiche
    print(f"\n7️⃣  Salvataggio risultati...")

    # Summary stats
    stats = {
        "total_samples": int(len(df)),
        "total_words": int(len(all_words)),
        "unique_words": int(len(word_counts)),
        "vocab_size": int(len(word_counts)),
        "avg_words_per_caption": float(df["num_words"].mean()),
        "median_words_per_caption": float(df["num_words"].median()),
        "min_words": int(df["num_words"].min()),
        "max_words": int(df["num_words"].max()),
        "std_words": float(df["num_words"].std()),
    }

    # Save to JSON
    import json

    with open(output_dir / "statistics.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"   ✓ Statistiche salvate in: {output_dir / 'statistics.json'}")

    # Save word frequencies
    with open(output_dir / "word_frequencies.txt", "w") as f:
        for word, count in word_counts.most_common():
            f.write(f"{word}\t{count}\n")
    print(f"   ✓ Frequenze parole salvate in: {output_dir / 'word_frequencies.txt'}")

    # Save enhanced dataframe
    df.to_csv(output_dir / "dataset_with_stats.csv", index=False)
    print(
        f"   ✓ Dataset con statistiche salvato in: {output_dir / 'dataset_with_stats.csv'}"
    )

    # 8. Visualizzazioni
    print(f"\n8️⃣  Creazione visualizzazioni...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Distribuzione lunghezza parole
    axes[0, 0].hist(df["num_words"], bins=30, edgecolor="black", alpha=0.7)
    axes[0, 0].set_xlabel("Numero Parole")
    axes[0, 0].set_ylabel("Frequenza")
    axes[0, 0].set_title("Distribuzione Lunghezza Caption (parole)")
    mean_words = df["num_words"].mean()
    axes[0, 0].axvline(
        mean_words, color="red", linestyle="--", label=f"Media: {mean_words:.1f}"
    )
    axes[0, 0].legend()

    # Plot 2: Boxplot per emozione
    df.boxplot(column="num_words", by="emotion", ax=axes[0, 1])
    axes[0, 1].set_xlabel("Emozione")
    axes[0, 1].set_ylabel("Numero Parole")
    axes[0, 1].set_title("Lunghezza Caption per Emozione")
    plt.sca(axes[0, 1])
    plt.xticks(rotation=0)

    # Plot 3: Top 20 parole
    top_words = word_counts.most_common(20)
    words, counts = zip(*top_words)
    axes[1, 0].barh(range(len(words)), counts, alpha=0.7)
    axes[1, 0].set_yticks(range(len(words)))
    axes[1, 0].set_yticklabels(words)
    axes[1, 0].set_xlabel("Frequenza")
    axes[1, 0].set_title("Top 20 Parole Più Frequenti")
    axes[1, 0].invert_yaxis()

    # Plot 4: Distribuzione bins
    bin_counts = df["length_bin"].value_counts().sort_index()
    axes[1, 1].bar(range(len(bin_counts)), bin_counts.values, alpha=0.7)
    axes[1, 1].set_xticks(range(len(bin_counts)))
    axes[1, 1].set_xticklabels(bin_counts.index, rotation=45)
    axes[1, 1].set_xlabel("Range Parole")
    axes[1, 1].set_ylabel("Numero Caption")
    axes[1, 1].set_title("Distribuzione Caption per Range")

    plt.tight_layout()
    plt.savefig(output_dir / "dataset_analysis_plots.png", dpi=300, bbox_inches="tight")
    print(f"   ✓ Grafici salvati in: {output_dir / 'dataset_analysis_plots.png'}")

    # 9. Raccomandazioni
    print(f"\n9️⃣  Raccomandazioni per il modello:")
    print(f"\n   📝 Tokenizer:")
    print(f"      - Vocab size consigliato: {min(len(word_counts) + 1000, 10000)}")
    print(f"      - Min frequency: 2 (rimuovi hapax)")
    print(f"      - Special tokens: [PAD], [UNK], [SOS], [EOS]")

    print(f"\n   📐 Modello:")
    print(f"      - Max sequence length: {int(df['num_words'].quantile(0.95)) + 2}")
    print(f"        (copre 95% caption + SOS/EOS)")
    print(f"      - Batch size: considera padding fino a max_len")

    print(f"\n   🔄 Data Augmentation:")
    if df["num_words"].std() > 5:
        print(f"      ⚠️  Alta variabilità lunghezza caption")
        print(f"      → Considera curriculum learning (corte→lunghe)")

    print(f"\n   ⚖️  Balance:")
    emotion_balance = df["emotion"].value_counts()
    min_class = emotion_balance.min()
    max_class = emotion_balance.max()
    if max_class / min_class > 1.2:
        print(f"      ⚠️  Leggero sbilanciamento classi")
        print(f"      → Considera weighted sampling o class weights")
    else:
        print(f"      ✓ Dataset ben bilanciato")

    print(f"\n" + "=" * 70)
    print(f"✅ ANALISI COMPLETATA!")
    print(f"=" * 70)
    print(f"\n📁 Tutti i risultati salvati in: {output_dir}/")
    print(f"   - statistics.json")
    print(f"   - word_frequencies.txt")
    print(f"   - dataset_with_stats.csv")
    print(f"   - dataset_analysis_plots.png")
    print(f"\n")

    return df, stats


def main():
    parser = argparse.ArgumentParser(
        description="Analizza dataset sign language caption"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="data/processed/golden_label_sentiment.csv",
        help="Path al CSV con annotazioni",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/dataset_analysis",
        help="Directory output per risultati",
    )

    args = parser.parse_args()

    # Run analysis
    df, stats = analyze_captions(args.csv_path, args.output_dir)

    print("\n💡 Prossimi step suggeriti:")
    print("   1. Revisionare grafici in results/dataset_analysis/")
    print("   2. Decidere vocab_size e max_seq_length per tokenizer")
    print("   3. Creare train/val/test split (70/15/15)")
    print("   4. Procedere con estrazione landmarks completa")
    print("\n")


if __name__ == "__main__":
    main()
