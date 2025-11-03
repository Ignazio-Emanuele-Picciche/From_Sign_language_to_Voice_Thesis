"""
Analisi Dataset Utterances
===========================

Analizza il dataset utterances_with_translations.csv (2127 video)
per capire statistiche e preparare per training Sign-to-Text.

Usage:
    python analyze_utterances_dataset.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import re
import json


def clean_text(text):
    """Pulisce il testo per analisi."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_simple(text):
    """Tokenizzazione semplice."""
    return clean_text(text).split()


def analyze_utterances_dataset(
    csv_path="data/processed/utterances_with_translations.csv",
    video_dir="data/raw/ASLLRP/batch_utterance_video_v3_1",
    output_dir="results/utterances_analysis",
):
    """Analizza dataset utterances."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("📊 ANALISI DATASET UTTERANCES (Sign-to-Text)")
    print("=" * 80)

    # 1. Carica dataset
    print(f"\n1️⃣  Caricamento dataset...")
    df = pd.read_csv(csv_path)
    print(f"   ✓ Caricati {len(df)} utterances")
    print(f"   ✓ Colonne: {df.columns.tolist()}")

    # 2. Check video files esistenti
    print(f"\n2️⃣  Verifica video files...")
    video_dir = Path(video_dir)
    existing_videos = []
    missing_videos = []

    for idx, row in df.iterrows():
        video_path = video_dir / row["video_name"]
        if video_path.exists():
            existing_videos.append(row["video_name"])
        else:
            missing_videos.append(row["video_name"])

        if (idx + 1) % 500 == 0:
            print(f"   Verificati {idx + 1}/{len(df)} video...")

    print(f"\n   📁 Video disponibili:")
    print(
        f"      - Esistenti: {len(existing_videos)} ({len(existing_videos)/len(df)*100:.1f}%)"
    )
    print(
        f"      - Mancanti:  {len(missing_videos)} ({len(missing_videos)/len(df)*100:.1f}%)"
    )

    # Filtra solo video esistenti + caption valide
    df["video_exists"] = df["video_name"].apply(lambda x: (video_dir / x).exists())
    df["has_caption"] = ~df["caption"].isna()
    df_available = df[df["video_exists"] & df["has_caption"]].copy()

    print(
        f"\n   ✓ Dataset filtrato: {len(df_available)} utterances con video e caption"
    )
    if len(df[~df["has_caption"]]) > 0:
        print(f"   ⚠️  Caption mancanti: {len(df[~df['has_caption']])} utterances")

    # 3. Analisi caption
    print(f"\n3️⃣  Analisi caption...")

    df_available["num_words"] = df_available["caption"].apply(
        lambda x: len(tokenize_simple(x))
    )
    df_available["num_chars"] = df_available["caption"].apply(len)

    print(f"\n   📏 Lunghezza in PAROLE:")
    print(f"      - Min:    {df_available['num_words'].min()}")
    print(f"      - Max:    {df_available['num_words'].max()}")
    print(f"      - Mean:   {df_available['num_words'].mean():.2f}")
    print(f"      - Median: {df_available['num_words'].median():.0f}")
    print(f"      - Std:    {df_available['num_words'].std():.2f}")

    # 4. Vocabulary
    print(f"\n4️⃣  Analisi vocabulary...")

    all_words = []
    for caption in df_available["caption"]:
        all_words.extend(tokenize_simple(caption))

    word_counts = Counter(all_words)

    print(f"   📚 Vocabulary:")
    print(f"      - Unique words: {len(word_counts)}")
    print(f"      - Total words:  {len(all_words)}")
    print(f"      - Avg frequency: {len(all_words) / len(word_counts):.2f}")

    print(f"\n   🔝 Top 20 parole:")
    for word, count in word_counts.most_common(20):
        print(f"      - '{word}': {count} ({count/len(all_words)*100:.1f}%)")

    # 5. Source collections
    print(f"\n5️⃣  Analisi per sorgente...")

    source_counts = df_available["Source collection"].value_counts()
    print(f"\n   📂 Collections:")
    for source, count in source_counts.head(10).items():
        avg_words = df_available[df_available["Source collection"] == source][
            "num_words"
        ].mean()
        print(f"      - {source}: {count} utterances (avg {avg_words:.1f} words)")

    # 6. Train/Val/Test Split
    print(f"\n6️⃣  Creazione split train/val/test...")

    # Shuffle e split
    df_shuffled = df_available.sample(frac=1, random_state=42).reset_index(drop=True)

    n_total = len(df_shuffled)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val

    train_df = df_shuffled[:n_train]
    val_df = df_shuffled[n_train : n_train + n_val]
    test_df = df_shuffled[n_train + n_val :]

    print(f"\n   ✂️  Split:")
    print(f"      - Train: {len(train_df)} ({len(train_df)/n_total*100:.1f}%)")
    print(f"      - Val:   {len(val_df)} ({len(val_df)/n_total*100:.1f}%)")
    print(f"      - Test:  {len(test_df)} ({len(test_df)/n_total*100:.1f}%)")

    # 7. Salva risultati
    print(f"\n7️⃣  Salvataggio risultati...")

    # Statistics JSON
    stats = {
        "total_utterances": int(len(df)),
        "available_utterances": int(len(df_available)),
        "missing_videos": int(len(missing_videos)),
        "total_words": int(len(all_words)),
        "unique_words": int(len(word_counts)),
        "vocab_size": int(len(word_counts)),
        "avg_words_per_caption": float(df_available["num_words"].mean()),
        "median_words_per_caption": float(df_available["num_words"].median()),
        "min_words": int(df_available["num_words"].min()),
        "max_words": int(df_available["num_words"].max()),
        "std_words": float(df_available["num_words"].std()),
        "train_size": int(len(train_df)),
        "val_size": int(len(val_df)),
        "test_size": int(len(test_df)),
    }

    with open(output_dir / "statistics.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"   ✓ Statistiche: {output_dir / 'statistics.json'}")

    # Save splits
    train_df.to_csv(output_dir / "train_split.csv", index=False)
    val_df.to_csv(output_dir / "val_split.csv", index=False)
    test_df.to_csv(output_dir / "test_split.csv", index=False)
    print(
        f"   ✓ Splits salvati: {output_dir}/train_split.csv, val_split.csv, test_split.csv"
    )

    # Word frequencies
    with open(output_dir / "word_frequencies.txt", "w") as f:
        for word, count in word_counts.most_common():
            f.write(f"{word}\t{count}\n")
    print(f"   ✓ Word frequencies: {output_dir / 'word_frequencies.txt'}")

    # Missing videos list
    if missing_videos:
        with open(output_dir / "missing_videos.txt", "w") as f:
            for video in missing_videos:
                f.write(f"{video}\n")
        print(f"   ✓ Missing videos: {output_dir / 'missing_videos.txt'}")

    # 8. Visualizzazioni
    print(f"\n8️⃣  Creazione visualizzazioni...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Distribuzione lunghezza
    axes[0, 0].hist(df_available["num_words"], bins=50, edgecolor="black", alpha=0.7)
    axes[0, 0].set_xlabel("Numero Parole")
    axes[0, 0].set_ylabel("Frequenza")
    axes[0, 0].set_title(
        f"Distribuzione Lunghezza Caption ({len(df_available)} utterances)"
    )
    mean_words = df_available["num_words"].mean()
    axes[0, 0].axvline(
        mean_words, color="red", linestyle="--", label=f"Media: {mean_words:.1f}"
    )
    axes[0, 0].legend()

    # Plot 2: Top sources
    top_sources = source_counts.head(10)
    axes[0, 1].barh(range(len(top_sources)), top_sources.values, alpha=0.7)
    axes[0, 1].set_yticks(range(len(top_sources)))
    axes[0, 1].set_yticklabels([s[:30] for s in top_sources.index])
    axes[0, 1].set_xlabel("Numero Utterances")
    axes[0, 1].set_title("Top 10 Source Collections")
    axes[0, 1].invert_yaxis()

    # Plot 3: Top words
    top_words = word_counts.most_common(20)
    words, counts = zip(*top_words)
    axes[1, 0].barh(range(len(words)), counts, alpha=0.7)
    axes[1, 0].set_yticks(range(len(words)))
    axes[1, 0].set_yticklabels(words)
    axes[1, 0].set_xlabel("Frequenza")
    axes[1, 0].set_title("Top 20 Parole Più Frequenti")
    axes[1, 0].invert_yaxis()

    # Plot 4: Split sizes
    split_labels = ["Train", "Val", "Test"]
    split_sizes = [len(train_df), len(val_df), len(test_df)]
    colors = ["#4CAF50", "#FFC107", "#2196F3"]
    axes[1, 1].pie(split_sizes, labels=split_labels, autopct="%1.1f%%", colors=colors)
    axes[1, 1].set_title("Train/Val/Test Split")

    plt.tight_layout()
    plt.savefig(
        output_dir / "utterances_analysis_plots.png", dpi=300, bbox_inches="tight"
    )
    print(f"   ✓ Grafici: {output_dir / 'utterances_analysis_plots.png'}")

    # 9. Raccomandazioni
    print(f"\n9️⃣  Raccomandazioni modello:")
    print(f"\n   📝 Tokenizer:")
    print(f"      - Vocab size: {min(len(word_counts) + 1000, 15000)}")
    print(
        f"      - Max sequence length: {int(df_available['num_words'].quantile(0.95)) + 2}"
    )

    print(f"\n   🎯 Training:")
    print(f"      - Batch size: 16-32 (dipende da GPU)")
    print(f"      - Curriculum learning: consigliato (varietà lunghezze)")
    print(f"      - Data augmentation: temporal subsampling landmarks")

    print(f"\n" + "=" * 80)
    print(f"✅ ANALISI COMPLETATA!")
    print(f"=" * 80)
    print(f"\n📁 Risultati in: {output_dir}/")
    print(
        f"\n🚀 Prossimo step: Estrarre landmarks MediaPipe da {len(df_available)} video"
    )
    print(f"   Comando: python extract_landmarks_mediapipe.py")
    print(f"\n")

    return df_available, train_df, val_df, test_df, stats


if __name__ == "__main__":
    analyze_utterances_dataset()
