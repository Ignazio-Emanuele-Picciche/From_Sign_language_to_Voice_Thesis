"""
Audio Comparison Analysis - Analizza e confronta audio emotivi
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Setup path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from explainability.audio.acoustic_analyzer import AcousticAnalyzer


def analyze_audio_directory(audio_dir: Path, output_csv: Path = None) -> pd.DataFrame:
    """
    Analizza tutti gli audio in una directory

    Args:
        audio_dir (Path): Directory contenente i file audio
        output_csv (Path, optional): Path dove salvare il CSV dei risultati

    Returns:
        pd.DataFrame: DataFrame con le analisi
    """
    results = []

    # Trova tutti i file audio
    audio_files = list(audio_dir.glob("*.mp3"))

    print(f"\n{'='*60}")
    print(f"ANALISI AUDIO DIRECTORY")
    print(f"{'='*60}")
    print(f"Directory: {audio_dir}")
    print(f"File trovati: {len(audio_files)}")

    for i, audio_path in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] Analizzando: {audio_path.name}")

        # Determina emozione dal nome file
        filename = audio_path.stem.lower()
        if "positive" in filename:
            emotion = "Positive"
        elif "negative" in filename:
            emotion = "Negative"
        else:
            emotion = "Unknown"

        try:
            # Analizza audio
            analyzer = AcousticAnalyzer(str(audio_path))

            pitch_features = analyzer.extract_pitch_features()
            rate_features = analyzer.extract_rate_features()
            energy_features = analyzer.extract_energy_features()

            result = {
                "filename": audio_path.name,
                "emotion": emotion,
                "mean_pitch_hz": pitch_features.get(
                    "mean_pitch_hz", pitch_features.get("mean_pitch", 0)
                ),
                "std_pitch_hz": pitch_features.get(
                    "std_pitch_hz", pitch_features.get("std_pitch", 0)
                ),
                "min_pitch_hz": pitch_features.get(
                    "min_pitch_hz", pitch_features.get("min_pitch", 0)
                ),
                "max_pitch_hz": pitch_features.get(
                    "max_pitch_hz", pitch_features.get("max_pitch", 0)
                ),
                "range_pitch_hz": pitch_features.get(
                    "range_pitch_hz", pitch_features.get("range_pitch", 0)
                ),
                "jitter_percent": pitch_features.get("jitter", 0),
                "shimmer_percent": pitch_features.get("shimmer", 0),
                "hnr_db": pitch_features.get("hnr", 0),
                "speaking_rate_syll_sec": rate_features.get("speaking_rate", 0),
                "duration_sec": rate_features.get("duration", 0),
                "tempo": rate_features.get("tempo", 0),
                "mean_energy_db": energy_features.get(
                    "mean_energy_db", energy_features.get("mean_energy", 0)
                ),
                "std_energy_db": energy_features.get(
                    "std_energy_db", energy_features.get("std_energy", 0)
                ),
                "max_energy_db": energy_features.get(
                    "max_energy_db", energy_features.get("max_energy", 0)
                ),
                "dynamic_range_db": energy_features.get(
                    "dynamic_range_db", energy_features.get("dynamic_range", 0)
                ),
            }

            results.append(result)

            # Stampa info principali
            print(f"  Emozione: {emotion}")
            print(f"  Pitch medio: {pitch_features.get('mean_pitch_hz', 0):.1f} Hz")
            print(
                f"  Speaking rate: {rate_features.get('speaking_rate', 0):.2f} syll/sec"
            )
            print(f"  Energy: {energy_features.get('mean_energy_db', 0):.1f} dB")

        except Exception as e:
            print(f"  ⚠️ Errore nell'analisi: {e}")
            continue

    # Crea DataFrame
    df = pd.DataFrame(results)

    # Salva CSV se richiesto
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"\n💾 Risultati salvati in: {output_csv}")

    return df


def calculate_statistics(df: pd.DataFrame) -> dict:
    """
    Calcola statistiche descrittive per emozione

    Args:
        df (pd.DataFrame): DataFrame con i risultati

    Returns:
        dict: Dizionario con statistiche
    """
    stats = {}

    # Filtra solo Positive e Negative
    df_clean = df[df["emotion"].isin(["Positive", "Negative"])]

    if len(df_clean) == 0:
        return stats

    for emotion in ["Positive", "Negative"]:
        data = df_clean[df_clean["emotion"] == emotion]

        if len(data) == 0:
            continue

        stats[emotion] = {
            "count": len(data),
            "pitch": {
                "mean": data["mean_pitch_hz"].mean(),
                "std": data["mean_pitch_hz"].std(),
                "min": data["mean_pitch_hz"].min(),
                "max": data["mean_pitch_hz"].max(),
            },
            "rate": {
                "mean": data["speaking_rate_syll_sec"].mean(),
                "std": data["speaking_rate_syll_sec"].std(),
                "min": data["speaking_rate_syll_sec"].min(),
                "max": data["speaking_rate_syll_sec"].max(),
            },
            "energy": {
                "mean": data["mean_energy_db"].mean(),
                "std": data["mean_energy_db"].std(),
                "min": data["mean_energy_db"].min(),
                "max": data["mean_energy_db"].max(),
            },
        }

    # Calcola differenze percentuali
    if "Positive" in stats and "Negative" in stats:
        neg_pitch = stats["Negative"]["pitch"]["mean"]
        neg_rate = stats["Negative"]["rate"]["mean"]
        neg_energy = abs(stats["Negative"]["energy"]["mean"])

        stats["differences"] = {
            "pitch_percent": (
                (stats["Positive"]["pitch"]["mean"] - neg_pitch) / neg_pitch * 100
            ),
            "rate_percent": (
                (stats["Positive"]["rate"]["mean"] - neg_rate) / neg_rate * 100
            ),
            "energy_percent": (
                (
                    stats["Positive"]["energy"]["mean"]
                    - stats["Negative"]["energy"]["mean"]
                )
                / neg_energy
                * 100
            ),
        }

    return stats


def print_statistics_report(stats: dict):
    """
    Stampa un report delle statistiche

    Args:
        stats (dict): Dizionario con statistiche
    """
    print(f"\n{'='*60}")
    print("STATISTICAL SUMMARY")
    print(f"{'='*60}")

    for emotion in ["Positive", "Negative"]:
        if emotion not in stats:
            continue

        s = stats[emotion]
        print(f"\n{emotion.upper()} (n={s['count']}):")
        print(f"  Pitch:  {s['pitch']['mean']:.1f} ± {s['pitch']['std']:.1f} Hz")
        print(f"          Range: [{s['pitch']['min']:.1f}, {s['pitch']['max']:.1f}]")
        print(f"  Rate:   {s['rate']['mean']:.2f} ± {s['rate']['std']:.2f} syll/sec")
        print(f"          Range: [{s['rate']['min']:.2f}, {s['rate']['max']:.2f}]")
        print(f"  Energy: {s['energy']['mean']:.1f} ± {s['energy']['std']:.1f} dB")
        print(f"          Range: [{s['energy']['min']:.1f}, {s['energy']['max']:.1f}]")

    if "differences" in stats:
        diff = stats["differences"]
        print(f"\nDIFFERENCES (Positive vs Negative):")
        print(f"  Pitch:  {diff['pitch_percent']:+.1f}%")
        print(f"  Rate:   {diff['rate_percent']:+.1f}%")
        print(f"  Energy: {diff['energy_percent']:+.1f}%")


def create_comparison_plots(df: pd.DataFrame, output_path: Path):
    """
    Crea grafici comparativi tra emozioni

    Args:
        df (pd.DataFrame): DataFrame con i risultati
        output_path (Path): Path dove salvare il grafico
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Filtra solo Positive e Negative
    df_clean = df[df["emotion"].isin(["Positive", "Negative"])]

    if len(df_clean) == 0:
        print("⚠️ Nessun dato da visualizzare")
        return

    # Setup plot style
    sns.set_style("whitegrid")
    sns.set_palette("Set2")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Audio Prosody Analysis - Emotion Comparison",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    # 1. Pitch comparison
    sns.boxplot(data=df_clean, x="emotion", y="mean_pitch_hz", ax=axes[0, 0])
    sns.swarmplot(
        data=df_clean,
        x="emotion",
        y="mean_pitch_hz",
        ax=axes[0, 0],
        color="black",
        alpha=0.5,
        size=4,
    )
    axes[0, 0].set_title("Pitch Comparison", fontsize=12, fontweight="bold")
    axes[0, 0].set_ylabel("Mean Pitch (Hz)")
    axes[0, 0].set_xlabel("")

    # 2. Rate comparison
    sns.boxplot(data=df_clean, x="emotion", y="speaking_rate_syll_sec", ax=axes[0, 1])
    sns.swarmplot(
        data=df_clean,
        x="emotion",
        y="speaking_rate_syll_sec",
        ax=axes[0, 1],
        color="black",
        alpha=0.5,
        size=4,
    )
    axes[0, 1].set_title("Speaking Rate Comparison", fontsize=12, fontweight="bold")
    axes[0, 1].set_ylabel("Speaking Rate (syllables/sec)")
    axes[0, 1].set_xlabel("")

    # 3. Energy comparison
    sns.boxplot(data=df_clean, x="emotion", y="mean_energy_db", ax=axes[1, 0])
    sns.swarmplot(
        data=df_clean,
        x="emotion",
        y="mean_energy_db",
        ax=axes[1, 0],
        color="black",
        alpha=0.5,
        size=4,
    )
    axes[1, 0].set_title("Energy Comparison", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("Mean Energy (dB)")
    axes[1, 0].set_xlabel("Emotion")

    # 4. Statistical summary
    axes[1, 1].axis("off")

    # Calcola statistiche
    stats = calculate_statistics(df_clean)

    stats_text = "STATISTICAL SUMMARY\n" + "=" * 40 + "\n\n"

    for emotion in ["Positive", "Negative"]:
        if emotion not in stats:
            continue
        s = stats[emotion]
        stats_text += f"{emotion.upper()} (n={s['count']}):\n"
        stats_text += (
            f"  Pitch:  {s['pitch']['mean']:.1f} ± {s['pitch']['std']:.1f} Hz\n"
        )
        stats_text += (
            f"  Rate:   {s['rate']['mean']:.2f} ± {s['rate']['std']:.2f} syll/s\n"
        )
        stats_text += (
            f"  Energy: {s['energy']['mean']:.1f} ± {s['energy']['std']:.1f} dB\n\n"
        )

    if "differences" in stats:
        diff = stats["differences"]
        stats_text += "DIFFERENCES (Positive vs Negative):\n"
        stats_text += f"  Pitch:  {diff['pitch_percent']:+.1f}%\n"
        stats_text += f"  Rate:   {diff['rate_percent']:+.1f}%\n"
        stats_text += f"  Energy: {diff['energy_percent']:+.1f}%\n\n"

        # Interpretazione
        stats_text += "INTERPRETATION:\n"
        if abs(diff["pitch_percent"]) > 3:
            stats_text += "  ✓ Pitch: Significant difference\n"
        else:
            stats_text += "  ⚠ Pitch: Marginal difference\n"

        if abs(diff["rate_percent"]) > 5:
            stats_text += "  ✓ Rate: Significant difference\n"
        else:
            stats_text += "  ⚠ Rate: Marginal difference\n"

        if abs(diff["energy_percent"]) > 2:
            stats_text += "  ✓ Energy: Significant difference\n"
        else:
            stats_text += "  ⚠ Energy: Marginal difference\n"

    axes[1, 1].text(
        0.05,
        0.95,
        stats_text,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        transform=axes[1, 1].transAxes,
    )

    plt.tight_layout()

    # Salva figura
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n📊 Grafico salvato in: {output_path}")

    return fig
