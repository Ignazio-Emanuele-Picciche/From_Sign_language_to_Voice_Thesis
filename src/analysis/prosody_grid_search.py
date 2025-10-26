"""
Grid Search per Parametri Prosodici Ottimali

Questo script:
1. Genera audio con 3 configurazioni di parametri prosodici
2. Analizza acusticamente gli audio generati
3. Confronta le configurazioni per trovare quella ottimale
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Aggiungi src al path
sys.path.append(str(Path(__file__).parent.parent))

from tts.tts_generator import generate_emotional_audio
from tts import emotion_mapper
from explainability.audio.acoustic_analyzer import AcousticAnalyzer


# ============================================================================
# CONFIGURAZIONI DA TESTARE
# ============================================================================

CONFIGURATIONS = {
    "conservative": {
        "name": "Conservative (Attuale)",
        "Positive": {"rate": "+15%", "pitch": "+8%", "volume": "+5%"},
        "Negative": {"rate": "-12%", "pitch": "-6%", "volume": "-3%"},
        "description": "Parametri attuali - conservativi",
    },
    "moderate": {
        "name": "Moderate (+50%)",
        "Positive": {"rate": "+22%", "pitch": "+12%", "volume": "+8%"},
        "Negative": {"rate": "-18%", "pitch": "-9%", "volume": "-5%"},
        "description": "Incremento 50% rispetto a conservative",
    },
    "aggressive": {
        "name": "Aggressive (+100%)",
        "Positive": {"rate": "+30%", "pitch": "+20%", "volume": "+15%"},
        "Negative": {"rate": "-30%", "pitch": "-20%", "volume": "-15%"},
        "description": "Raddoppio parametri - molto marcato",
    },
}


# ============================================================================
# CAMPIONI DI TEST
# ============================================================================

TEST_SAMPLES = [
    {
        "video_id": "test_pos_001",
        "text": "I was like, Oh wow, that is fine.",
        "emotion": "Positive",
        "confidence": 1.0,
    },
    {
        "video_id": "test_pos_002",
        "text": "That's great, I'm really happy about this!",
        "emotion": "Positive",
        "confidence": 0.95,
    },
    {
        "video_id": "test_pos_003",
        "text": "Wonderful news, everything worked out perfectly.",
        "emotion": "Positive",
        "confidence": 0.92,
    },
    {
        "video_id": "test_pos_004",
        "text": "I love this, it makes me feel amazing.",
        "emotion": "Positive",
        "confidence": 0.88,
    },
    {
        "video_id": "test_pos_005",
        "text": "This is exactly what I wanted to hear!",
        "emotion": "Positive",
        "confidence": 0.90,
    },
    {
        "video_id": "test_neg_001",
        "text": "I feel sad and disappointed about this situation.",
        "emotion": "Negative",
        "confidence": 1.0,
    },
    {
        "video_id": "test_neg_002",
        "text": "This is really frustrating and upsetting.",
        "emotion": "Negative",
        "confidence": 0.93,
    },
    {
        "video_id": "test_neg_003",
        "text": "I'm not happy with how things turned out.",
        "emotion": "Negative",
        "confidence": 0.89,
    },
    {
        "video_id": "test_neg_004",
        "text": "Everything went wrong, I'm very upset.",
        "emotion": "Negative",
        "confidence": 0.91,
    },
    {
        "video_id": "test_neg_005",
        "text": "This makes me feel terrible and hopeless.",
        "emotion": "Negative",
        "confidence": 0.87,
    },
]


# ============================================================================
# FUNZIONI PRINCIPALI
# ============================================================================


def generate_audio_for_all_configs(output_base_dir: str = "results/grid_search"):
    """
    Genera audio per tutte le configurazioni
    """
    print("=" * 80)
    print("🎵 GRID SEARCH - GENERAZIONE AUDIO")
    print("=" * 80)

    for config_name, config_data in CONFIGURATIONS.items():
        print(f"\n{'='*80}")
        print(f"📊 Configurazione: {config_data['name']}")
        print(f"    {config_data['description']}")
        print(f"{'='*80}")

        # Crea directory per questa configurazione
        config_dir = Path(output_base_dir) / config_name
        config_dir.mkdir(parents=True, exist_ok=True)

        # Genera audio per ogni campione
        for sample in TEST_SAMPLES:
            emotion = sample["emotion"]

            # Prendi i parametri prosodici per questa configurazione
            prosody_params = config_data[emotion]

            # Nome file output
            output_file = config_dir / f"{sample['video_id']}.mp3"

            # Genera audio con parametri custom
            try:
                # Salva parametri originali
                original_params = emotion_mapper.PROSODY_MAPPING[emotion].copy()

                # Imposta parametri della configurazione corrente
                emotion_mapper.PROSODY_MAPPING[emotion].update(prosody_params)

                # Genera audio usando caption con il testo
                generate_emotional_audio(
                    emotion=emotion,
                    confidence=sample["confidence"],
                    video_name=sample["video_id"],
                    output_dir=str(config_dir),
                    use_simple_template=False,
                    caption=sample["text"],  # Usa il testo personalizzato
                )

                # Ripristina parametri originali
                emotion_mapper.PROSODY_MAPPING[emotion] = original_params

                print(f"  ✅ {sample['video_id']} ({emotion})")

            except Exception as e:
                print(f"  ❌ {sample['video_id']}: {e}")
                import traceback

                traceback.print_exc()

    print(f"\n{'='*80}")
    print(f"✅ Generazione completata!")
    print(f"📁 Audio salvati in: {output_base_dir}/")
    print(f"{'='*80}\n")


def analyze_all_configs(
    input_base_dir: str = "results/grid_search",
    output_dir: str = "results/grid_search/analysis",
):
    """
    Analizza acusticamente tutti gli audio generati
    """
    print("\n" + "=" * 80)
    print("🔬 ANALISI ACUSTICA")
    print("=" * 80)

    all_results = []

    for config_name in CONFIGURATIONS.keys():
        config_dir = Path(input_base_dir) / config_name

        if not config_dir.exists():
            print(f"⚠️  Directory non trovata: {config_dir}")
            continue

        print(f"\n📊 Analizzo configurazione: {config_name}")

        # Analizza tutti gli audio
        audio_files = list(config_dir.glob("*.mp3"))

        for audio_file in audio_files:
            # Estrai info dal nome file (formato: test_pos_001_positive.mp3)
            filename_stem = audio_file.stem

            # Rimuovi il suffisso _positive o _negative
            if filename_stem.endswith("_positive") or filename_stem.endswith(
                "_negative"
            ):
                video_id = "_".join(filename_stem.split("_")[:-1])
            else:
                video_id = filename_stem

            sample = next((s for s in TEST_SAMPLES if s["video_id"] == video_id), None)

            if not sample:
                continue

            # Analizza audio
            try:
                analyzer = AcousticAnalyzer(str(audio_file))

                # Estrai features
                pitch_features = analyzer.extract_pitch_features()
                energy_features = analyzer.extract_energy_features()
                rate_features = analyzer.extract_rate_features()

                # Combina features con nomi consistenti
                features = {
                    "pitch_mean": pitch_features.get("mean_pitch_hz", 0),
                    "pitch_std": pitch_features.get("std_pitch_hz", 0),
                    "energy_mean": energy_features.get("mean_energy_db", 0),
                    "energy_std": energy_features.get(
                        "std_energy_db", 0
                    ),  # Potrebbe non esistere
                    "rate_syllables_per_sec": rate_features.get(
                        "rate_syllables_per_sec", 0
                    ),
                }

                # Aggiungi metadati
                features["config"] = config_name
                features["config_name"] = CONFIGURATIONS[config_name]["name"]
                features["video_id"] = video_id
                features["emotion"] = sample["emotion"]
                features["text"] = sample["text"]

                all_results.append(features)

                print(
                    f"  ✅ {video_id}: pitch={features['pitch_mean']:.1f}Hz, "
                    f"energy={features['energy_mean']:.1f}dB"
                )

            except Exception as e:
                print(f"  ❌ {video_id}: {e}")

    # Crea DataFrame
    df = pd.DataFrame(all_results)

    # Salva risultati
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / "grid_search_results.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n✅ Analisi completata!")
    print(f"📁 Risultati salvati in: {csv_path}")

    return df


def compare_configurations(
    df: pd.DataFrame, output_dir: str = "results/grid_search/analysis"
):
    """
    Confronta le configurazioni e genera report
    """
    print("\n" + "=" * 80)
    print("📈 CONFRONTO CONFIGURAZIONI")
    print("=" * 80)

    output_path = Path(output_dir)

    # ========================================================================
    # 1. STATISTICHE DESCRITTIVE
    # ========================================================================

    print("\n" + "─" * 80)
    print("📊 STATISTICHE PER CONFIGURAZIONE")
    print("─" * 80)

    stats_summary = []

    for config in df["config"].unique():
        config_df = df[df["config"] == config]
        config_name = config_df["config_name"].iloc[0]

        print(f"\n{config_name}:")
        print("-" * 40)

        for emotion in ["Positive", "Negative"]:
            emotion_df = config_df[config_df["emotion"] == emotion]

            if len(emotion_df) == 0:
                continue

            pitch_mean = emotion_df["pitch_mean"].mean()
            pitch_std = emotion_df["pitch_mean"].std()
            energy_mean = emotion_df["energy_mean"].mean()
            energy_std = emotion_df["energy_mean"].std()

            print(f"  {emotion}:")
            print(f"    Pitch:  {pitch_mean:.2f} ± {pitch_std:.2f} Hz")
            print(f"    Energy: {energy_mean:.2f} ± {energy_std:.2f} dB")

            stats_summary.append(
                {
                    "config": config,
                    "config_name": config_name,
                    "emotion": emotion,
                    "pitch_mean": pitch_mean,
                    "pitch_std": pitch_std,
                    "energy_mean": energy_mean,
                    "energy_std": energy_std,
                    "n_samples": len(emotion_df),
                }
            )

    # ========================================================================
    # 2. CALCOLA DIFFERENZE TRA POSITIVE E NEGATIVE
    # ========================================================================

    print("\n" + "─" * 80)
    print("📏 DIFFERENZE POSITIVE vs NEGATIVE")
    print("─" * 80)

    comparison_results = []

    for config in df["config"].unique():
        config_df = df[df["config"] == config]
        config_name = config_df["config_name"].iloc[0]

        pos_df = config_df[config_df["emotion"] == "Positive"]
        neg_df = config_df[config_df["emotion"] == "Negative"]

        if len(pos_df) == 0 or len(neg_df) == 0:
            continue

        # Differenze
        pitch_diff = pos_df["pitch_mean"].mean() - neg_df["pitch_mean"].mean()
        pitch_diff_pct = (pitch_diff / neg_df["pitch_mean"].mean()) * 100

        energy_diff = pos_df["energy_mean"].mean() - neg_df["energy_mean"].mean()
        energy_diff_pct = abs((energy_diff / neg_df["energy_mean"].mean()) * 100)

        # T-test
        from scipy import stats

        pitch_ttest = stats.ttest_ind(pos_df["pitch_mean"], neg_df["pitch_mean"])
        energy_ttest = stats.ttest_ind(pos_df["energy_mean"], neg_df["energy_mean"])

        # Cohen's d
        def cohens_d(group1, group2):
            n1, n2 = len(group1), len(group2)
            var1, var2 = group1.var(), group2.var()
            pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            return (group1.mean() - group2.mean()) / pooled_std

        pitch_d = cohens_d(pos_df["pitch_mean"], neg_df["pitch_mean"])
        energy_d = cohens_d(pos_df["energy_mean"], neg_df["energy_mean"])

        print(f"\n{config_name}:")
        print("-" * 40)
        print(f"  Pitch Difference:")
        print(f"    Δ = {pitch_diff:+.2f} Hz ({pitch_diff_pct:+.2f}%)")
        print(
            f"    t = {pitch_ttest.statistic:.3f}, p = {pitch_ttest.pvalue:.6f} {'***' if pitch_ttest.pvalue < 0.001 else '**' if pitch_ttest.pvalue < 0.01 else '*' if pitch_ttest.pvalue < 0.05 else 'ns'}"
        )
        print(
            f"    Cohen's d = {pitch_d:.3f} ({'small' if abs(pitch_d) < 0.5 else 'medium' if abs(pitch_d) < 0.8 else 'large'})"
        )

        print(f"  Energy Difference:")
        print(f"    Δ = {energy_diff:+.2f} dB ({energy_diff_pct:+.2f}%)")
        print(
            f"    t = {energy_ttest.statistic:.3f}, p = {energy_ttest.pvalue:.6f} {'***' if energy_ttest.pvalue < 0.001 else '**' if energy_ttest.pvalue < 0.01 else '*' if energy_ttest.pvalue < 0.05 else 'ns'}"
        )
        print(
            f"    Cohen's d = {energy_d:.3f} ({'small' if abs(energy_d) < 0.5 else 'medium' if abs(energy_d) < 0.8 else 'large'})"
        )

        comparison_results.append(
            {
                "config": config,
                "config_name": config_name,
                "pitch_diff_hz": pitch_diff,
                "pitch_diff_pct": pitch_diff_pct,
                "pitch_p_value": pitch_ttest.pvalue,
                "pitch_cohens_d": pitch_d,
                "energy_diff_db": energy_diff,
                "energy_diff_pct": energy_diff_pct,
                "energy_p_value": energy_ttest.pvalue,
                "energy_cohens_d": energy_d,
            }
        )

    # Salva confronto
    comparison_df = pd.DataFrame(comparison_results)
    comparison_path = output_path / "configuration_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    # ========================================================================
    # 3. VISUALIZZAZIONI
    # ========================================================================

    print("\n" + "─" * 80)
    print("📊 GENERAZIONE GRAFICI")
    print("─" * 80)

    # Setup style
    sns.set_style("whitegrid")

    # 3.1 Box plots comparativi
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Pitch
    sns.boxplot(data=df, x="config_name", y="pitch_mean", hue="emotion", ax=axes[0])
    axes[0].set_title(
        "Pitch Comparison Across Configurations", fontsize=14, fontweight="bold"
    )
    axes[0].set_xlabel("Configuration", fontsize=12)
    axes[0].set_ylabel("Mean Pitch (Hz)", fontsize=12)
    axes[0].legend(title="Emotion")
    axes[0].grid(True, alpha=0.3)

    # Energy
    sns.boxplot(data=df, x="config_name", y="energy_mean", hue="emotion", ax=axes[1])
    axes[1].set_title(
        "Energy Comparison Across Configurations", fontsize=14, fontweight="bold"
    )
    axes[1].set_xlabel("Configuration", fontsize=12)
    axes[1].set_ylabel("Mean Energy (dB)", fontsize=12)
    axes[1].legend(title="Emotion")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_path / "config_comparison_boxplots.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"  ✅ Boxplots salvati: {plot_path}")
    plt.close()

    # 3.2 Bar plot con effect sizes
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(comparison_df))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        comparison_df["pitch_cohens_d"],
        width,
        label="Pitch",
        color="steelblue",
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2,
        comparison_df["energy_cohens_d"],
        width,
        label="Energy",
        color="coral",
        alpha=0.8,
    )

    ax.set_xlabel("Configuration", fontsize=12)
    ax.set_ylabel("Cohen's d (Effect Size)", fontsize=12)
    ax.set_title("Effect Sizes Across Configurations", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df["config_name"], rotation=15, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Small effect")
    ax.axhline(y=0.8, color="gray", linestyle="--", alpha=0.5, label="Medium effect")

    plt.tight_layout()
    effect_plot_path = output_path / "effect_sizes_comparison.png"
    plt.savefig(effect_plot_path, dpi=300, bbox_inches="tight")
    print(f"  ✅ Effect sizes salvati: {effect_plot_path}")
    plt.close()

    # ========================================================================
    # 4. RACCOMANDAZIONE
    # ========================================================================

    print("\n" + "=" * 80)
    print("🎯 RACCOMANDAZIONE")
    print("=" * 80)

    # Trova configurazione con miglior Cohen's d per pitch
    best_config_row = comparison_df.loc[comparison_df["pitch_cohens_d"].idxmax()]

    print(f"\n🏆 Configurazione Ottimale: {best_config_row['config_name']}")
    print(f"\n   Motivo:")
    print(f"   • Pitch Cohen's d = {best_config_row['pitch_cohens_d']:.3f} (massimo)")
    print(f"   • Pitch p-value = {best_config_row['pitch_p_value']:.6f}")
    print(f"   • Energy Cohen's d = {best_config_row['energy_cohens_d']:.3f}")
    print(f"   • Energy p-value = {best_config_row['energy_p_value']:.6f}")

    print(f"\n   Parametri da usare:")
    best_config = best_config_row["config"]
    print(f"   {CONFIGURATIONS[best_config]['Positive']}")
    print(f"   {CONFIGURATIONS[best_config]['Negative']}")

    print("\n" + "=" * 80)

    return comparison_df


# ============================================================================
# MAIN
# ============================================================================


def main():
    """
    Esegue la grid search completa
    """
    print("\n")
    print("=" * 80)
    print("🔬 GRID SEARCH PER PARAMETRI PROSODICI OTTIMALI")
    print("=" * 80)
    print("\nQuesto script testerà 3 configurazioni di parametri prosodici:")
    for config_name, config_data in CONFIGURATIONS.items():
        print(f"\n  • {config_data['name']}")
        print(f"    Positive: {config_data['Positive']}")
        print(f"    Negative: {config_data['Negative']}")

    print(f"\nSaranno generati {len(TEST_SAMPLES)} audio per configurazione")
    print(f"Totale: {len(TEST_SAMPLES) * len(CONFIGURATIONS)} file audio\n")

    input("Premi ENTER per iniziare...")

    # Step 1: Genera audio
    generate_audio_for_all_configs()

    # Step 2: Analizza
    df = analyze_all_configs()

    # Step 3: Confronta
    comparison_df = compare_configurations(df)

    print("\n" + "=" * 80)
    print("✅ GRID SEARCH COMPLETATA!")
    print("=" * 80)
    print("\nFile generati:")
    print("  📁 results/grid_search/conservative/")
    print("  📁 results/grid_search/moderate/")
    print("  📁 results/grid_search/aggressive/")
    print("  📊 results/grid_search/analysis/grid_search_results.csv")
    print("  📊 results/grid_search/analysis/configuration_comparison.csv")
    print("  📈 results/grid_search/analysis/config_comparison_boxplots.png")
    print("  📈 results/grid_search/analysis/effect_sizes_comparison.png")
    print("\n")


if __name__ == "__main__":
    main()
