"""
Statistical Tests - Test statistici per confronto emozioni
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple


def run_statistical_analysis(df: pd.DataFrame) -> Dict:
    """
    Esegue test statistici per confrontare Positive vs Negative

    Args:
        df (pd.DataFrame): DataFrame con i risultati

    Returns:
        Dict: Risultati dei test statistici
    """
    results = {}

    # Filtra solo Positive e Negative
    df_clean = df[df["emotion"].isin(["Positive", "Negative"])]

    if len(df_clean) == 0:
        return results

    positive_data = df_clean[df_clean["emotion"] == "Positive"]
    negative_data = df_clean[df_clean["emotion"] == "Negative"]

    if len(positive_data) == 0 or len(negative_data) == 0:
        print("⚠️ Dati insufficienti per test statistici")
        return results

    print(f"\n{'='*60}")
    print("STATISTICAL TESTS (Independent t-test)")
    print(f"{'='*60}")
    print(f"Sample sizes: Positive={len(positive_data)}, Negative={len(negative_data)}")

    # Features da testare
    features = {
        "mean_pitch_hz": "Pitch (Hz)",
        "speaking_rate_syll_sec": "Speaking Rate (syll/sec)",
        "mean_energy_db": "Energy (dB)",
    }

    for col, label in features.items():
        pos_values = positive_data[col].dropna()
        neg_values = negative_data[col].dropna()

        if len(pos_values) < 2 or len(neg_values) < 2:
            continue

        # T-test
        t_stat, p_value = stats.ttest_ind(pos_values, neg_values)

        # Effect size (Cohen's d)
        cohens_d = calculate_cohens_d(pos_values, neg_values)

        # Differenza percentuale
        mean_pos = pos_values.mean()
        mean_neg = neg_values.mean()
        percent_diff = ((mean_pos - mean_neg) / abs(mean_neg)) * 100

        results[col] = {
            "feature": label,
            "positive_mean": mean_pos,
            "positive_std": pos_values.std(),
            "negative_mean": mean_neg,
            "negative_std": neg_values.std(),
            "percent_difference": percent_diff,
            "t_statistic": t_stat,
            "p_value": p_value,
            "cohens_d": cohens_d,
            "significant": p_value < 0.05,
        }

        # Stampa risultati
        print(f"\n{label}:")
        print(f"  Positive: {mean_pos:.2f} ± {pos_values.std():.2f}")
        print(f"  Negative: {mean_neg:.2f} ± {neg_values.std():.2f}")
        print(f"  Difference: {percent_diff:+.1f}%")
        print(f"  t-statistic: {t_stat:.3f}")
        print(
            f"  p-value: {p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}"
        )
        print(f"  Cohen's d: {cohens_d:.3f} ({interpret_cohens_d(cohens_d)})")

    return results


def calculate_cohens_d(group1: pd.Series, group2: pd.Series) -> float:
    """
    Calcola Cohen's d per effect size

    Args:
        group1: Dati gruppo 1
        group2: Dati gruppo 2

    Returns:
        float: Cohen's d
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    # Cohen's d
    d = (group1.mean() - group2.mean()) / pooled_std

    return d


def interpret_cohens_d(d: float) -> str:
    """
    Interpreta il valore di Cohen's d

    Args:
        d (float): Cohen's d

    Returns:
        str: Interpretazione
    """
    abs_d = abs(d)

    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def run_normality_tests(df: pd.DataFrame) -> Dict:
    """
    Test di normalità (Shapiro-Wilk) per verificare assunzioni del t-test

    Args:
        df (pd.DataFrame): DataFrame con i risultati

    Returns:
        Dict: Risultati test normalità
    """
    results = {}

    df_clean = df[df["emotion"].isin(["Positive", "Negative"])]

    features = ["mean_pitch_hz", "speaking_rate_syll_sec", "mean_energy_db"]

    print(f"\n{'='*60}")
    print("NORMALITY TESTS (Shapiro-Wilk)")
    print(f"{'='*60}")

    for emotion in ["Positive", "Negative"]:
        data = df_clean[df_clean["emotion"] == emotion]

        print(f"\n{emotion}:")

        for col in features:
            values = data[col].dropna()

            if len(values) < 3:
                continue

            stat, p_value = stats.shapiro(values)

            results[f"{emotion}_{col}"] = {
                "statistic": stat,
                "p_value": p_value,
                "normal": p_value > 0.05,
            }

            print(
                f"  {col}: W={stat:.4f}, p={p_value:.4f} {'(Normal)' if p_value > 0.05 else '(Not Normal)'}"
            )

    return results


def create_statistical_report(df: pd.DataFrame, output_path: str):
    """
    Crea un report completo dei test statistici

    Args:
        df (pd.DataFrame): DataFrame con i risultati
        output_path (str): Path dove salvare il report
    """
    from pathlib import Path

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("STATISTICAL ANALYSIS REPORT - TTS AUDIO COMPARISON\n")
        f.write("=" * 70 + "\n\n")

        # 1. Descriptive statistics
        f.write("1. DESCRIPTIVE STATISTICS\n")
        f.write("-" * 70 + "\n\n")

        df_clean = df[df["emotion"].isin(["Positive", "Negative"])]

        for emotion in ["Positive", "Negative"]:
            data = df_clean[df_clean["emotion"] == emotion]
            f.write(f"{emotion} (n={len(data)}):\n")
            f.write(
                f"  Pitch (Hz):        {data['mean_pitch_hz'].mean():.2f} ± {data['mean_pitch_hz'].std():.2f}\n"
            )
            f.write(
                f"  Rate (syll/sec):   {data['speaking_rate_syll_sec'].mean():.2f} ± {data['speaking_rate_syll_sec'].std():.2f}\n"
            )
            f.write(
                f"  Energy (dB):       {data['mean_energy_db'].mean():.2f} ± {data['mean_energy_db'].std():.2f}\n\n"
            )

        # 2. Normality tests
        f.write("\n2. NORMALITY TESTS (Shapiro-Wilk)\n")
        f.write("-" * 70 + "\n\n")

        normality_results = run_normality_tests(df)

        # 3. T-tests
        f.write("\n3. INDEPENDENT T-TESTS (Positive vs Negative)\n")
        f.write("-" * 70 + "\n\n")

        stat_results = run_statistical_analysis(df)

        for col, result in stat_results.items():
            f.write(f"{result['feature']}:\n")
            f.write(
                f"  Positive: {result['positive_mean']:.2f} ± {result['positive_std']:.2f}\n"
            )
            f.write(
                f"  Negative: {result['negative_mean']:.2f} ± {result['negative_std']:.2f}\n"
            )
            f.write(f"  Difference: {result['percent_difference']:+.1f}%\n")
            f.write(f"  t({result['t_statistic']:.3f}), p={result['p_value']:.4f}\n")
            f.write(
                f"  Cohen's d: {result['cohens_d']:.3f} ({interpret_cohens_d(result['cohens_d'])})\n"
            )
            f.write(f"  Significance: {'YES' if result['significant'] else 'NO'}\n\n")

        # 4. Interpretazione
        f.write("\n4. INTERPRETATION\n")
        f.write("-" * 70 + "\n\n")

        sig_features = [r["feature"] for r in stat_results.values() if r["significant"]]

        if len(sig_features) > 0:
            f.write(f"Significant differences found in: {', '.join(sig_features)}\n\n")
            f.write("This confirms that the TTS system successfully applies\n")
            f.write("prosodic modulation that is measurably different between\n")
            f.write("Positive and Negative emotions.\n")
        else:
            f.write("No significant differences found. Consider:\n")
            f.write("- Increasing prosody parameters\n")
            f.write("- Larger sample size\n")
            f.write("- Different TTS engine with better emotion control\n")

    print(f"\n📄 Report salvato in: {output_path}")
