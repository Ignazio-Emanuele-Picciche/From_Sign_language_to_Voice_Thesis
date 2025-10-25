"""
Audio Analysis Runner - Script principale per analisi audio
"""

import sys
from pathlib import Path
import argparse

# Setup path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR / "src"))

from analysis.audio_comparison import (
    analyze_audio_directory,
    create_comparison_plots,
    calculate_statistics,
    print_statistics_report,
)
from analysis.statistical_tests import (
    run_statistical_analysis,
    run_normality_tests,
    create_statistical_report,
)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Analizza e confronta audio emotivi generati dal TTS"
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="test_varied_audio",
        help="Directory contenente gli audio da analizzare",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/analysis",
        help="Directory dove salvare i risultati",
    )
    parser.add_argument(
        "--skip_plots", action="store_true", help="Salta la generazione dei grafici"
    )
    parser.add_argument(
        "--skip_stats", action="store_true", help="Salta i test statistici"
    )

    args = parser.parse_args()

    # Path setup
    audio_dir = BASE_DIR / args.audio_dir
    output_dir = BASE_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("AUDIO ANALYSIS - TTS EMOTION COMPARISON")
    print("=" * 70)
    print(f"Audio directory: {audio_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Step 1: Analizza tutti gli audio
    print("\n" + "=" * 70)
    print("STEP 1: ACOUSTIC ANALYSIS")
    print("=" * 70)

    csv_path = output_dir / "audio_analysis_results.csv"
    df = analyze_audio_directory(audio_dir, csv_path)

    if len(df) == 0:
        print("\n⚠️ Nessun audio trovato o analizzato!")
        return

    print(f"\n✅ Analizzati {len(df)} file audio")

    # Step 2: Statistiche descrittive
    print("\n" + "=" * 70)
    print("STEP 2: DESCRIPTIVE STATISTICS")
    print("=" * 70)

    stats = calculate_statistics(df)
    print_statistics_report(stats)

    # Step 3: Test statistici
    if not args.skip_stats:
        print("\n" + "=" * 70)
        print("STEP 3: STATISTICAL TESTS")
        print("=" * 70)

        # Normality tests
        normality_results = run_normality_tests(df)

        # T-tests
        stat_results = run_statistical_analysis(df)

        # Salva report completo
        report_path = output_dir / "statistical_report.txt"
        create_statistical_report(df, report_path)

    # Step 4: Visualizzazioni
    if not args.skip_plots:
        print("\n" + "=" * 70)
        print("STEP 4: VISUALIZATIONS")
        print("=" * 70)

        plot_path = output_dir / "emotion_comparison_plots.png"
        create_comparison_plots(df, plot_path)

    # Step 5: Summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  📊 Data: {csv_path}")
    if not args.skip_plots:
        print(f"  📈 Plots: {plot_path}")
    if not args.skip_stats:
        print(f"  📄 Report: {report_path}")

    # Quick summary
    print(f"\n{'='*70}")
    print("QUICK SUMMARY")
    print(f"{'='*70}")

    df_clean = df[df["emotion"].isin(["Positive", "Negative"])]

    if "differences" in stats:
        diff = stats["differences"]
        print(f"\nDifferences (Positive vs Negative):")
        print(f"  Pitch:  {diff['pitch_percent']:+.1f}%")
        print(f"  Rate:   {diff['rate_percent']:+.1f}%")
        print(f"  Energy: {diff['energy_percent']:+.1f}%")

        print(f"\nInterpretation:")
        if abs(diff["pitch_percent"]) > 3:
            print(f"  ✓ Pitch shows significant difference")
        else:
            print(f"  ⚠ Pitch shows marginal difference")

        if abs(diff["rate_percent"]) > 5:
            print(f"  ✓ Rate shows significant difference")
        else:
            print(f"  ⚠ Rate shows marginal difference")

        if abs(diff["energy_percent"]) > 2:
            print(f"  ✓ Energy shows significant difference")
        else:
            print(f"  ⚠ Energy shows marginal difference")

    print(f"\n{'='*70}")
    print("✅ Analysis completed successfully!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
