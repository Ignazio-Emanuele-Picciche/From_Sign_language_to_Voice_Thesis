"""
Prosody Validator - Valida che i parametri prosodici siano stati applicati correttamente
"""

import numpy as np
from typing import Dict, List
from .acoustic_analyzer import AcousticAnalyzer


def validate_prosody(
    generated_audio_path: str,
    baseline_audio_path: str,
    target_prosody: Dict[str, str],
    tolerance: float = 0.15,
) -> Dict:
    """
    Valida che i parametri prosodici target siano stati applicati all'audio generato

    Args:
        generated_audio_path (str): Path audio emotivo generato
        baseline_audio_path (str): Path audio baseline neutrale
        target_prosody (dict): Parametri target {'rate': '+15%', 'pitch': '+8%', 'volume': '+5%'}
        tolerance (float): Tolleranza per considerare "applicato" (default: 15%)

    Returns:
        dict: Report di validazione completo

    Example:
        >>> validate_prosody('positive.mp3', 'baseline.mp3', {'rate': '+15%', 'pitch': '+8%', ...})
        {
            'pitch_delta_measured': 7.8,
            'pitch_delta_target': 8.0,
            'pitch_accuracy': 0.975,
            'pitch_applied': True,
            ...
        }
    """
    # Analizza entrambi gli audio
    print(f"Analyzing baseline: {baseline_audio_path}")
    baseline_analyzer = AcousticAnalyzer(baseline_audio_path)
    baseline_features = baseline_analyzer.get_all_features()

    print(f"Analyzing generated: {generated_audio_path}")
    generated_analyzer = AcousticAnalyzer(generated_audio_path)
    generated_features = generated_analyzer.get_all_features()

    # Estrai target values
    def parse_prosody_value(value_str: str) -> float:
        """Converte '+15%' -> 15.0, '-12%' -> -12.0"""
        return float(value_str.replace("%", ""))

    target_pitch_delta = parse_prosody_value(target_prosody["pitch"])
    target_rate_delta = parse_prosody_value(target_prosody["rate"])
    target_volume_delta = parse_prosody_value(target_prosody["volume"])

    # --- PITCH VALIDATION ---
    baseline_pitch = baseline_features["mean_pitch_hz"]
    generated_pitch = generated_features["mean_pitch_hz"]

    if baseline_pitch > 0:
        measured_pitch_delta = (
            (generated_pitch - baseline_pitch) / baseline_pitch
        ) * 100
    else:
        measured_pitch_delta = 0.0

    pitch_error = abs(measured_pitch_delta - target_pitch_delta)
    pitch_accuracy = (
        max(0.0, 1.0 - (pitch_error / abs(target_pitch_delta)))
        if target_pitch_delta != 0
        else 1.0
    )
    pitch_applied = pitch_error <= (abs(target_pitch_delta) * tolerance)

    # --- RATE VALIDATION ---
    baseline_rate = baseline_features["speaking_rate_syll_sec"]
    generated_rate = generated_features["speaking_rate_syll_sec"]

    if baseline_rate > 0:
        measured_rate_delta = ((generated_rate - baseline_rate) / baseline_rate) * 100
    else:
        measured_rate_delta = 0.0

    rate_error = abs(measured_rate_delta - target_rate_delta)
    rate_accuracy = (
        max(0.0, 1.0 - (rate_error / abs(target_rate_delta)))
        if target_rate_delta != 0
        else 1.0
    )
    rate_applied = rate_error <= (abs(target_rate_delta) * tolerance)

    # --- VOLUME VALIDATION ---
    baseline_volume = baseline_features["mean_energy_db"]
    generated_volume = generated_features["mean_energy_db"]

    # Per dB, il delta è già in scala logaritmica
    # Approssimazione: ogni +3dB ≈ raddoppio percezione volume
    # Semplifichiamo: confrontiamo direttamente dB difference
    measured_volume_delta_db = generated_volume - baseline_volume

    # Converti target % in aspettativa dB (approssimazione)
    # +5% volume ≈ +0.4 dB, +10% ≈ +0.8 dB (approssimato)
    expected_volume_delta_db = target_volume_delta * 0.08

    volume_error_db = abs(measured_volume_delta_db - expected_volume_delta_db)
    volume_accuracy = max(
        0.0, 1.0 - (volume_error_db / max(0.5, abs(expected_volume_delta_db)))
    )
    volume_applied = volume_error_db <= 1.0  # Tolleranza 1 dB

    # --- OVERALL VALIDATION ---
    overall_accuracy = (pitch_accuracy + rate_accuracy + volume_accuracy) / 3
    all_applied = pitch_applied and rate_applied and volume_applied

    # Costruisci report
    report = {
        # Pitch
        "pitch_delta_target": target_pitch_delta,
        "pitch_delta_measured": round(measured_pitch_delta, 2),
        "pitch_accuracy": round(pitch_accuracy, 3),
        "pitch_applied": pitch_applied,
        "pitch_baseline_hz": round(baseline_pitch, 2),
        "pitch_generated_hz": round(generated_pitch, 2),
        # Rate
        "rate_delta_target": target_rate_delta,
        "rate_delta_measured": round(measured_rate_delta, 2),
        "rate_accuracy": round(rate_accuracy, 3),
        "rate_applied": rate_applied,
        "rate_baseline_syll_sec": round(baseline_rate, 2),
        "rate_generated_syll_sec": round(generated_rate, 2),
        # Volume
        "volume_delta_target": target_volume_delta,
        "volume_delta_measured_db": round(measured_volume_delta_db, 2),
        "volume_accuracy": round(volume_accuracy, 3),
        "volume_applied": volume_applied,
        "volume_baseline_db": round(baseline_volume, 2),
        "volume_generated_db": round(generated_volume, 2),
        # Overall
        "overall_accuracy": round(overall_accuracy, 3),
        "all_prosody_applied": all_applied,
        # Raw features
        "baseline_features": baseline_features,
        "generated_features": generated_features,
    }

    return report


def print_validation_report(report: Dict):
    """
    Stampa un report formattato dei risultati di validazione
    """
    print(f"\n{'='*70}")
    print("PROSODY VALIDATION REPORT")
    print(f"{'='*70}")

    print(f"\n📊 OVERALL ACCURACY: {report['overall_accuracy']*100:.1f}%")
    print(f"✅ All prosody applied: {'YES' if report['all_prosody_applied'] else 'NO'}")

    print(f"\n{'─'*70}")
    print("PITCH (Frequenza Fondamentale)")
    print(f"{'─'*70}")
    print(f"  Target delta:    {report['pitch_delta_target']:+.1f}%")
    print(f"  Measured delta:  {report['pitch_delta_measured']:+.1f}%")
    print(f"  Accuracy:        {report['pitch_accuracy']*100:.1f}%")
    print(f"  Applied:         {'✅ YES' if report['pitch_applied'] else '❌ NO'}")
    print(f"  Baseline:        {report['pitch_baseline_hz']:.1f} Hz")
    print(f"  Generated:       {report['pitch_generated_hz']:.1f} Hz")

    print(f"\n{'─'*70}")
    print("RATE (Velocità di Eloquio)")
    print(f"{'─'*70}")
    print(f"  Target delta:    {report['rate_delta_target']:+.1f}%")
    print(f"  Measured delta:  {report['rate_delta_measured']:+.1f}%")
    print(f"  Accuracy:        {report['rate_accuracy']*100:.1f}%")
    print(f"  Applied:         {'✅ YES' if report['rate_applied'] else '❌ NO'}")
    print(f"  Baseline:        {report['rate_baseline_syll_sec']:.2f} syll/sec")
    print(f"  Generated:       {report['rate_generated_syll_sec']:.2f} syll/sec")

    print(f"\n{'─'*70}")
    print("VOLUME (Energia)")
    print(f"{'─'*70}")
    print(f"  Target delta:    {report['volume_delta_target']:+.1f}%")
    print(f"  Measured delta:  {report['volume_delta_measured_db']:+.2f} dB")
    print(f"  Accuracy:        {report['volume_accuracy']*100:.1f}%")
    print(f"  Applied:         {'✅ YES' if report['volume_applied'] else '❌ NO'}")
    print(f"  Baseline:        {report['volume_baseline_db']:.1f} dB")
    print(f"  Generated:       {report['volume_generated_db']:.1f} dB")


def validate_prosody_batch(
    audio_pairs: List[tuple], target_prosodies: List[Dict], baseline_audio_path: str
) -> List[Dict]:
    """
    Valida prosody per un batch di audio

    Args:
        audio_pairs (list): Lista di (generated_path, emotion) tuples
        target_prosodies (list): Lista di dict con target prosody
        baseline_audio_path (str): Path audio baseline

    Returns:
        list: Lista di report di validazione
    """
    reports = []

    for (generated_path, emotion), target_prosody in zip(audio_pairs, target_prosodies):
        try:
            report = validate_prosody(
                generated_path, baseline_audio_path, target_prosody
            )
            report["audio_path"] = generated_path
            report["emotion"] = emotion
            reports.append(report)
        except Exception as e:
            print(f"Errore validazione {generated_path}: {e}")
            reports.append(None)

    return reports


if __name__ == "__main__":
    # Test del modulo
    import sys
    import os

    print("=" * 70)
    print("TEST PROSODY VALIDATOR")
    print("=" * 70)

    # Paths
    baseline_path = "test_tts_output/baseline.mp3"
    positive_path = "test_tts_output/test_positive.mp3"

    if not os.path.exists(baseline_path) or not os.path.exists(positive_path):
        print(f"\n⚠️  File di test non trovati. Genera prima con tts_generator.py")
        sys.exit(1)

    # Target prosody per Positive
    target_prosody = {"rate": "+15%", "pitch": "+8%", "volume": "+5%"}

    # Valida
    print(f"\nValidating: {positive_path} vs {baseline_path}")
    report = validate_prosody(positive_path, baseline_path, target_prosody)

    # Stampa report
    print_validation_report(report)

    print(f"\n{'='*70}")
    print("✅ Test completato!")
