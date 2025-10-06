# =================================================================================================
# ANALISI COMPARATIVA TRA DATI DI TRAINING E GOLDEN LABELS
# =================================================================================================
#
# Questo script confronta le statistiche dei features estratti dai golden labels
# con quelli del dataset di training per identificare possibili discrepanze
# che potrebbero causare il problema di classificazione.
#
# COMANDO PER ESEGUIRE:
# python src/models/analyze_data_distribution.py
#
# =================================================================================================

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# --- Setup del Percorso di Base e Import delle Utilità ---
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
sys.path.insert(0, BASE_DIR)

from src.utils.training_utils import get_data_paths, get_datasets
from src.models.test_golden_labels import GoldenLabelDataset
from torch.utils.data import DataLoader


def compute_dataset_statistics(dataset, name):
    """
    Calcola statistiche complete per un dataset
    """
    print(f"\n{'='*60}")
    print(f"ANALISI STATISTICHE - {name}")
    print(f"{'='*60}")

    # Carica tutti i dati
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    all_features = []
    all_labels = []

    print(f"Caricamento dati in corso...")
    for i, (features, label) in enumerate(loader):
        if i % 50 == 0:
            print(f"  Caricato {i}/{len(loader)} campioni...")
        all_features.append(features.numpy())
        all_labels.append(label.numpy())

    # Concatena tutti i dati
    all_features = np.concatenate(
        all_features, axis=0
    )  # Shape: [N, seq_len, num_features]
    all_labels = np.concatenate(all_labels, axis=0)

    print(f"\nDimensioni dati:")
    print(f"  Features shape: {all_features.shape}")
    print(f"  Labels shape: {all_labels.shape}")

    # Statistiche per etichetta
    unique_labels, label_counts = np.unique(all_labels, return_counts=True)
    print(f"\nDistribuzione etichette:")
    for label, count in zip(unique_labels, label_counts):
        percentage = count / len(all_labels) * 100
        label_name = (
            dataset.labels[label] if hasattr(dataset, "labels") else f"Class_{label}"
        )
        print(f"  {label_name}: {count} ({percentage:.1f}%)")

    # Statistiche dei features per sequenza (media su tutti i frame)
    # Media su tutti i frame per ogni campione
    features_per_sample = all_features.mean(axis=1)  # Shape: [N, num_features]

    print(f"\nStatistiche features (media per campione):")
    print(f"  Media globale: {features_per_sample.mean():.6f}")
    print(f"  Deviazione standard globale: {features_per_sample.std():.6f}")
    print(f"  Minimo: {features_per_sample.min():.6f}")
    print(f"  Massimo: {features_per_sample.max():.6f}")
    print(f"  Mediana: {np.median(features_per_sample):.6f}")

    # Statistiche per classe
    print(f"\nStatistiche per classe:")
    for label in unique_labels:
        mask = all_labels == label
        class_features = features_per_sample[mask]
        label_name = (
            dataset.labels[label] if hasattr(dataset, "labels") else f"Class_{label}"
        )

        print(f"  {label_name}:")
        print(f"    Media: {class_features.mean():.6f}")
        print(f"    Std: {class_features.std():.6f}")
        print(f"    Range: [{class_features.min():.6f}, {class_features.max():.6f}]")

    # Statistiche per feature (per identificare feature problematiche)
    print(f"\nAnalisi per feature (prime 10):")
    for i in range(min(10, features_per_sample.shape[1])):
        feature_vals = features_per_sample[:, i]
        print(
            f"  Feature {i}: media={feature_vals.mean():.4f}, std={feature_vals.std():.4f}, range=[{feature_vals.min():.4f}, {feature_vals.max():.4f}]"
        )

    # Controlla se ci sono valori NaN o infiniti
    nan_count = np.isnan(features_per_sample).sum()
    inf_count = np.isinf(features_per_sample).sum()
    zero_count = (features_per_sample == 0).sum()

    print(f"\nControllo integrità dati:")
    print(f"  Valori NaN: {nan_count}")
    print(f"  Valori infiniti: {inf_count}")
    print(
        f"  Valori zero: {zero_count} ({zero_count/features_per_sample.size*100:.2f}%)"
    )

    return {
        "features": features_per_sample,
        "labels": all_labels,
        "stats": {
            "mean": features_per_sample.mean(),
            "std": features_per_sample.std(),
            "min": features_per_sample.min(),
            "max": features_per_sample.max(),
            "median": np.median(features_per_sample),
            "nan_count": nan_count,
            "inf_count": inf_count,
            "zero_count": zero_count,
        },
    }


def compare_distributions(train_data, golden_data):
    """
    Confronta le distribuzioni tra training e golden labels
    """
    print(f"\n{'='*60}")
    print(f"CONFRONTO DISTRIBUZIONI")
    print(f"{'='*60}")

    train_features = train_data["features"]
    golden_features = golden_data["features"]

    print(f"\nConfronto statistiche globali:")
    print(f"{'Metrica':<20} {'Training':<15} {'Golden':<15} {'Differenza':<15}")
    print(f"{'-'*65}")

    metrics = ["mean", "std", "min", "max", "median"]
    for metric in metrics:
        train_val = train_data["stats"][metric]
        golden_val = golden_data["stats"][metric]
        diff = abs(train_val - golden_val)
        diff_pct = diff / abs(train_val) * 100 if train_val != 0 else float("inf")

        print(
            f"{metric:<20} {train_val:<15.6f} {golden_val:<15.6f} {diff:.6f} ({diff_pct:.1f}%)"
        )

    # Test di normalità (Kolmogorov-Smirnov)
    from scipy.stats import ks_2samp

    print(f"\nTest di Kolmogorov-Smirnov (similarità distribuzioni):")
    # Test su features mediati per campione
    train_mean_features = train_features.mean(axis=1)
    golden_mean_features = golden_features.mean(axis=1)

    ks_stat, p_value = ks_2samp(train_mean_features, golden_mean_features)
    print(f"  KS statistic: {ks_stat:.6f}")
    print(f"  p-value: {p_value:.6f}")

    if p_value < 0.05:
        print(
            f"  ❌ PROBLEMA: Le distribuzioni sono significativamente diverse (p < 0.05)"
        )
        print(f"     Questo potrebbe spiegare le poor performance del modello!")
    else:
        print(f"  ✅ Le distribuzioni sono simili (p >= 0.05)")

    # Analisi per feature individuale
    print(f"\nAnalisi differenze per feature (prime 10):")
    print(
        f"{'Feature':<10} {'Train Mean':<12} {'Golden Mean':<12} {'Diff %':<10} {'KS p-val':<10}"
    )
    print(f"{'-'*60}")

    for i in range(min(10, train_features.shape[1])):
        train_feat = train_features[:, i]
        golden_feat = golden_features[:, i]

        train_mean = train_feat.mean()
        golden_mean = golden_feat.mean()
        diff_pct = (
            abs(train_mean - golden_mean) / abs(train_mean) * 100
            if train_mean != 0
            else 0
        )

        # Test KS per questa feature
        _, p_val = ks_2samp(train_feat, golden_feat)

        status = "❌" if p_val < 0.05 else "✅"
        print(
            f"{i:<10} {train_mean:<12.4f} {golden_mean:<12.4f} {diff_pct:<10.1f} {p_val:<10.4f} {status}"
        )


def suggest_solutions(train_data, golden_data):
    """
    Suggerisce possibili soluzioni basate sull'analisi
    """
    print(f"\n{'='*60}")
    print(f"SUGGERIMENTI PER MIGLIORARE LE PERFORMANCE")
    print(f"{'='*60}")

    train_stats = train_data["stats"]
    golden_stats = golden_data["stats"]

    # Calcola differenze percentuali
    mean_diff = (
        abs(train_stats["mean"] - golden_stats["mean"]) / abs(train_stats["mean"]) * 100
    )
    std_diff = (
        abs(train_stats["std"] - golden_stats["std"]) / abs(train_stats["std"]) * 100
    )

    print(f"\n1. NORMALIZZAZIONE:")
    if mean_diff > 10 or std_diff > 20:
        print(f"   ❌ PROBLEMA: Grande differenza nelle statistiche")
        print(f"      - Differenza media: {mean_diff:.1f}%")
        print(f"      - Differenza std: {std_diff:.1f}%")
        print(
            f"   💡 SOLUZIONE: Applicare normalizzazione ai golden labels usando statistiche di training"
        )
        print(f"      train_mean = {train_stats['mean']:.6f}")
        print(f"      train_std = {train_stats['std']:.6f}")
        print(
            f"      golden_normalized = (golden - golden_mean) / golden_std * train_std + train_mean"
        )
    else:
        print(f"   ✅ Le statistiche sono simili")

    print(f"\n2. DOMAIN ADAPTATION:")
    print(f"   💡 Considera fine-tuning del modello sui golden labels")
    print(f"   💡 Oppure training con domain adaptation techniques")

    print(f"\n3. THRESHOLD TUNING:")
    print(
        f"   💡 Invece di usare soglia 0.5, trova la soglia ottimale sui golden labels"
    )
    print(f"   💡 Usa cross-validation per trovare la migliore soglia")

    print(f"\n4. ENSEMBLE METHODS:")
    print(f"   💡 Combina predizioni di più modelli")
    print(f"   💡 Usa voting o averaging")


def main():
    """
    Funzione principale
    """
    print("Inizializzazione analisi comparativa...")

    # Carica dati di training
    print("\nCaricamento dati di training...")
    (
        train_landmarks_dirs,
        train_processed_files,
        val_landmarks_dir,
        val_processed_file,
    ) = get_data_paths(BASE_DIR)

    train_dataset, val_dataset = get_datasets(
        train_landmarks_dirs,
        train_processed_files,
        val_landmarks_dir,
        val_processed_file,
        downsample_majority_class=False,  # Non fare downsampling per l'analisi
        downsample_ratio=0.0,
    )

    # Carica dati golden labels
    print("\nCaricamento dati golden labels...")
    landmarks_dir = os.path.join(
        BASE_DIR, "data", "raw", "ASLLRP", "mediapipe_output_golden_label", "json"
    )
    processed_file = os.path.join(
        BASE_DIR, "data", "processed", "golden_label_sentiment.csv"
    )

    golden_dataset = GoldenLabelDataset(landmarks_dir, processed_file)

    # Analizza training dataset
    print("\n" + "=" * 80)
    print("FASE 1: ANALISI DATI DI TRAINING")
    print("=" * 80)
    train_data = compute_dataset_statistics(train_dataset, "TRAINING DATASET")

    # Analizza golden labels dataset
    print("\n" + "=" * 80)
    print("FASE 2: ANALISI GOLDEN LABELS")
    print("=" * 80)
    golden_data = compute_dataset_statistics(golden_dataset, "GOLDEN LABELS")

    # Confronta le distribuzioni
    print("\n" + "=" * 80)
    print("FASE 3: CONFRONTO E DIAGNOSI")
    print("=" * 80)
    compare_distributions(train_data, golden_data)

    # Suggerisci soluzioni
    suggest_solutions(train_data, golden_data)

    print(f"\n{'='*80}")
    print(f"ANALISI COMPLETATA")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
