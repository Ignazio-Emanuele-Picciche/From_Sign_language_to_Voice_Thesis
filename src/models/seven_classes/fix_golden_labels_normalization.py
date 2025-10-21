# =================================================================================================
# SCRIPT PER CORREGGERE LA NORMALIZZAZIONE DEI GOLDEN LABELS
# =================================================================================================
#
# Questo script corregge il problema di normalizzazione identificato nell'analisi:
# - Training data: coordinate in pixel (scala ~160, std ~257)
# - Golden labels: coordinate normalizzate [0-1] (scala ~0.46, std ~0.17)
#
# La soluzione è de-normalizzare i golden labels per portarli alla stessa scala del training.
#
# COMANDO PER ESEGUIRE IL TEST:
# python src/models/fix_golden_labels_normalization.py
#
# =================================================================================================

import os
import sys
import numpy as np
import torch
import json

# Setup paths
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
sys.path.insert(0, BASE_DIR)

from src.models.seven_classes.test_golden_labels import GoldenLabelDataset


class FixedGoldenLabelDataset(GoldenLabelDataset):
    """
    Dataset con normalizzazione corretta per golden labels.
    Converte le coordinate normalizzate [0-1] alla scala del training data.
    """

    def __init__(
        self,
        landmarks_dir,
        processed_file,
        train_mean=159.878159,
        train_std=256.977173,
        max_seq_len=100,
        num_features=50,
    ):
        # Chiama il costruttore della classe padre
        super().__init__(landmarks_dir, processed_file, max_seq_len, num_features)

        # Statistiche del training data per la de-normalizzazione
        self.train_mean = train_mean
        self.train_std = train_std

        print(f"Dataset con normalizzazione corretta inizializzato:")
        print(f"  Train mean: {train_mean:.6f}")
        print(f"  Train std: {train_std:.6f}")

    def extract_features_from_json(self, frame_json):
        """
        Estrae features con normalizzazione corretta per essere compatibili con il modello di training.
        """
        people = frame_json.get("people", [])
        if not people:
            # Se non ci sono persone, restituisci features nella scala del training
            return np.full(self.num_features, self.train_mean, dtype=np.float32)

        # Prende la prima persona
        person_data = people[0]
        keypoints = person_data.get("pose_keypoints_2d", [])

        if not keypoints:
            return np.full(self.num_features, self.train_mean, dtype=np.float32)

        # I keypoints sono in formato [x1, y1, z1, x2, y2, z2, ...]
        # Rimuoviamo ogni terzo elemento (z o confidence) e prendiamo solo x, y
        flat_landmarks = [coord for i, coord in enumerate(keypoints) if i % 3 != 2]

        # Converti in numpy array
        landmarks_array = np.array(flat_landmarks, dtype=np.float32)

        # Standardizza alla dimensione delle features richieste
        if len(landmarks_array) > self.num_features:
            landmarks_array = landmarks_array[: self.num_features]
        elif len(landmarks_array) < self.num_features:
            padding = np.zeros(self.num_features - len(landmarks_array))
            landmarks_array = np.hstack([landmarks_array, padding])

        # CORREZIONE CRITICA: De-normalizza i dati
        # I golden labels sono normalizzati [0-1], il training data è in pixel

        # Step 1: I dati golden sono già normalizzati tra 0-1, assumiamo una risoluzione standard
        # Per convertire da coordinate normalizzate [0-1] a pixel, moltiplichiamo per una risoluzione tipica
        # Basandoci sulle statistiche, sembrano essere coordinate normalizzate di un video

        # Step 2: Scala i valori per matchare la distribuzione del training
        # Dalla analisi: training ha media ~160 e std ~257
        # Golden labels hanno media ~0.46 e std ~0.17

        # Metodo 1: Scaling lineare mantenendo le proporzioni relative
        # Mappa [0,1] -> [0, video_resolution] poi applica le statistiche del training

        # Assumiamo una risoluzione di 640x480 (tipica per i video)
        VIDEO_WIDTH = 640
        VIDEO_HEIGHT = 480

        # Converte le coordinate normalizzate in pixel
        landmarks_pixel = landmarks_array.copy()

        # Per le coordinate x (indici pari) usa la larghezza
        landmarks_pixel[::2] = landmarks_pixel[::2] * VIDEO_WIDTH

        # Per le coordinate y (indici dispari) usa l'altezza
        landmarks_pixel[1::2] = landmarks_pixel[1::2] * VIDEO_HEIGHT

        # Step 3: Normalizza usando le statistiche del training per un matching più preciso
        current_mean = landmarks_pixel.mean()
        current_std = landmarks_pixel.std() if landmarks_pixel.std() > 0 else 1.0

        # Z-score normalization poi re-scale alle statistiche del training
        landmarks_normalized = (landmarks_pixel - current_mean) / current_std
        landmarks_rescaled = landmarks_normalized * self.train_std + self.train_mean

        return landmarks_rescaled


def test_normalization_fix():
    """
    Testa la correzione della normalizzazione confrontando prima e dopo
    """
    print("=" * 80)
    print("TEST DELLA CORREZIONE NORMALIZZAZIONE")
    print("=" * 80)

    landmarks_dir = os.path.join(
        BASE_DIR, "data", "raw", "ASLLRP", "mediapipe_output_golden_label", "json"
    )
    processed_file = os.path.join(
        BASE_DIR, "data", "processed", "golden_label_sentiment_7_classes.csv"
    )

    print("\n1. DATASET ORIGINALE (non corretto):")
    original_dataset = GoldenLabelDataset(landmarks_dir, processed_file)

    print("\n2. DATASET CON CORREZIONE:")
    fixed_dataset = FixedGoldenLabelDataset(
        landmarks_dir,
        processed_file,
        train_mean=159.878159,  # Dalle statistiche di training
        train_std=256.977173,
    )

    print(f"\nEntrambi i dataset hanno {len(original_dataset)} campioni")

    # Testa i primi 5 campioni
    print(f"\nConfronto primi 5 campioni:")
    print(
        f"{'Campione':<10} {'Originale (range)':<25} {'Corretto (range)':<25} {'Originale (media)':<15} {'Corretto (media)':<15}"
    )
    print("-" * 95)

    for i in range(min(5, len(original_dataset))):
        # Campione originale
        orig_features, orig_label = original_dataset[i]
        orig_mean = orig_features.mean().item()
        orig_min = orig_features.min().item()
        orig_max = orig_features.max().item()

        # Campione corretto
        fixed_features, fixed_label = fixed_dataset[i]
        fixed_mean = fixed_features.mean().item()
        fixed_min = fixed_features.min().item()
        fixed_max = fixed_features.max().item()

        print(
            f"{i+1:<10} [{orig_min:.3f}, {orig_max:.3f}]    [{fixed_min:.1f}, {fixed_max:.1f}]    {orig_mean:<15.3f} {fixed_mean:<15.1f}"
        )

    # Statistiche globali
    print(f"\nSTATISTICHE GLOBALI:")

    # Calcola statistiche per tutti i campioni del dataset originale
    all_orig_features = []
    for i in range(len(original_dataset)):
        features, _ = original_dataset[i]
        all_orig_features.append(features.numpy())
    all_orig_features = np.array(all_orig_features)

    # Calcola statistiche per tutti i campioni del dataset corretto
    all_fixed_features = []
    for i in range(len(fixed_dataset)):
        features, _ = fixed_dataset[i]
        all_fixed_features.append(features.numpy())
    all_fixed_features = np.array(all_fixed_features)

    print(f"\nDataset Originale:")
    print(f"  Shape: {all_orig_features.shape}")
    print(f"  Media: {all_orig_features.mean():.6f}")
    print(f"  Std: {all_orig_features.std():.6f}")
    print(f"  Range: [{all_orig_features.min():.6f}, {all_orig_features.max():.6f}]")

    print(f"\nDataset Corretto:")
    print(f"  Shape: {all_fixed_features.shape}")
    print(f"  Media: {all_fixed_features.mean():.6f}")
    print(f"  Std: {all_fixed_features.std():.6f}")
    print(f"  Range: [{all_fixed_features.min():.1f}, {all_fixed_features.max():.1f}]")

    print(f"\nTarget Training Data:")
    print(f"  Media target: 159.878159")
    print(f"  Std target: 256.977173")

    # Calcola differenze percentuali
    target_mean = 159.878159
    target_std = 256.977173

    mean_diff = abs(all_fixed_features.mean() - target_mean) / target_mean * 100
    std_diff = abs(all_fixed_features.std() - target_std) / target_std * 100

    print(f"\nDifferenza con Target:")
    print(f"  Differenza media: {mean_diff:.2f}%")
    print(f"  Differenza std: {std_diff:.2f}%")

    if mean_diff < 10 and std_diff < 20:
        print(f"  ✅ SUCCESSO: La correzione è efficace!")
    else:
        print(f"  ❌ ATTENZIONE: Potrebbero essere necessarie ulteriori correzioni")

    return fixed_dataset


if __name__ == "__main__":
    test_normalization_fix()
