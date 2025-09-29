# -*- coding: utf-8 -*-
"""
Modulo di utilità per l'addestramento dei modelli.

Questo file centralizza le funzioni comuni utilizzate sia per l'addestramento stand        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = prepare_batch_fn((xb, yb), device, non_blocking=False)
                # Cast esplicito a float32 per compatibilità con MPS
                outputs = model(xb.to(torch.float32))
                preds = torch.argmax(outputs, dim=1)
                y_true.extend(yb.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())un_train.py`) sia per l'ottimizzazione degli iperparametri (`hyperparameter_tuning.py`).
L'obiettivo è ridurre la duplicazione del codice e migliorare la manutenibilità.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score

from ignite.engine import Events, create_supervised_evaluator
from ignite.metrics import Loss, Accuracy

from src.data_pipeline.landmark_dataset import LandmarkDataset
from src.models.vivit.video_dataset import VideoDataset  # Aggiunto import
from src.models.lstm_model import EmotionLSTM
from src.models.stgcn_model import STGCN


def get_data_paths(base_dir):
    """Restituisce i percorsi standard per i dati di training e validazione."""
    train_landmarks_dirs = [
        os.path.join(base_dir, "data", "raw", "train", "openpose_output_train", "json"),
        os.path.join(base_dir, "data", "raw", "ASLLRP", "mediapipe_output_34", "json"),
    ]
    train_processed_files = [
        os.path.join(
            base_dir, "data", "processed", "train", "video_sentiment_data_0.34.csv"
        ),
        os.path.join(
            base_dir,
            "data",
            "processed",
            "asllrp_video_sentiment_data_0.34_without_golden.csv",
        ),
    ]
    val_landmarks_dir = os.path.join(
        base_dir, "data", "raw", "val", "openpose_output_val", "json"
    )
    val_processed_file = os.path.join(
        base_dir, "data", "processed", "val", "video_sentiment_data_0.34.csv"
    )
    return (
        train_landmarks_dirs,
        train_processed_files,
        val_landmarks_dir,
        val_processed_file,
    )


def get_datasets(
    train_landmarks_dirs,
    train_processed_file,
    val_landmarks_dir,
    val_processed_file,
    downsample_majority_class=False,
    majority_class_label="Positive",
    downsample_ratio=1.0,
):
    """
    Carica e restituisce i dataset di training e validazione.

    Args:
        train_landmarks_dirs (list): Percorsi alle directory dei landmark di training.
        train_processed_file (str): Percorso al file CSV processato di training.
        val_landmarks_dir (str): Percorso alla directory dei landmark di validazione.
        val_processed_file (str): Percorso al file CSV processato di validazione.
        downsample_majority_class (bool): Se True, esegue il downsampling della classe maggioritaria.
        majority_class_label (str): L'etichetta della classe maggioritaria da sottocampionare.
        downsample_ratio (float): Il rapporto a cui ridurre la classe maggioritaria.
                                 1.0 significa che avrà la stessa dimensione della minoritaria.
    """
    train_dataset = LandmarkDataset(
        landmarks_dir=train_landmarks_dirs, processed_file=train_processed_file
    )
    val_dataset = LandmarkDataset(
        landmarks_dir=val_landmarks_dir, processed_file=val_processed_file
    )

    if downsample_majority_class:
        print("Esecuzione del downsampling sulla classe maggioritaria...")
        # Downsample del training set
        df_train = train_dataset.processed
        class_counts_train = df_train["emotion"].value_counts()
        minority_class_label_train = class_counts_train.idxmin()
        minority_count_train = class_counts_train.min()

        n_majority_new = int(minority_count_train * downsample_ratio)

        df_majority_train = df_train[df_train["emotion"] == majority_class_label]
        df_minority_train = df_train[df_train["emotion"] == minority_class_label_train]

        df_majority_downsampled_train = df_majority_train.sample(
            n=n_majority_new, random_state=42
        )
        df_train_downsampled = pd.concat(
            [df_majority_downsampled_train, df_minority_train]
        )
        train_dataset.processed = df_train_downsampled.sample(
            frac=1, random_state=42
        ).reset_index(drop=True)

        print(f"Train set originale: {class_counts_train.to_dict()}")
        print(
            "Train set dopo downsampling:"
            f" {train_dataset.processed['emotion'].value_counts().to_dict()}"
        )

    return train_dataset, val_dataset


def get_sampler(dataset):
    """Crea un WeightedRandomSampler per bilanciare le classi."""
    class_weights = get_class_weights(dataset, device="cpu")
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights.tolist())}

    if hasattr(dataset, "video_info"):  # Caso per VideoDataset
        sample_weights = [
            class_weights_dict[dataset.label2id[label]]
            for label in dataset.video_info["emotion"]
        ]
    else:  # Caso per LandmarkDataset
        labels_arr = dataset.processed["emotion"].map(dataset.label_map).values
        sample_weights = [class_weights_dict[label] for label in labels_arr]

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )
    return sampler


def get_dataloaders(
    train_landmarks_dirs,
    train_processed_file,
    val_landmarks_dir,
    val_processed_file,
    batch_size,
    num_workers=0,
    use_sampler=False,
):
    """
    Crea e restituisce i DataLoader per training e validazione.
    """
    train_dataset, val_dataset = get_datasets(
        train_landmarks_dirs,
        train_processed_file,
        val_landmarks_dir,
        val_processed_file,
    )

    train_sampler = None
    shuffle = True
    if use_sampler:
        train_sampler = get_sampler(train_dataset)
        shuffle = False  # Sampler e shuffle sono mutuamente esclusivi

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, train_dataset


def get_class_weights(dataset, device):
    """Calcola i pesi per la funzione di loss per contrastare lo sbilanciamento delle classi."""
    if hasattr(dataset, "video_info"):  # Caso per VideoDataset
        labels_arr = [
            dataset.label2id[label] for label in dataset.video_info["emotion"]
        ]
        class_counts = np.bincount(labels_arr)
        total_samples = len(dataset.video_info)
    else:  # Caso per LandmarkDataset
        labels_arr = dataset.processed["emotion"].map(dataset.label_map).values
        class_counts = np.bincount(labels_arr)
        total_samples = len(labels_arr)

    num_classes = len(class_counts)

    # Esegui i calcoli in float32 per evitare problemi di cast su MPS
    class_weights = torch.tensor(
        [np.float32(total_samples / (num_classes * count)) for count in class_counts],
        dtype=torch.float32,
    )
    return class_weights.to(device)


def create_model(model_type, input_size, num_classes, device, params):
    """Crea e restituisce il modello specificato (LSTM o ST-GCN) con i parametri dati."""
    if model_type == "lstm":
        model = EmotionLSTM(
            input_size,
            params["hidden_size"],
            params["num_layers"],
            num_classes,
            dropout=params["dropout"],
        ).to(device)
    elif model_type == "stgcn":
        coords_per_point = 2
        num_point = input_size // coords_per_point
        model = STGCN(
            num_classes,
            num_point,
            num_person=1,
            in_channels=coords_per_point,
            dropout_rate=params["dropout"],
        ).to(device)
    else:
        raise ValueError(f"Tipo di modello non supportato: {model_type}")
    return model


def prepare_batch(batch, device, model_type, non_blocking=False):
    """Prepara un batch di dati, rimodellandolo se necessario per ST-GCN."""
    x, y = batch
    if model_type == "stgcn":
        b, t, f = x.shape
        coords_per_point = 2
        num_point = f // coords_per_point
        x = x.view(b, t, num_point, coords_per_point)
    return (
        x.to(device, non_blocking=non_blocking),
        y.to(device, non_blocking=non_blocking),
    )


def setup_ignite_evaluator(model, criterion, device, model_type, val_loader):
    """
    Configura e restituisce un evaluator di Ignite con le metriche necessarie (Loss, Accuracy, F1).
    """
    prepare_batch_fn = lambda batch, device, non_blocking: prepare_batch(
        batch, device, model_type, non_blocking
    )

    val_metrics = {
        "val_loss": Loss(criterion),
        "val_acc": Accuracy(),
    }
    evaluator = create_supervised_evaluator(
        model, metrics=val_metrics, device=device, prepare_batch=prepare_batch_fn
    )

    @evaluator.on(Events.COMPLETED)
    def compute_f1_scores(engine):
        y_true, y_pred = [], []
        model.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = prepare_batch_fn((xb, yb), device, non_blocking=False)
                outputs = model(xb.to(torch.float32))
                preds = torch.argmax(outputs, dim=1)
                y_true.extend(yb.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

        engine.state.metrics["val_f1_macro"] = f1_macro
        engine.state.metrics["val_f1"] = f1_weighted
        engine.state.metrics["val_f1_class_0"] = float(f1_per_class[0])
        engine.state.metrics["val_f1_class_1"] = (
            float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0
        )

    return evaluator
