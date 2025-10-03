# =================================================================================================
# COMANDI UTILI PER L'ESECUZIONE
# =================================================================================================
#
# 1. AVVIARE IL SERVER MLFLOW
#    Per monitorare gli esperimenti, prima di eseguire lo script di tuning,
#    aprire un terminale e lanciare il server MLflow:
#
#    mlflow server --host 127.0.0.1 --port 8080
#
# 2. ESEGUIRE L'OTTIMIZZAZIONE DEGLI IPERPARAMETRI PER VIVIT
#    Questo script utilizza Optuna per trovare i migliori iperparametri per il modello ViViT.
#
#    ESEMPIO DI COMANDO:
#    Esegue 40 tentativi di ottimizzazione, addestrando per 20 epoche ogni volta.
#
#    python src/models/vivit/hyperparameter_tuning_vivit.py --n_trials 40 --num_epochs 20 --downsample_ratio 1.0
#
#    python src/models/vivit/hyperparameter_tuning_vivit.py --n_trials 20 --num_epochs 7 --downsample_ratio 1.0 --num_workers 2
#
# =================================================================================================

import os
import sys
import argparse
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
import mlflow
import mlflow.pytorch
from optuna.pruners import MedianPruner
import random
from torch.backends import cudnn
from optuna.samplers import TPESampler
from collections import Counter
import gc

# Ignite imports
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping as IgniteEarlyStopping
from ignite.handlers import ModelCheckpoint
from optuna.integration import PyTorchIgnitePruningHandler
from optuna.exceptions import TrialPruned
from ignite.metrics import Loss, Accuracy, Fbeta, Precision, Recall
from ignite.contrib.handlers import ProgressBar

# --- Sezione 1: Setup del Percorso di Base e Import delle Utilità ---
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
    ),
)
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)

from src.models.vivit.vivit_model import create_vivit_model
from src.models.vivit.video_dataset import VideoDataset
from src.utils.training_utils import (
    get_class_weights,
    get_sampler,
    get_datasets,
    get_data_paths,
    prepare_batch,
    get_video_datasets,
)  # Aggiunto import

# --- Sezione 2: Setup di MLflow ---
mlflow.set_tracking_uri("http://127.0.0.1:8080")
experiment_name = "ViViT - Hyperparameter Tuning VADE 0.34"
if mlflow.get_experiment_by_name(experiment_name) is None:
    mlflow.create_experiment(experiment_name)
else:
    mlflow.set_experiment(experiment_name)

# --- Sezione 3: Definizione dei Percorsi Dati ---
(
    TRAIN_LANDMARKS_DIR,
    TRAIN_PROCESSED_FILE,
    VAL_LANDMARKS_DIR,
    VAL_PROCESSED_FILE,
) = get_data_paths(BASE_DIR)

# Directory video per training (include ASLLRP per coerenza con hyperparameter_tuning.py)
TRAIN_VIDEO_DIRS = [
    os.path.join(BASE_DIR, "data", "raw", "train", "raw_videos_front_train"),
    os.path.join(BASE_DIR, "data", "raw", "ASLLRP", "batch_utterance_video_v3_1"),
]
VAL_VIDEO_DIR = os.path.join(BASE_DIR, "data", "raw", "val", "raw_videos_front_val")

# File annotazioni per training (include ASLLRP per coerenza con hyperparameter_tuning.py)
TRAIN_ANNOTATIONS_FILES = [
    os.path.join(
        BASE_DIR, "data", "processed", "train", "video_sentiment_data_0.34.csv"
    ),
    os.path.join(
        BASE_DIR,
        "data",
        "processed",
        "asllrp_video_sentiment_data_0.34_without_golden.csv",
    ),
]
VAL_ANNOTATIONS_FILE = os.path.join(
    BASE_DIR, "data", "processed", "val", "video_sentiment_data_0.34.csv"
)

# --- Sezione 4: Variabili Globali ---
NUM_EPOCHS = 10
PATIENCE = 5
MODEL_NAME = "google/vivit-b-16x2-kinetics400"
NUM_WORKERS = 2  # Aumentato da 0 per parallelizzare il caricamento


# --- Funzioni di Utilità per Dati e Modello ---
# Le funzioni prepare_batch, get_sampler, get_class_weights sono ora in training_utils.py


def objective(trial, train_dataset, val_dataset):
    """
    Funzione "obiettivo" di Optuna per un singolo trial di ottimizzazione.
    """
    # --- Sezione 5: Definizione dello Spazio di Ricerca degli Iperparametri ---
    learning_rate = trial.suggest_float(
        "learning_rate", 1e-5, 1e-3, log=True
    )  # Range più ampio
    batch_size = trial.suggest_categorical(
        "batch_size", [1, 2]
    )  # Solo 1 e 2 per memoria
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)

    # --- Sezione 6: Caricamento Dati ---
    train_sampler = get_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=False,  # Disabilitato per risparmiare memoria
        prefetch_factor=1,  # Ridotto per memoria
        persistent_workers=True,  # Mantiene worker attivi per efficienza
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        prefetch_factor=1,
        persistent_workers=True,
    )

    # --- Sezione 7: Setup del Modello e dell'Addestramento ---
    num_classes = len(train_dataset.labels)
    model, _ = create_vivit_model(num_classes, model_name=MODEL_NAME)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model.to(device)

    class_weights_tensor = get_class_weights(train_dataset, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # --- Ignite Trainer ed Evaluator con ottimizzazioni memoria ---
    def train_step(engine, batch):
        model.train()
        optimizer.zero_grad()

        # Gradient accumulation per simulare batch più grandi
        pixel_values, labels = prepare_batch(batch, device, "vivit")
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss

        # Normalizza loss per gradient accumulation se batch_size è 1
        if batch_size == 1:
            loss = loss / 2  # Simula batch_size 2

        loss.backward()

        # Accumula gradienti ogni 2 step se batch_size è 1
        if batch_size == 1 and engine.state.iteration % 2 == 0:
            optimizer.step()
            optimizer.zero_grad()
        elif batch_size > 1:
            optimizer.step()

        return loss.item()

    trainer = Engine(train_step)

    # Progress bar meno verbosa
    pbar = ProgressBar(persist=False)  # Non persistente per memoria
    pbar.attach(trainer, output_transform=lambda x: {"loss": f"{x:.4f}"})

    def eval_step(engine, batch):
        model.eval()
        with torch.no_grad():
            pixel_values, labels = prepare_batch(batch, device, "vivit")
            # Libera memoria immediatamente dopo ogni batch
            outputs = model(pixel_values=pixel_values)
            del pixel_values  # Explicit cleanup
            return outputs, labels

    def output_transform(output):
        y_pred, y = output
        return y_pred.logits, y

    evaluator = Engine(eval_step)
    val_metrics = {
        "accuracy": Accuracy(output_transform=output_transform, device=device),
        "loss": Loss(criterion, output_transform=output_transform, device=device),
        "f1_macro": Fbeta(
            1.0, average="macro", output_transform=output_transform, device=device
        ),
        "precision_macro": Precision(
            average="macro", output_transform=output_transform, device=device
        ),
        "recall_macro": Recall(
            average="macro", output_transform=output_transform, device=device
        ),
    }
    for name, metric in val_metrics.items():
        metric.attach(evaluator, name)

    # --- Sezione 8: Esecuzione del Trial e Logging ---
    with mlflow.start_run(nested=True):
        mlflow.log_params(
            {
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "weight_decay": weight_decay,
                "model_name": MODEL_NAME,
            }
        )

        best_f1_macro = 0.0

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            nonlocal best_f1_macro
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            val_f1_macro = metrics["f1_macro"]
            val_loss = metrics["loss"]
            val_accuracy = metrics["accuracy"]
            val_precision = metrics["precision_macro"]
            val_recall = metrics["recall_macro"]

            if val_f1_macro > best_f1_macro:
                best_f1_macro = val_f1_macro

            mlflow.log_metrics(
                {
                    "val_loss": val_loss,
                    "val_f1_macro": val_f1_macro,
                    "val_accuracy": val_accuracy,
                    "val_precision_macro": val_precision,
                    "val_recall_macro": val_recall,
                },
                step=engine.state.epoch,
            )

            trial.report(val_f1_macro, engine.state.epoch)
            if trial.should_prune():
                raise TrialPruned()

            # Pulizia memoria più aggressiva
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()

        # --- Sezione 9: Pruning e Early Stopping ---
        early_stopper = IgniteEarlyStopping(
            patience=PATIENCE,
            score_function=lambda eng: eng.state.metrics["f1_macro"],
            trainer=trainer,
        )
        evaluator.add_event_handler(Events.COMPLETED, early_stopper)

        try:
            trainer.run(train_loader, max_epochs=NUM_EPOCHS)
        except TrialPruned:
            mlflow.set_tag("status", "pruned")
            raise
        except RuntimeError as e:
            if "out of memory" in str(e):
                mlflow.set_tag("status", "oom")
                # Segnala a Optuna di potare il trial, ma non come un fallimento
                # Restituiamo un valore molto basso per indicare una cattiva performance
                return 0.0
            else:
                raise e

        mlflow.log_metric("best_val_f1_macro", best_f1_macro)

        # Pulizia finale più aggressiva
        del model, optimizer, class_weights_tensor, criterion
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

        return best_f1_macro


def main():
    """
    Funzione principale che orchestra l'intero processo di ottimizzazione.
    """
    # --- Sezione 10: Parsing degli Argomenti e Setup dello Studio ---
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for ViViT with Optuna and MLflow"
    )
    parser.add_argument(
        "--n_trials", type=int, default=10, help="Number of Optuna trials"
    )
    parser.add_argument("--num_epochs", type=int, default=5, help="Epochs per trial")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/vivit-b-16x2-kinetics400",
        help="Hugging Face model name",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of worker processes for data loading.",
    )
    parser.add_argument(
        "--downsample_ratio",
        type=float,
        default=0.0,
        help="Ratio to downsample the majority class. 1.0 means balance with minority. 0.0 to disable.",
    )

    args = parser.parse_args()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True) # This can cause issues with some operations

    global NUM_EPOCHS, MODEL_NAME, NUM_WORKERS
    NUM_EPOCHS = args.num_epochs
    MODEL_NAME = args.model_name
    NUM_WORKERS = args.num_workers

    # --- Caricamento Dati (una sola volta) ---
    print("Caricamento e pre-processing dei dataset...")
    # Creiamo il modello una volta per ottenere la configurazione corretta
    temp_model, image_processor = create_vivit_model(
        num_classes=2, model_name=MODEL_NAME
    )  # num_classes è temporaneo
    num_frames = temp_model.config.num_frames
    del temp_model  # Liberiamo la memoria

    # Carica i dataset video per ViViT usando la funzione condivisa con supporto completo
    # Ora include TUTTI i dataset (train + ASLLRP) per coerenza con hyperparameter_tuning.py
    train_dataset, val_dataset = get_video_datasets(
        train_video_dirs=TRAIN_VIDEO_DIRS,  # Tutte le directory
        train_annotations_files=TRAIN_ANNOTATIONS_FILES,  # Tutti i file annotazioni
        val_video_dir=VAL_VIDEO_DIR,
        val_annotations_file=VAL_ANNOTATIONS_FILE,
        image_processor=image_processor,
        num_frames=num_frames,
        downsample_majority_class=args.downsample_ratio > 0,
        downsample_ratio=args.downsample_ratio,
    )

    print(
        f"   Directory video: {len(TRAIN_VIDEO_DIRS)} ({[os.path.basename(d) for d in TRAIN_VIDEO_DIRS]})"
    )
    print(
        f"   File annotazioni: {len(TRAIN_ANNOTATIONS_FILES)} ({[os.path.basename(f) for f in TRAIN_ANNOTATIONS_FILES]})"
    )

    print("Dataset caricati.")

    # Stampa la distribuzione delle etichette con i nomi delle classi
    if hasattr(train_dataset, "video_info"):
        train_dist = Counter(train_dataset.video_info["emotion"])
        val_dist = Counter(val_dataset.video_info["emotion"])
    else:
        train_dist = Counter(train_dataset.labels)
        val_dist = Counter(val_dataset.labels)
    print("Train label distribution:", train_dist)
    print("Val label distribution:", val_dist)

    # --- Setup di MLflow dinamico ---
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    base_experiment_name = "VIVIT - VADER 0.34 - HT Emotion Recognition"
    if args.downsample_ratio > 0:
        experiment_name = (
            f"{base_experiment_name} - Downsampled {args.downsample_ratio}"
        )
    else:
        experiment_name = f"{base_experiment_name} - Full"

    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    model_short_name = MODEL_NAME.split("/")[-1]
    run_name = f"Optuna_ViViT_{model_short_name}_tuning"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("seed", seed)
        study = optuna.create_study(
            direction="maximize",
            pruner=MedianPruner(n_startup_trials=PATIENCE, n_warmup_steps=1),
            sampler=TPESampler(seed=seed),
        )
        # Passiamo i dataset caricati alla funzione objective
        study.optimize(
            lambda trial: objective(trial, train_dataset, val_dataset),
            n_trials=args.n_trials,
        )

        mlflow.log_param("n_trials", args.n_trials)
        mlflow.log_params(study.best_trial.params)
        mlflow.log_metric("best_val_f1_macro_study", study.best_value)
        mlflow.set_tags(
            {
                "project": "EmoSign",
                "optimizer_engine": "optuna",
                "model_family": "ViViT",
            }
        )

    print("Best trial:", study.best_trial.params)
    print("Best value:", study.best_value)


if __name__ == "__main__":
    main()
