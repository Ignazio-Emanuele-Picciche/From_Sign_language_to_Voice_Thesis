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
#    Esegue 10 tentativi di ottimizzazione, addestrando per 10 epoche ogni volta.
#
#    python src/models/vivit/hyperparameter_tuning_vivit.py --n_trials 10 --num_epochs 10
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

# --- Sezione 2: Setup di MLflow ---
mlflow.set_tracking_uri("http://127.0.0.1:8080")
experiment_name = "ViViT - Hyperparameter Tuning VADE 0.34"
if mlflow.get_experiment_by_name(experiment_name) is None:
    mlflow.create_experiment(experiment_name)
else:
    mlflow.set_experiment(experiment_name)

# --- Sezione 3: Definizione dei Percorsi Dati ---
TRAIN_VIDEO_DIR = os.path.join(
    BASE_DIR, "data", "raw", "train", "raw_videos_front_train"
)
VAL_VIDEO_DIR = os.path.join(BASE_DIR, "data", "raw", "val", "raw_videos_front_val")
TRAIN_ANNOTATIONS_FILE = os.path.join(
    BASE_DIR, "data", "processed", "train", "video_sentiment_data_0.34.csv"
)
VAL_ANNOTATIONS_FILE = os.path.join(
    BASE_DIR, "data", "processed", "val", "video_sentiment_data_0.34.csv"
)

# --- Sezione 4: Variabili Globali ---
NUM_EPOCHS = 10
PATIENCE = 5
MODEL_NAME = "google/vivit-b-16x2-kinetics400"
NUM_WORKERS = 0


# --- Funzioni di Utilità per Dati e Modello ---


def get_class_weights(dataset):
    class_counts = Counter(dataset.video_info["emotion"])
    class_weights = {
        dataset.label2id[cls]: 1.0 / count for cls, count in class_counts.items()
    }
    weights = [class_weights[i] for i in sorted(class_weights.keys())]
    return torch.tensor(weights, dtype=torch.float)


def get_sampler(dataset):
    class_weights = get_class_weights(dataset)
    sample_weights = [
        class_weights[dataset.label2id[label]]
        for label in dataset.video_info["emotion"]
    ]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )
    return sampler


def prepare_batch(batch, device=None, non_blocking=False):
    pixel_values = batch["pixel_values"].to(device, non_blocking=non_blocking)
    labels = batch["labels"].to(device, non_blocking=non_blocking)
    return pixel_values, labels


def objective(trial, train_dataset, val_dataset):
    """
    Funzione "obiettivo" di Optuna per un singolo trial di ottimizzazione.
    """
    # --- Sezione 5: Definizione dello Spazio di Ricerca degli Iperparametri ---
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [1, 2, 4])
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)

    # --- Sezione 6: Caricamento Dati ---
    # I dataset sono già caricati, creiamo solo i dataloader
    train_sampler = get_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
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

    class_weights_tensor = get_class_weights(train_dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # --- Ignite Trainer ed Evaluator ---
    def train_step(engine, batch):
        model.train()
        optimizer.zero_grad()
        pixel_values, labels = prepare_batch(batch, device=device)
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        return loss.item()

    trainer = Engine(train_step)

    # Aggiungiamo una progress bar per avere un feedback visivo
    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, output_transform=lambda x: {"batch loss": x})

    def eval_step(engine, batch):
        model.eval()
        with torch.no_grad():
            pixel_values, labels = prepare_batch(batch, device=device)
            outputs = model(pixel_values=pixel_values)
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
        default=2,
        help="Number of worker processes for data loading.",
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
    # (es. num_frames) e l'image_processor che servono al dataset
    temp_model, image_processor = create_vivit_model(
        num_classes=2, model_name=MODEL_NAME
    )  # num_classes è temporaneo
    num_frames = temp_model.config.num_frames
    del temp_model  # Liberiamo la memoria

    train_dataset = VideoDataset(
        TRAIN_ANNOTATIONS_FILE, TRAIN_VIDEO_DIR, image_processor, num_frames=num_frames
    )
    val_dataset = VideoDataset(
        VAL_ANNOTATIONS_FILE, VAL_VIDEO_DIR, image_processor, num_frames=num_frames
    )
    print("Dataset caricati.")

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
