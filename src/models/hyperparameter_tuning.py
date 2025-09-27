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
# 2. ESEGUIRE L'OTTIMIZZAZIONE DEGLI IPERPARAMETRI
#    Questo script utilizza Optuna per trovare i migliori iperparametri per un dato modello.
#    È possibile specificare il tipo di modello (lstm o stgcn), il numero di "trial" (tentativi)
#    e il numero di epoche per ogni trial.
#
#    ESEMPIO DI COMANDO PER LSTM:
#    Esegue 40 tentativi di ottimizzazione per il modello LSTM, addestrando per 40 epoche ogni volta.
#
#    python3 src/models/hyperparameter_tuning.py --model_type lstm --n_trials 40 --num_epochs 40
#
#    ESEMPIO DI COMANDO PER ST-GCN:
#    Esegue 20 tentativi di ottimizzazione per il modello ST-GCN, addestrando per 25 epoche ogni volta.
#
#    python3 src/models/hyperparameter_tuning.py --model_type stgcn --n_trials 20 --num_epochs 25
#
#    python src/models/hyperparameter_tuning.py --model_type lstm --n_trials 40 --num_epochs 40 --num_workers 0
# =================================================================================================

import os
import sys
import argparse
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import mlflow
import mlflow.pytorch
from optuna.pruners import MedianPruner
import random
from torch.backends import cudnn
from optuna.samplers import TPESampler

import gc
from torch.utils.data import WeightedRandomSampler
import logging

# Ignite imports
# os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping as IgniteEarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
from optuna.integration import PyTorchIgnitePruningHandler
from optuna.exceptions import TrialPruned
from ignite.metrics import Loss, Accuracy, Fbeta, Precision, Recall
from ignite.contrib.handlers import ProgressBar

# --- Sezione 1: Setup del Percorso di Base e Import delle Utilità ---
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
)
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)

from src.utils.training_utils import (
    get_data_paths,
    get_datasets,
    get_class_weights,
    create_model,
    prepare_batch,
    setup_ignite_evaluator,
    get_sampler,
)

# --- Sezione 2: Setup di MLflow ---
mlflow.set_tracking_uri("http://127.0.0.1:8080")
experiment_name = "VADER 0.34 - HT Emotion Recognition Experiment"
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

# --- Sezione 4: Variabili Globali ---
NUM_EPOCHS = 5
MODEL_TYPE = "lstm"
PATIENCE = 4
NUM_WORKERS = 0

torch.set_default_dtype(torch.float32)


def objective(trial, train_dataset, val_dataset):
    """
    Funzione "obiettivo" di Optuna per un singolo trial di ottimizzazione.
    """
    # --- Sezione 5: Definizione dello Spazio di Ricerca degli Iperparametri ---
    if MODEL_TYPE == "lstm":
        hidden_size = trial.suggest_int(
            "hidden_size", 256, 512, step=32
        )  # RIDOTTO: Limite superiore ridotto da 768 a 512
        num_layers = trial.suggest_int("num_layers", 1, 3)  # Riduciamo i layer
        dropout = trial.suggest_float(
            "dropout", 0.3, 0.7
        )  # Aumentiamo il range del dropout
        learning_rate = trial.suggest_float(
            "learning_rate", 1e-6, 1e-5, log=True
        )  # Range ridotto
        batch_size = trial.suggest_categorical(
            "batch_size", [32, 64, 96]
        )  # RIDOTTO: Rimossi i valori più alti
        weight_decay = trial.suggest_float(
            "weight_decay", 1e-3, 0.1, log=True
        )  # Partiamo da un valore più alto
        model_params = {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
        }
    else:  # stgcn
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
        model_params = {"dropout": dropout}

    # --- Sezione 6: Creazione dei Dataloader specifici per il trial ---
    train_sampler = get_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=torch.utils.data.BatchSampler(
            train_sampler, batch_size, drop_last=False
        ),
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
    device = torch.device("cpu")

    class_weights_tensor = get_class_weights(train_dataset, device)
    input_size = train_dataset[0][0].shape[1]
    num_classes = len(train_dataset.labels)

    model = create_model(MODEL_TYPE, input_size, num_classes, device, model_params)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    print(
        f"Trial {trial.number}: lr={learning_rate}, batch_size={batch_size}, weight_decay={weight_decay}, dropout={model_params.get('dropout', None)}"
    )
    print("Class weights:", class_weights_tensor.cpu().numpy())
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=5)

    # --- Ignite Trainer ed Evaluator ---
    def train_step(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device, MODEL_TYPE)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        # Assicura che il tipo di dato sia float32 per la compatibilità con MPS
        return torch.tensor(loss.item(), dtype=torch.float32)

    trainer = Engine(train_step)
    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, output_transform=lambda x: {"batch loss": x})

    def eval_step(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, device, MODEL_TYPE)
            y_pred = model(x)
            # Cast esplicito per garantire la compatibilità con le metriche su MPS
            return y_pred.to(dtype=torch.float32, device=device), y.to(
                dtype=torch.long, device=device
            )

    evaluator = Engine(eval_step)
    val_metrics = {
        "accuracy": Accuracy(),
        "loss": Loss(criterion),
        "f1_macro": Fbeta(1, average="macro"),
        "precision_macro": Precision(average="macro"),
        "recall_macro": Recall(average="macro"),
    }
    for name, metric in val_metrics.items():
        metric.attach(evaluator, name)

    # --- Sezione 8: Esecuzione del Trial e Logging ---
    with mlflow.start_run(nested=True):
        base_params = {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "model_type": MODEL_TYPE,
        }
        mlflow.log_params({**base_params, **model_params})

        best_f1_macro = 0.0

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            nonlocal best_f1_macro
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            val_f1_macro = metrics["f1_macro"]
            val_loss = metrics["loss"]

            if val_f1_macro > best_f1_macro:
                best_f1_macro = val_f1_macro

            mlflow.log_metrics(
                {
                    "val_loss": val_loss,
                    "val_f1_macro": val_f1_macro,
                    "val_accuracy": metrics["accuracy"],
                    "val_precision_macro": metrics["precision_macro"],
                    "val_recall_macro": metrics["recall_macro"],
                },
                step=engine.state.epoch,
            )

            scheduler.step(val_f1_macro)
            trial.report(val_f1_macro, engine.state.epoch)
            if trial.should_prune():
                raise TrialPruned()

            gc.collect()
            # if torch.backends.mps.is_available():
            #     torch.mps.empty_cache()

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
            # Questo blocco viene eseguito quando Optuna interrompe un trial non promettente.
            # È un comportamento atteso, non un errore.
            mlflow.set_tag("status", "pruned")
            # Rilanciamo l'eccezione per far sapere a Optuna di terminare il trial.
            raise TrialPruned(
                f"Trial pruned at epoch {trainer.state.epoch} because performance was not promising."
            )
        except Exception as e:
            # Cattura qualsiasi altra eccezione imprevista durante il training
            logging.error(
                f"Trial {trial.number} failed due to an unexpected error: {e}",
                exc_info=True,
            )
            mlflow.set_tag("status", "failed")
            # Rilancia l'eccezione per far fallire il trial in Optuna
            raise

        mlflow.log_metric("best_val_f1_macro", best_f1_macro)

        # --- Sezione 11: Pulizia della Memoria ---
        # !Versione attuale (commentata per riferimento)
        # # Rilascia esplicitamente la memoria occupata da modello, optimizer e dataloader
        # del (
        #     model,
        #     optimizer,
        #     criterion,
        #     trainer,
        #     evaluator,
        #     current_train_loader,
        #     current_val_loader,
        # )
        # gc.collect()
        # # if torch.backends.mps.is_available():
        # #     torch.mps.empty_cache()

        # !NUOVA VERSIONE: Pulizia aggressiva per prevenire memory leak
        # Rimuovi esplicitamente gli handler per prevenire memory leak
        trainer.remove_event_handler(log_validation_results, Events.EPOCH_COMPLETED)
        evaluator.remove_event_handler(early_stopper, Events.COMPLETED)
        pbar.attach(trainer)

        # !Versione attuale
        # Rilascia esplicitamente la memoria occupata da modello, optimizer e dataloader
        del (
            model,
            optimizer,
            criterion,
            trainer,
            evaluator,
            train_loader,
            val_loader,
        )
        gc.collect()
        # !NUOVA VERSIONE: Pulizia aggressiva per prevenire memory leak
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        return best_f1_macro


def main():
    """
    Funzione principale che orchestra l'intero processo di ottimizzazione.
    """
    # --- Sezione 10: Parsing degli Argomenti e Setup dello Studio ---
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning with Optuna and MLflow"
    )
    parser.add_argument(
        "--model_type", type=str, choices=["lstm", "stgcn"], default="lstm"
    )
    parser.add_argument(
        "--n_trials", type=int, default=20, help="Number of Optuna trials"
    )
    parser.add_argument("--num_epochs", type=int, default=5, help="Epochs per trial")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
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
    # torch.use_deterministic_algorithms(True) # Può causare problemi

    global NUM_EPOCHS, MODEL_TYPE, NUM_WORKERS
    NUM_EPOCHS = args.num_epochs
    MODEL_TYPE = args.model_type
    NUM_WORKERS = args.num_workers

    # --- Caricamento Dati (una sola volta) ---
    print("Caricamento dei dataset...")
    train_dataset, val_dataset = get_datasets(
        TRAIN_LANDMARKS_DIR, TRAIN_PROCESSED_FILE, VAL_LANDMARKS_DIR, VAL_PROCESSED_FILE
    )
    print("Dataset caricati.")

    # Stampa la distribuzione delle etichette con i nomi delle classi
    train_labels = train_dataset.processed["emotion"].map(train_dataset.label_map)
    val_labels = val_dataset.processed["emotion"].map(val_dataset.label_map)
    train_dist = {
        label: count
        for label, count in zip(
            train_dataset.labels,
            np.bincount(train_labels, minlength=len(train_dataset.labels)),
        )
    }
    val_dist = {
        label: count
        for label, count in zip(
            val_dataset.labels,
            np.bincount(val_labels, minlength=len(val_dataset.labels)),
        )
    }
    print("Train label distribution:", train_dist)
    print("Val label distribution:", val_dist)

    run_name = f"Optuna_{MODEL_TYPE}_tuning_optimized"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("seed", seed)
        study = optuna.create_study(
            direction="maximize",
            pruner=MedianPruner(n_startup_trials=PATIENCE, n_warmup_steps=1),
            sampler=TPESampler(seed=seed),
        )
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
                "model_family": MODEL_TYPE,
            }
        )

    print("Best trial:", study.best_trial.params)
    print("Best value:", study.best_value)


if __name__ == "__main__":
    main()
