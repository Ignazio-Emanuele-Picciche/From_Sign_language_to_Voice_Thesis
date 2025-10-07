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
#    Esegue 20 tentativi di ottimizzazione per il modello ST-GCN, addestrando per 34 epoche ogni volta.
#
#    python3 src/models/hyperparameter_tuning.py --model_type stgcn --n_trials 20 --num_epochs 34
#
#    python src/models/hyperparameter_tuning.py --model_type lstm --n_trials 40 --num_epochs 40 --num_workers 0
#
#    python src/models/hyperparameter_tuning.py --model_type lstm --n_trials 40 --num_epochs 20 --downsample_ratio 1.0

#    ESEMPIO DI COMANDO PER LSTM CON FOCAL LOSS E OTTIMIZZAZIONE SU WEIGHTED F1:
# python src/models/hyperparameter_tuning.py \
# --model_type lstm \
# --n_trials 50 \
# --num_epochs 20 \
# --downsample_ratio 1.0 \
# --use_focal_loss \
# --optimize_metric weighted_f1 \
# --normalize_data
# =================================================================================================

import os
import sys
import argparse
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
# Spostato all'interno della funzione main() per renderlo dinamico

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
PATIENCE = 10  # Aumentata la pazienza per dare più tempo al modello
NUM_WORKERS = 0
USE_FOCAL_LOSS = False  # Controllato da argomenti CLI
OPTIMIZE_METRIC = "f1_macro"  # Metrica di ottimizzazione
NORMALIZE_DATA = False  # Controllato da argomenti CLI
NORMALIZATION_TYPE = "minmax"  # Tipo di normalizzazione

torch.set_default_dtype(torch.float32)


# =================================================================================================
# NORMALIZED DATASET WRAPPER
# =================================================================================================
class NormalizedDatasetWrapper:
    """
    Wrapper per applicare normalizzazione ai dataset esistenti
    """

    def __init__(
        self, dataset, scaler=None, fit_scaler=True, normalization_type="minmax"
    ):
        self.dataset = dataset
        self.scaler = scaler
        self.fit_scaler = fit_scaler
        self.normalization_type = normalization_type

        # Crea lo scaler se non fornito
        if self.scaler is None:
            if normalization_type == "minmax":
                self.scaler = MinMaxScaler(feature_range=(0, 1))
            elif normalization_type == "standard":
                self.scaler = StandardScaler()
            else:
                raise ValueError(
                    f"Normalizzazione non supportata: {normalization_type}"
                )

        # Fitta lo scaler sui dati se richiesto (solo per hyperparameter tuning con campione piccolo)
        if self.fit_scaler:
            self._fit_scaler()

    def _fit_scaler(self):
        """Fitta lo scaler su un campione deterministico di dati del dataset"""
        all_features = []
        sample_size = min(
            100, len(self.dataset)
        )  # Stesso sample_size di run_train.py per consistenza

        print(f"  Fitting scaler su {sample_size} campioni sequenziali...")
        for i in range(sample_size):
            features, _ = self.dataset[i]
            features_flat = features.reshape(-1).numpy()
            all_features.extend(features_flat)

        all_features = np.array(all_features).reshape(-1, 1)
        self.scaler.fit(all_features)
        print(f"  Scaler fittato su {len(all_features)} features")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """Applica normalizzazione al campione"""
        features, label = self.dataset[idx]

        if self.scaler is not None:
            original_shape = features.shape
            features_flat = features.reshape(-1).numpy().reshape(-1, 1)
            features_normalized = self.scaler.transform(features_flat)
            features = torch.from_numpy(
                features_normalized.reshape(original_shape)
            ).float()

        return features, label

    # Proxy attributes per compatibilità
    @property
    def labels(self):
        return self.dataset.labels

    @property
    def label_map(self):
        return self.dataset.label_map

    @property
    def num_features(self):
        return self.dataset.num_features

    @property
    def processed(self):
        return self.dataset.processed


# =================================================================================================
# FOCAL LOSS IMPLEMENTATION
# =================================================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss per gestire dataset sbilanciati.

    Args:
        alpha (float): Peso per bilanciare le classi (default: 1.0)
        gamma (float): Parametro di focusing per ridurre il peso dei campioni facili (default: 2.0)
        weight (Tensor): Pesi delle classi (come in CrossEntropyLoss)
    """

    def __init__(self, alpha=1.0, gamma=2.0, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        # Calcola cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction="none")

        # Calcola pt = e^(-ce_loss)
        pt = torch.exp(-ce_loss)

        # Calcola focal loss: α * (1-pt)^γ * ce_loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()


def objective(trial, train_dataset, val_dataset):
    """
    Funzione "obiettivo" di Optuna per un singolo trial di ottimizzazione.
    """
    # --- Sezione 5: Definizione dello Spazio di Ricerca degli Iperparametri ---

    # Loss function selection
    if USE_FOCAL_LOSS:
        use_focal_loss = trial.suggest_categorical("use_focal_loss", [True, False])
        focal_alpha = (
            trial.suggest_float("focal_alpha", 0.5, 2.0) if use_focal_loss else 1.0
        )
        focal_gamma = (
            trial.suggest_float("focal_gamma", 0.5, 3.0) if use_focal_loss else 2.0
        )
    else:
        use_focal_loss = False
        focal_alpha = 1.0
        focal_gamma = 2.0

    if MODEL_TYPE == "lstm":
        hidden_size = trial.suggest_int(
            "hidden_size", 128, 384, step=32
        )  # Riduciamo la complessità del modello
        num_layers = trial.suggest_int(
            "num_layers", 1, 2
        )  # Preferiamo modelli meno profondi
        dropout = trial.suggest_float(
            "dropout", 0.5, 0.8
        )  # Aumentiamo il range del dropout in modo aggressivo
        learning_rate = trial.suggest_float(
            "learning_rate", 5e-6, 5e-5, log=True
        )  # Esploriamo learning rate più bassi
        # Adjust learning rate for Focal Loss
        if use_focal_loss:
            learning_rate = learning_rate * trial.suggest_float(
                "lr_focal_multiplier", 1.0, 2.5
            )
        batch_size = trial.suggest_categorical(
            "batch_size", [32, 64]
        )  # Batch size più piccoli possono aiutare la generalizzazione
        weight_decay = trial.suggest_float(
            "weight_decay", 5e-3, 0.5, log=True
        )  # Aumentiamo il range del weight decay
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

    # --- Sezione 6: Applicazione Normalizzazione (se abilitata) ---
    current_train_dataset = train_dataset
    current_val_dataset = val_dataset

    if NORMALIZE_DATA:
        # Per hyperparameter tuning, usiamo lo stesso scaler per tutti i trial per consistenza
        if not hasattr(train_dataset, "_global_scaler"):
            # Prima volta: crea e fitta lo scaler
            if NORMALIZATION_TYPE == "minmax":
                global_scaler = MinMaxScaler(feature_range=(0, 1))
            else:
                global_scaler = StandardScaler()

            # Fitta su un campione deterministico del training set
            print(
                f"Fitting scaler globale '{NORMALIZATION_TYPE}' per hyperparameter tuning..."
            )
            sample_features = []
            sample_size = min(100, len(train_dataset))
            print(f"  Usando {sample_size} campioni sequenziali per determinismo...")
            for i in range(sample_size):
                features, _ = train_dataset[i]
                sample_features.extend(features.reshape(-1).numpy())

            sample_features = np.array(sample_features).reshape(-1, 1)
            global_scaler.fit(sample_features)
            print(f"  Scaler globale fittato su {len(sample_features)} features")
            train_dataset._global_scaler = global_scaler

        # Wrappa i dataset con normalizzazione
        current_train_dataset = NormalizedDatasetWrapper(
            train_dataset,
            scaler=train_dataset._global_scaler,
            fit_scaler=False,
            normalization_type=NORMALIZATION_TYPE,
        )

        current_val_dataset = NormalizedDatasetWrapper(
            val_dataset,
            scaler=train_dataset._global_scaler,
            fit_scaler=False,
            normalization_type=NORMALIZATION_TYPE,
        )

    # --- Sezione 6: Creazione dei Dataloader specifici per il trial ---
    train_sampler = get_sampler(current_train_dataset)
    train_loader = DataLoader(
        current_train_dataset,
        batch_sampler=torch.utils.data.BatchSampler(
            train_sampler, batch_size, drop_last=False
        ),
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        current_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # --- Sezione 7: Setup del Modello e dell'Addestramento ---
    device = torch.device("cpu")

    class_weights_tensor = get_class_weights(current_train_dataset, device)
    input_size = current_train_dataset.num_features  # Usa la dimensione fissa
    num_classes = len(current_train_dataset.labels)

    model = create_model(MODEL_TYPE, input_size, num_classes, device, model_params)

    # --- LOSS FUNCTION SELECTION ---
    if use_focal_loss:
        criterion = FocalLoss(
            alpha=focal_alpha, gamma=focal_gamma, weight=class_weights_tensor
        )
        loss_type = "FocalLoss"
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        loss_type = "CrossEntropyLoss"
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    print(
        f"Trial {trial.number}: lr={learning_rate}, batch_size={batch_size}, weight_decay={weight_decay}, dropout={model_params.get('dropout', None)}"
    )
    print("Class weights:", class_weights_tensor.cpu().numpy())
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.2, patience=3
    )  # Scheduler più reattivo

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
            "loss_function": loss_type,
            "normalize_data": NORMALIZE_DATA,
            "normalization_type": NORMALIZATION_TYPE if NORMALIZE_DATA else None,
        }
        if use_focal_loss:
            base_params.update(
                {
                    "focal_alpha": focal_alpha,
                    "focal_gamma": focal_gamma,
                }
            )
        mlflow.log_params({**base_params, **model_params})

        best_f1_macro = 0.0
        best_weighted_f1 = 0.0
        best_class_gap = float("inf")

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            nonlocal best_f1_macro, best_weighted_f1, best_class_gap
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            val_f1_macro = metrics["f1_macro"]
            val_loss = metrics["loss"]

            # Calcola metriche weighted e analisi dettagliata
            with torch.no_grad():
                all_preds = []
                all_labels = []
                all_probs = []

                for xb, yb in val_loader:
                    x, y = prepare_batch((xb, yb), device, MODEL_TYPE)
                    outputs = model(x)
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(outputs, dim=1)

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(y.cpu().numpy())
                    all_probs.extend(
                        probs[:, 1].cpu().numpy()
                        if probs.shape[1] > 1
                        else probs[:, 0].cpu().numpy()
                    )  # Probabilità classe Positive

                # Calcola weighted metrics
                val_weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
                val_weighted_acc = accuracy_score(all_labels, all_preds)

                # Calcola class gap (differenza F1 tra classi)
                class_f1_scores = f1_score(all_labels, all_preds, average=None)
                if len(class_f1_scores) >= 2:
                    class_gap = abs(class_f1_scores[1] - class_f1_scores[0])
                else:
                    class_gap = 0.0

                # Analisi per classe
                neg_probs = [p for p, l in zip(all_probs, all_labels) if l == 0]
                pos_probs = [p for p, l in zip(all_probs, all_labels) if l == 1]

                avg_neg_prob = np.mean(neg_probs) if neg_probs else 0
                avg_pos_prob = np.mean(pos_probs) if pos_probs else 0

            # Update best metrics
            if val_f1_macro > best_f1_macro:
                best_f1_macro = val_f1_macro
            if val_weighted_f1 > best_weighted_f1:
                best_weighted_f1 = val_weighted_f1
            if class_gap < best_class_gap:
                best_class_gap = class_gap

            # Calcola F1 per classe
            class_f1_scores = f1_score(all_labels, all_preds, average=None)
            val_f1_class_0 = class_f1_scores[0] if len(class_f1_scores) > 0 else 0.0
            val_f1_class_1 = class_f1_scores[1] if len(class_f1_scores) > 1 else 0.0

            mlflow.log_metrics(
                {
                    "val_loss": val_loss,
                    "val_f1_macro": val_f1_macro,
                    "val_weighted_f1": val_weighted_f1,
                    "val_weighted_acc": val_weighted_acc,
                    "val_accuracy": metrics["accuracy"],
                    "val_precision_macro": metrics["precision_macro"],
                    "val_recall_macro": metrics["recall_macro"],
                    "val_class_gap": class_gap,
                    "val_f1_class_0": val_f1_class_0,
                    "val_f1_class_1": val_f1_class_1,
                    "avg_prob_negative": avg_neg_prob,
                    "avg_prob_positive": avg_pos_prob,
                },
                step=engine.state.epoch,
            )

            # Stampa progress base
            print(
                f"Trial {trial.number}, Epoch {engine.state.epoch}: "
                f"F1 Macro: {val_f1_macro:.4f}, Weighted F1: {val_weighted_f1:.4f}, "
                f"Weighted Acc: {val_weighted_acc:.4f}, Class Gap: {class_gap:.4f}"
            )

            # Stampa dettagliata ogni 5 epoche (per hyperparameter tuning)
            if engine.state.epoch % 5 == 0:
                print(
                    f"\nDetailed Analysis - Trial {trial.number} (Epoch {engine.state.epoch}):"
                )
                print(f"Avg prob for Negative samples: {avg_neg_prob:.4f}")
                print(f"Avg prob for Positive samples: {avg_pos_prob:.4f}")
                print(
                    f"Val F1 Class 0: {val_f1_class_0:.4f}, Val F1 Class 1: {val_f1_class_1:.4f}"
                )
                print(f"Class Gap: {class_gap:.4f}")
                print("Classification Report:")
                print(
                    classification_report(
                        all_labels, all_preds, target_names=["Negative", "Positive"]
                    )
                )
                print("Confusion Matrix:")
                print(confusion_matrix(all_labels, all_preds))
                print("-" * 50)

            # Scegli la metrica per il scheduler e il pruning
            if OPTIMIZE_METRIC == "weighted_f1":
                optimize_value = val_weighted_f1
            elif OPTIMIZE_METRIC == "class_gap":
                optimize_value = -class_gap  # Minimizziamo il gap
            else:
                optimize_value = val_f1_macro

            scheduler.step(optimize_value)
            trial.report(optimize_value, engine.state.epoch)
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
        mlflow.log_metric("best_val_weighted_f1", best_weighted_f1)
        mlflow.log_metric("best_val_class_gap", best_class_gap)

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

        # Return the metric we're optimizing for
        if OPTIMIZE_METRIC == "weighted_f1":
            return best_weighted_f1
        elif OPTIMIZE_METRIC == "class_gap":
            return -best_class_gap  # Optuna maximizes, so negative gap
        else:
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
    parser.add_argument(
        "--downsample_ratio",
        type=float,
        default=0.0,  # 0.0 means no downsampling
        help="Ratio to downsample the majority class. 1.0 means balance with minority. 0.0 to disable.",
    )
    parser.add_argument(
        "--use_focal_loss",
        action="store_true",
        help="Enable Focal Loss optimization in hyperparameter search.",
    )
    parser.add_argument(
        "--optimize_metric",
        type=str,
        choices=["f1_macro", "weighted_f1", "class_gap"],
        default="f1_macro",
        help="Metric to optimize during hyperparameter tuning.",
    )
    parser.add_argument(
        "--normalize_data",
        action="store_true",
        help="Applica normalizzazione ai dati durante hyperparameter tuning",
    )
    parser.add_argument(
        "--normalization_type",
        type=str,
        default="minmax",
        choices=["minmax", "standard"],
        help="Tipo di normalizzazione: 'minmax' per [0,1], 'standard' per z-score",
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

    global NUM_EPOCHS, MODEL_TYPE, NUM_WORKERS, USE_FOCAL_LOSS, OPTIMIZE_METRIC, NORMALIZE_DATA, NORMALIZATION_TYPE
    NUM_EPOCHS = args.num_epochs
    MODEL_TYPE = args.model_type
    NUM_WORKERS = args.num_workers
    USE_FOCAL_LOSS = args.use_focal_loss
    OPTIMIZE_METRIC = args.optimize_metric
    NORMALIZE_DATA = args.normalize_data
    NORMALIZATION_TYPE = args.normalization_type

    # --- Caricamento Dati (una sola volta) ---
    print("Caricamento dei dataset...")
    train_dataset, val_dataset = get_datasets(
        TRAIN_LANDMARKS_DIR,
        TRAIN_PROCESSED_FILE,
        VAL_LANDMARKS_DIR,
        VAL_PROCESSED_FILE,
        downsample_majority_class=args.downsample_ratio > 0,
        downsample_ratio=args.downsample_ratio,
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

    # --- Setup di MLflow dinamico ---
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    base_experiment_name = "VADER 0.34 - HT Emotion Recognition"

    # Build experiment name with configuration details
    name_parts = []
    if args.downsample_ratio > 0:
        name_parts.append(f"DS:{args.downsample_ratio}")
    if args.use_focal_loss:
        name_parts.append("FocalLoss")
    if args.normalize_data:
        name_parts.append(f"Norm:{args.normalization_type}")
    if args.optimize_metric != "f1_macro":
        name_parts.append(f"Opt:{args.optimize_metric}")

    if name_parts:
        experiment_name = f"{base_experiment_name} ({', '.join(name_parts)})"
    else:
        experiment_name = f"{base_experiment_name} (Baseline)"

    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    # ---

    run_name = f"Optuna_{MODEL_TYPE}_tuning_optimized"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("seed", seed)
        mlflow.log_param("downsample_ratio", args.downsample_ratio)
        mlflow.log_param("use_focal_loss", args.use_focal_loss)
        mlflow.log_param("optimize_metric", args.optimize_metric)
        mlflow.log_param("normalize_data", args.normalize_data)
        mlflow.log_param("normalization_type", args.normalization_type)
        # Logga la distribuzione effettiva dopo il downsampling
        mlflow.log_param("train_label_distribution", train_dist)
        mlflow.log_param("val_label_distribution", val_dist)

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
                "run_id": run.info.run_id,
            }
        )

    print("Best trial:", study.best_trial.params)
    print("Best value:", study.best_value)


if __name__ == "__main__":
    main()
