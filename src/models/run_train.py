# =================================================================================================
# COMANDI UTILI PER L'ESECUZIONE
# =================================================================================================
#
# 1. AVVIARE IL SERVER MLFLOW
#    Per monitorare gli esperimenti, prima di eseguire lo script di training,
#    aprire un terminale e lanciare il server MLflow:
#
#    mlflow server --host 127.0.0.1 --port 8080
#
# 2. ESEGUIRE L'ADDESTRAMENTO DEL MODELLO
#    Questo script permette di addestrare un modello per il riconoscimento delle emozioni.
#    È possibile specificare il tipo di modello (lstm o stgcn) e i relativi iperparametri.
#
#    ESEMPIO DI COMANDO PER LSTM (con iperparametri ottimizzati):
#    Questo comando avvia l'addestramento di un modello LSTM con i migliori iperparametri
#    trovati durante la fase di tuning.
#
#    python src/models/run_train.py \
#      --model_type lstm \
#      --batch_size 128 \
#      --hidden_size 896 \
#      --num_layers 3 \
#      --learning_rate 2.621087878265438e-05 \
#      --dropout 0.23993475643167195 \
#      --num_epochs 100 \
#      --seed 44
#
#    ESEMPIO DI COMANDO PER ST-GCN (da personalizzare):
#    Questo comando avvia l'addestramento di un modello ST-GCN.
#    Gli iperparametri come learning_rate e dropout andrebbero ottimizzati.
#
#    python src/models/run_train.py \
#      --model_type stgcn \
#      --batch_size 64 \
#      --learning_rate 1e-4 \
#      --dropout 0.1 \
#      --num_epochs 100 \
#      --seed 44
#
# *   Comando per LSTM con VADER 0.34
#   python src/models/run_train.py \
#       --model_type lstm \
#      --batch_size 32 \
#      --hidden_size 352 \
#      --num_layers 1 \
#      --learning_rate 2.4165903162442322e-05 \
#      --dropout 0.5293016342019151 \
#      --weight_decay 0.04890168508329703 \
#      --num_epochs 100 \
#      --patience 20 \
#      --seed 42 \
#      --downsample_ratio 1.0


# python src/models/run_train.py \
#     --model_type lstm \
#     --batch_size 32 \
#     --hidden_size 384 \
#     --num_layers 2 \
#     --learning_rate 1.2686928153291844e-05 \
#     --dropout 0.5913841307520112   \
#     --weight_decay 0.04890168508329703 \
#     --num_epochs 100 \
#     --patience 20 \
#     --seed 42 \
#     --downsample_ratio 1.0


# python src/models/run_train.py \
# --model_type lstm \
# --batch_size 32 \
# --hidden_size 192 \
# --num_layers 2 \
# --learning_rate 3.5253626370141006e-05 \
# --dropout 0.5422772674924288   \
# --weight_decay 0.012485368572526356 \
# --num_epochs 100 \
# --patience 20 \
# --seed 42 \
# --downsample_ratio 1.0 \
# --normalize_data


# python src/models/run_train.py \
# --model_type lstm \
# --batch_size 32 \
# --hidden_size 192 \
# --num_layers 2 \
# --learning_rate 1.655914944036191e-05 \
# --dropout 0.5935133228268233   \
# --weight_decay 0.43464957555697725 \
# --num_epochs 100 \
# --patience 20 \
# --seed 42 \
# --downsample_ratio 1.0 \
# --normalize_data

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os, sys
import gc
import argparse
import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature
import requests  # for pinging MLflow server
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# --- Sezione 1: Setup del Percorso di Base e Import delle Utilità ---
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
sys.path.insert(0, BASE_DIR)

# Ignite imports
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
from ignite.engine import Events, create_supervised_trainer
from ignite.handlers import EarlyStopping as IgniteEarlyStopping, ModelCheckpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.backends import cudnn
import random
from ignite.contrib.handlers import ProgressBar

from src.utils.training_utils import (
    get_data_paths,
    get_datasets,
    get_class_weights,
    create_model,
    prepare_batch,
    setup_ignite_evaluator,
    get_sampler,
)

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

        # Fitta lo scaler sui dati se richiesto
        if self.fit_scaler:
            print(f"Fitting scaler '{normalization_type}' sui dati di training...")
            self._fit_scaler()
            print(f"Scaler fittato.")

    def _fit_scaler(self):
        """Fitta lo scaler su un campione di dati del dataset"""
        all_features = []
        sample_size = min(100, len(self.dataset))  # Usa un campione per efficienza
        indices = np.random.choice(len(self.dataset), sample_size, replace=False)

        for idx in indices:
            if idx % 20 == 0:
                print(f"  Elaborando campione {idx}/{sample_size} per scaler")

            features, _ = self.dataset[idx]
            features_flat = features.reshape(-1).numpy()
            all_features.extend(features_flat)

        all_features = np.array(all_features).reshape(-1, 1)
        self.scaler.fit(all_features)
        print(f"Scaler fittato su {len(all_features)} features")

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

    def __init__(self, alpha=1.743106263727894, gamma=1.3918833167339733, weight=None):
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


def main(args):
    """Funzione principale che orchestra l'addestramento e la valutazione del modello."""
    # --- Sezione 2: Setup dell'Ambiente ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Training su dispositivo: {device}")

    mlflow.set_tracking_uri("http://127.0.0.1:8080")

    # --- Setup di MLflow dinamico (come in hyperparameter_tuning) ---
    base_experiment_name = "VADER 0.34 - Emotion Recognition"

    # Costruisci nome esperimento con configurazioni
    name_parts = []
    if args.downsample_ratio > 0:
        name_parts.append(f"DS:{args.downsample_ratio}")
    if args.normalize_data:
        name_parts.append(f"Norm:{args.normalization_type}")

    if name_parts:
        experiment_name = f"{base_experiment_name} ({', '.join(name_parts)})"
    else:
        experiment_name = f"{base_experiment_name} (Baseline)"

    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    # --- Sezione 3: Caricamento Dati e Creazione del Modello ---
    (
        train_landmarks_dir,
        train_processed_file,
        val_landmarks_dir,
        val_processed_file,
    ) = get_data_paths(BASE_DIR)

    # --- Caricamento Dati (con supporto downsampling) ---
    print("Caricamento dei dataset...")
    train_dataset, val_dataset = get_datasets(
        train_landmarks_dir,
        train_processed_file,
        val_landmarks_dir,
        val_processed_file,
        downsample_majority_class=args.downsample_ratio > 0,
        downsample_ratio=args.downsample_ratio,
    )

    # --- Applicazione Normalizzazione (se abilitata) ---
    if args.normalize_data:
        print(f"\nApplicazione normalizzazione '{args.normalization_type}'...")

        # Wrappa il training dataset con normalizzazione (fitta lo scaler)
        train_dataset = NormalizedDatasetWrapper(
            train_dataset, normalization_type=args.normalization_type, fit_scaler=True
        )

        # Wrappa il validation dataset con lo stesso scaler (senza fittare)
        val_dataset = NormalizedDatasetWrapper(
            val_dataset,
            scaler=train_dataset.scaler,
            fit_scaler=False,
            normalization_type=args.normalization_type,
        )

        # Verifica normalizzazione
        sample_features, _ = train_dataset[0]
        print(f"Normalizzazione applicata:")
        print(f"  Range: [{sample_features.min():.6f}, {sample_features.max():.6f}]")
        print(f"  Media: {sample_features.mean():.6f}")
        print(f"  Std: {sample_features.std():.6f}")

    print("Dataset preparati.")

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

    # Creazione dei DataLoader
    train_sampler = get_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=torch.utils.data.BatchSampler(
            train_sampler, args.batch_size, drop_last=False
        ),
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    class_weights = get_class_weights(train_dataset, device)

    input_size = train_dataset.num_features  # Usa la dimensione fissa
    num_classes = len(train_dataset.labels)

    model_params = {
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
    }

    model = create_model(
        args.model_type, input_size, num_classes, device, model_params
    ).to(device)

    print(
        f"Model params: lr={args.learning_rate}, batch_size={args.batch_size}, weight_decay={args.weight_decay}, dropout={model_params.get('dropout', None)}"
    )
    print("Class weights:", class_weights.cpu().numpy())

    # ! --- LOSS FUNCTION SELECTION ---
    # CrossEntropyLoss (commentato, per rollback facile)
    # criterion = nn.CrossEntropyLoss(weight=class_weights)

    # * Focal Loss (per gestire meglio lo sbilanciamento delle classi)
    criterion = FocalLoss(
        alpha=0.5975773894779193, gamma=2.8722138431333333, weight=class_weights
    )
    print(f"Using Focal Loss with alpha=0.5975773894779193, gamma=2.8722138431333333	")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.2, patience=3)

    prepare_batch_fn = lambda batch, device, non_blocking: prepare_batch(
        batch, device, args.model_type, non_blocking
    )

    trainer = create_supervised_trainer(
        model, optimizer, criterion, device=device, prepare_batch=prepare_batch_fn
    )
    # Aggiunta barra di progresso simile a hyperparameter_tuning
    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, output_transform=lambda loss: {"batch loss": loss})

    # --- Sezione 4: Definizione delle Metriche e Handlers ---
    evaluator = setup_ignite_evaluator(
        model, criterion, device, args.model_type, val_loader
    )

    checkpoint_handler = ModelCheckpoint(
        dirname=os.path.join(BASE_DIR, "models"),
        filename_prefix=f"emotion_{args.model_type}",
        n_saved=1,
        create_dir=True,
        require_empty=False,
        score_function=lambda eng: eng.state.metrics.get("val_weighted_f1", 0),
        score_name="val_weighted_f1",
    )
    evaluator.add_event_handler(Events.COMPLETED, checkpoint_handler, {"model": model})

    earlystop_handler = IgniteEarlyStopping(
        patience=args.patience,
        score_function=lambda eng: eng.state.metrics["val_f1_macro"],
        trainer=trainer,
    )
    evaluator.add_event_handler(Events.COMPLETED, earlystop_handler)

    # --- Sezione 5: Ciclo di Addestramento ---
    print("Inizio training con validazione (Ignite)...")
    # Numero di campioni usati nel training e validation
    print(f"Numero di campioni di training: {len(train_loader.dataset)}")
    print(f"Numero di campioni di validation: {len(val_loader.dataset)}")
    run_name = f"EmotionRecognition_{args.model_type}_train"

    with mlflow.start_run(run_name=run_name) as run:
        params = {
            "model_type": args.model_type,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "seed": args.seed,
            "loss_function": "FocalLoss",
            "focal_alpha": 0.5975773894779193,
            "focal_gamma": 2.8722138431333333,
            "normalize_data": args.normalize_data,
            "normalization_type": (
                args.normalization_type if args.normalize_data else None
            ),
        }
        if args.model_type == "lstm":
            params.update(model_params)
        else:  # stgcn
            params["stgcn_dropout"] = args.dropout

        mlflow.log_params(params)
        mlflow.log_param("downsample_ratio", args.downsample_ratio)
        # Logga la distribuzione effettiva dopo il downsampling
        mlflow.log_param("train_label_distribution", train_dist)
        mlflow.log_param("val_label_distribution", val_dist)

        best_metrics = {
            "val_loss": float("inf"),
            "val_f1": 0.0,
            "val_f1_macro": 0.0,
            "val_weighted_f1": 0.0,
            "val_weighted_acc": 0.0,
        }

        @trainer.on(Events.EPOCH_COMPLETED)
        def validate_and_log(engine):
            model.eval()
            train_loss_sum = 0.0
            with torch.no_grad():
                for xb, yb in train_loader:
                    xb, yb = prepare_batch_fn((xb, yb), device, non_blocking=False)
                    outputs = model(xb.float())
                    loss = criterion(outputs, yb)
                    train_loss_sum += loss.item() * xb.size(0)
            train_loss = train_loss_sum / len(train_loader.dataset)

            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            val_loss = metrics["val_loss"]
            val_acc = metrics["val_acc"]
            val_f1 = metrics["val_f1"]
            val_f1_macro = metrics["val_f1_macro"]

            # Calcola metriche weighted e analisi dettagliata
            with torch.no_grad():
                all_preds = []
                all_labels = []
                all_probs = []

                for xb, yb in val_loader:
                    xb, yb = prepare_batch_fn((xb, yb), device, non_blocking=False)
                    outputs = model(xb.float())
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(outputs, dim=1)

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(yb.cpu().numpy())
                    all_probs.extend(
                        probs[:, 1].cpu().numpy()
                    )  # Probabilità classe Positive

                # Calcola weighted metrics
                val_weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
                val_weighted_acc = accuracy_score(all_labels, all_preds)

                # Analisi per classe
                neg_probs = [p for p, l in zip(all_probs, all_labels) if l == 0]
                pos_probs = [p for p, l in zip(all_probs, all_labels) if l == 1]

                avg_neg_prob = np.mean(neg_probs) if neg_probs else 0
                avg_pos_prob = np.mean(pos_probs) if pos_probs else 0

            if val_loss < best_metrics["val_loss"]:
                best_metrics["val_loss"] = val_loss
            if val_f1 > best_metrics["val_f1"]:
                best_metrics["val_f1"] = val_f1
            if val_f1_macro > best_metrics["val_f1_macro"]:
                best_metrics["val_f1_macro"] = val_f1_macro
            if val_weighted_f1 > best_metrics["val_weighted_f1"]:
                best_metrics["val_weighted_f1"] = val_weighted_f1
            if val_weighted_acc > best_metrics["val_weighted_acc"]:
                best_metrics["val_weighted_acc"] = val_weighted_acc

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "val_f1_macro": val_f1_macro,
                    "val_weighted_f1": val_weighted_f1,
                    "val_weighted_acc": val_weighted_acc,
                    "val_f1_class_0": metrics["val_f1_class_0"],
                    "val_f1_class_1": metrics["val_f1_class_1"],
                    "avg_prob_negative": avg_neg_prob,
                    "avg_prob_positive": avg_pos_prob,
                },
                step=engine.state.epoch,
            )

            print(
                f"Epoch {engine.state.epoch}/{args.num_epochs}, "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Val F1 Macro: {val_f1_macro:.4f}, "
                f"Val Weighted F1: {val_weighted_f1:.4f}, Val Weighted Acc: {val_weighted_acc:.4f}"
            )

            # Stampa dettagliata ogni 10 epoche
            if engine.state.epoch % 10 == 0:
                print(f"\nDetailed Analysis (Epoch {engine.state.epoch}):")
                print(f"Avg prob for Negative samples: {avg_neg_prob:.4f}")
                print(f"Avg prob for Positive samples: {avg_pos_prob:.4f}")
                print(
                    f"Val Weighted F1: {val_weighted_f1:.4f}, Val Weighted Acc: {val_weighted_acc:.4f}"
                )
                print("Classification Report:")
                print(
                    classification_report(
                        all_labels, all_preds, target_names=["Negative", "Positive"]
                    )
                )
                print("Confusion Matrix:")
                print(confusion_matrix(all_labels, all_preds))
                print("-" * 50)

            scheduler.step(val_weighted_f1)

            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()

        trainer.run(train_loader, max_epochs=args.num_epochs)

        print(
            f"Training completato. Best model saved in {checkpoint_handler.last_checkpoint}"
        )

        mlflow.log_metric("best_val_loss", best_metrics["val_loss"])
        mlflow.log_metric("best_val_f1", best_metrics["val_f1"])
        mlflow.log_metric("best_val_f1_macro", best_metrics["val_f1_macro"])
        mlflow.log_metric("best_val_weighted_f1", best_metrics["val_weighted_f1"])
        mlflow.log_metric("best_val_weighted_acc", best_metrics["val_weighted_acc"])

        data_sample, _ = next(iter(train_loader))
        data_sample, _ = prepare_batch_fn((data_sample, _), device, non_blocking=False)

        signature = infer_signature(
            data_sample.cpu().numpy(),
            model(data_sample.float()).cpu().detach().numpy(),
        )
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name="EmoSign_pytorch",
        )
        mlflow.set_tag("experiment_purpose", "PyTorch emotion recognition")


# --- Sezione 6: Parsing degli Argomenti da Linea di Comando ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Emotion Recognition Model")
    parser.add_argument(
        "--model_type", type=str, default="lstm", choices=["lstm", "stgcn"]
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--downsample_ratio",
        type=float,
        default=0.0,  # 0.0 means no downsampling
        help="Ratio to downsample the majority class. 1.0 means balance with minority. 0.0 to disable.",
    )
    parser.add_argument(
        "--normalize_data",
        action="store_true",
        help="Applica normalizzazione ai dati per migliorare robustezza e generalizzazione",
    )
    parser.add_argument(
        "--normalization_type",
        type=str,
        default="minmax",
        choices=["minmax", "standard"],
        help="Tipo di normalizzazione: 'minmax' per [0,1], 'standard' per z-score",
    )
    args = parser.parse_args()
    main(args)
