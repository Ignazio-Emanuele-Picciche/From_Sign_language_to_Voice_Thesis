# -----------------------------------------------------------------------------
# run_train_vivit.py
# -----------------------------------------------------------------------------
# Descrizione (in italiano):
# Questo script addestra un modello ViViT (Video Vision Transformer) per la
# classificazione delle emozioni a partire da video. Implementa il flusso
# tipico di training: caricamento dei dati, costruzione del modello (da
# Hugging Face), definizione di loss e ottimizzatore, loop di training con
# Ignite, valutazione su validation set, checkpointing e early stopping.
#
# Principali funzionalità:
# - Configurazione di MLflow per tracciare parametri e metriche del run.
# - Costruzione del modello ViViT tramite la funzione `create_vivit_model`.
# - Dataset video custom (`VideoDataset`) che fornisce batch di
#   `pixel_values` e `labels` compatibili con il modello.
# - Dataloader di training con `WeightedRandomSampler` per mitigare
#   sbilanciamento delle classi.
# - Loop di training basato su `ignite.engine.Engine` con vari metriche
#   (loss, accuracy, f1, precision, recall) calcolate sull'evaluator.
# - Checkpoint automatico del modello con il criterio di selezione
#   basato sulla F1 macro di validazione e EarlyStopping.
#
# Input / Assunzioni:
# - Esistono CSV di annotazioni in `data/processed/train/` e
#   `data/processed/val/` con almeno una colonna `emotion`.
# - I video raw sono accessibili nei percorsi indicati dalle variabili
#   `TRAIN_VIDEO_DIR` e `VAL_VIDEO_DIR` (definite nel file).
# - Un server MLflow può essere attivato per visualizzare i run
#   (opzionale ma consigliato): `mlflow server --host 127.0.0.1 --port 8080`.
#
# Output e effetti collaterali:
# - Salvataggio dei checkpoint nella cartella `models/` (file nominati
#   con prefisso `vivit_emotion` e punteggio di validazione).
# - Metriche e parametri loggati su MLflow per ogni epoca di validazione.
# - Il modello finale (ultimo checkpoint) sarà disponibile nella cartella
#   `models/` e il training produce log sullo stdout.
#
# Esempio di esecuzione:
# python src/models/vivit/run_train_vivit.py --num_epochs 50 --batch_size 1 --learning_rate 2.310201887845294e-06 --seed 42 --weight_decay 0.0003549878832196505
# python src/models/vivit/run_train_vivit.py --num_epochs 50 --batch_size 1 --learning_rate 1.5930522616241016e-05 --seed 42 --weight_decay 0.013311216080736894
# python src/models/vivit/run_train_vivit.py --num_epochs 50 --batch_size 1 --learning_rate 2.9106359131330718e-05 --seed 42 --weight_decay 0.006251373574521752
#
# Note tecniche veloci:
# - Usa `AdamW` sull'head di classificazione; la loss è CrossEntropy con
#   pesi calcolati dal dataset per gestire lo sbilanciamento.
# - Ignite è usato per separare train/eval e per collegare EarlyStopping
#   e ModelCheckpoint in modo semplice.
# -----------------------------------------------------------------------------

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import mlflow
import mlflow.pytorch
from collections import Counter
import logging
import gc  # Aggiunto per la garbage collection

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
from src.utils.training_utils import setup_ignite_evaluator  # Riutilizziamo l'evaluator

# Ignite imports
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.metrics import Loss, Accuracy, Precision, Recall, Fbeta

from ignite.contrib.handlers import ProgressBar


# Configura il logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Sezione 2: Definizione dei Percorsi Dati ---
TRAIN_VIDEO_DIR = os.path.join(
    BASE_DIR, "data", "raw", "train", "raw_videos_front_train"
)
VAL_VIDEO_DIR = os.path.join(BASE_DIR, "data", "raw", "val", "raw_videos_front_val")
# Aggiorniamo i percorsi per puntare ai file CSV corretti.
# Assumiamo che esista un file analogo per la validazione.
# Se non esiste, andrebbe creato o il dataset di training andrebbe splittato.
TRAIN_ANNOTATIONS_FILE = os.path.join(
    BASE_DIR, "data", "processed", "train", "video_sentiment_data_0.65.csv"
)
VAL_ANNOTATIONS_FILE = os.path.join(
    BASE_DIR, "data", "processed", "val", "video_sentiment_data_0.65.csv"
)
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "vivit_emotion.pth")


def get_class_weights(dataset):
    """Calcola i pesi per ogni classe per gestire lo sbilanciamento."""
    # La colonna delle etichette nel CSV si chiama 'emotion'
    class_counts = Counter(dataset.video_info["emotion"])
    class_weights = {
        dataset.label2id[cls]: 1.0 / count for cls, count in class_counts.items()
    }
    # Assicura che l'ordine sia corretto
    weights = [class_weights[i] for i in sorted(class_weights.keys())]
    return torch.tensor(weights, dtype=torch.float)


def get_sampler(dataset):
    """Crea un WeightedRandomSampler per il dataloader di training."""
    class_weights = get_class_weights(dataset)
    # La colonna delle etichette nel CSV si chiama 'emotion'
    sample_weights = [
        class_weights[dataset.label2id[label]]
        for label in dataset.video_info["emotion"]
    ]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )
    return sampler


def prepare_batch(batch, device=None, non_blocking=False):
    """Sposta il batch di dati sul dispositivo corretto."""
    pixel_values = batch["pixel_values"].to(device, non_blocking=non_blocking)
    labels = batch["labels"].to(device, non_blocking=non_blocking)
    return pixel_values, labels


def set_seed(seed):
    """Imposta il seed per la riproducibilità."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Per una riproducibilità completa (potrebbe rallentare il training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    """Funzione principale per l'addestramento."""
    # Imposta il seed per la riproducibilità
    set_seed(args.seed)

    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    experiment_name = "ViViT - Emotion Recognition"
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    # Estrai un nome breve del modello per il run_name
    model_short_name = args.model_name.split("/")[-1]
    run_name = f"{model_short_name}_train_{args.num_epochs}epochs"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(vars(args))

        # --- Setup del Modello e Device ---
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        logger.info(f"Using device: {device}")

        # --- Caricamento Dati ---
        # Creiamo un'istanza preliminare del dataset solo per ottenere l'image_processor e num_classes
        _, temp_image_processor = create_vivit_model(
            num_classes=2, model_name=args.model_name
        )  # num_classes temporaneo
        train_dataset = VideoDataset(
            TRAIN_ANNOTATIONS_FILE, TRAIN_VIDEO_DIR, temp_image_processor
        )
        val_dataset = VideoDataset(
            VAL_ANNOTATIONS_FILE, VAL_VIDEO_DIR, temp_image_processor
        )

        # Ora otteniamo il numero corretto di classi dal dataset
        num_classes = len(train_dataset.labels)
        logger.info(f"Numero di classi rilevato: {num_classes}")

        # Creiamo il modello con il numero corretto di classi
        model, image_processor = create_vivit_model(
            num_classes, model_name=args.model_name
        )
        model.to(device)
        # Calcolo e log dei parametri del modello
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

        # Usa il numero di frame del modello per campionare correttamente
        num_frames = model.config.num_frames  # es. 32
        # Ricrea i dataset con l'image_processor finale e il numero corretto di frame
        train_dataset = VideoDataset(
            TRAIN_ANNOTATIONS_FILE,
            TRAIN_VIDEO_DIR,
            image_processor,
            num_frames=num_frames,
        )
        val_dataset = VideoDataset(
            VAL_ANNOTATIONS_FILE, VAL_VIDEO_DIR, image_processor, num_frames=num_frames
        )

        train_sampler = get_sampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,  # Aggiunto per parallelizzare il caricamento
            pin_memory=True,  # Aggiunto per velocizzare il trasferimento dei dati a GPU
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        # --- Setup Addestramento ---
        optimizer = torch.optim.AdamW(  # Usiamo AdamW che è spesso migliore
            model.classifier.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        # Calcoliamo i pesi per la loss in base al dataset di training
        class_weights_tensor = get_class_weights(train_dataset).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

        # ! TODO DA AGGIUNGERE IN SEGUITO
        # Aggiungiamo uno scheduler per il learning rate
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
        )

        # --- Ignite Trainer ed Evaluator ---
        def train_step(engine, batch):
            model.train()
            optimizer.zero_grad()
            try:
                pixel_values, labels = prepare_batch(batch, device=device)
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                return loss.item()
            except Exception as e:
                logger.error(f"\nError during train_step: {e}")
                raise

        trainer = Engine(train_step)

        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, output_transform=lambda x: {"batch loss": x})

        # # Logging training progress
        # @trainer.on(Events.EPOCH_STARTED)
        # def log_epoch_start(engine):
        #     logger.info(f"Starting Epoch {engine.state.epoch}.")

        # @trainer.on(Events.ITERATION_COMPLETED(every=10))
        # def log_iteration(engine):
        #     loss = engine.state.output
        #     logger.info(
        #         f"Epoch[{engine.state.epoch}] Iteration[{engine.state.iteration}] Loss: {loss:.4f}"
        #     )

        # Per l'evaluator, dobbiamo adattare l'output_transform
        def output_transform(output):
            y_pred, y = output
            # L'output del modello HuggingFace è un oggetto, estraiamo i logits
            return y_pred.logits, y

        val_metrics = {
            "accuracy": Accuracy(output_transform=output_transform, device=device),
            "loss": Loss(criterion, output_transform=output_transform, device=device),
            "f1": Fbeta(
                1.0, average="macro", output_transform=output_transform, device=device
            ),
            "precision": Precision(
                average="macro", output_transform=output_transform, device=device
            ),
            "recall": Recall(
                average="macro", output_transform=output_transform, device=device
            ),
        }

        # --- Per f1 per classe e best metrics ---
        from sklearn.metrics import f1_score

        best_val_loss = float("inf")
        best_val_f1 = 0.0
        best_val_f1_macro = 0.0

        def eval_step(engine, batch):
            model.eval()
            with torch.no_grad():
                pixel_values, labels = prepare_batch(batch, device=device)
                outputs = model(pixel_values=pixel_values)
                return outputs, labels

        evaluator = Engine(eval_step)
        for name, metric in val_metrics.items():
            metric.attach(evaluator, name)

        # --- Handlers e Callbacks ---
        train_losses = []

        @trainer.on(Events.ITERATION_COMPLETED(every=len(train_loader)))
        def log_train_loss(engine):
            # Log train loss a fine epoca
            train_loss = engine.state.output
            train_losses.append(train_loss)
            mlflow.log_metric("train_loss", train_loss, step=engine.state.epoch)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            nonlocal best_val_loss, best_val_f1, best_val_f1_macro
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            # Calcolo f1 per classe
            y_true = []
            y_pred = []
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    pixel_values, labels = prepare_batch(batch, device=device)
                    outputs = model(pixel_values=pixel_values)
                    preds = torch.argmax(outputs.logits, dim=1)
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(preds.cpu().numpy())
            f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
            val_f1_class_0 = float(f1_per_class[0]) if len(f1_per_class) > 0 else 0.0
            val_f1_class_1 = float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0

            # Migliori metriche
            if metrics["loss"] < best_val_loss:
                best_val_loss = metrics["loss"]
            if metrics["accuracy"] > best_val_f1:
                best_val_f1 = metrics["accuracy"]
            if metrics["f1"] > best_val_f1_macro:
                best_val_f1_macro = metrics["f1"]

            # ! TODO DA AGGIUNGERE IN SEGUITO
            # Step dello scheduler
            scheduler.step(metrics["loss"])

            # Log learning rate
            current_lr = optimizer.param_groups[0]["lr"]
            mlflow.log_metric("learning_rate", current_lr, step=engine.state.epoch)

            logger.info(
                f"Validation Results - Epoch: {engine.state.epoch} | "
                f"Loss: {metrics['loss']:.4f} | "
                f"Accuracy: {metrics['accuracy']:.4f} | "
                f"F1: {metrics['f1']:.4f} | "
                f"F1_class_0: {val_f1_class_0:.4f} | F1_class_1: {val_f1_class_1:.4f} | "
                f"Precision: {metrics['precision']:.4f} | "
                f"Recall: {metrics['recall']:.4f}"
            )
            mlflow.log_metrics(
                {
                    "val_loss": metrics["loss"],
                    "val_accuracy": metrics["accuracy"],
                    "val_f1_macro": metrics["f1"],
                    "val_f1_class_0": val_f1_class_0,
                    "val_f1_class_1": val_f1_class_1,
                    "val_precision_macro": metrics["precision"],
                    "val_recall_macro": metrics["recall"],
                    "best_val_loss": best_val_loss,
                    "best_val_f1": best_val_f1,
                    "best_val_f1_macro": best_val_f1_macro,
                },
                step=engine.state.epoch,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Early stopping
        handler = EarlyStopping(
            patience=args.patience,
            score_function=lambda e: e.state.metrics["f1"],
            trainer=trainer,
        )
        evaluator.add_event_handler(Events.COMPLETED, handler)

        # Checkpoint
        checkpointer = ModelCheckpoint(
            os.path.join(BASE_DIR, "models"),
            "vivit_emotion",
            n_saved=1,
            create_dir=True,
            require_empty=False,
            score_function=lambda e: e.state.metrics["f1"],
            score_name="val_f1",
        )
        evaluator.add_event_handler(Events.COMPLETED, checkpointer, {"model": model})

        # --- Avvio Addestramento ---
        logger.info("Starting training...")
        trainer.run(train_loader, max_epochs=args.num_epochs)
        logger.info("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ViViT for emotion recognition.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/vivit-b-16x2-kinetics400",
        help="Name of the Hugging Face model to use.",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Training batch size."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-2, help="Weight decay for AdamW."
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Patience for early stopping."
    )

    # ! TODO DA AGGIUNGERE IN SEGUITO
    parser.add_argument(
        "--scheduler_patience", type=int, default=5, help="Patience for LR scheduler."
    )
    # NOTE: passo da 0.1 a 0.2 perche dopo l'aggiornamneto del LR la loss smette di migliorare. Significa che abbiamo auvuto un aggiornamento troppo drastico del LR.
    parser.add_argument(
        "--scheduler_factor", type=float, default=0.3, help="Factor for LR scheduler."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of worker processes for data loading.",
    )
    args = parser.parse_args()
    main(args)
