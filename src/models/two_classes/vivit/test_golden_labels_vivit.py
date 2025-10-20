# =================================================================================================
# TEST SCRIPT PER GOLDEN LABELS CON MODELLO VIVIT - 2 CLASSI
# =================================================================================================
#
# Questo script testa il modello ViViT addestrato (2 classi) sui golden labels.
# A differenza dell'LSTM che usa landmarks, ViViT lavora direttamente con i video.
#
# COMANDO PER ESEGUIRE IL TEST:
# python src/models/two_classes/vivit/test_golden_labels_vivit.py \
#   --model_uri mlartifacts/EXPERIMENT_ID/models/MODEL_ID/artifacts \
#   --batch_size 1 \
#   --save_results

# python src/models/two_classes/vivit/test_golden_labels_vivit.py \
#   --model_uri mlartifacts/697363764579443849/models/m-de73e05128734690a016c37e5610eeb2/artifacts \
#   --batch_size 1 \
#   --save_results

#
# =================================================================================================

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys
import argparse
import mlflow
import mlflow.pytorch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
)
from torch.utils.data import DataLoader
import logging

# --- Setup del Percorso di Base e Import delle Utilità ---
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir)
)
sys.path.insert(0, BASE_DIR)

from src.models.two_classes.vivit.video_dataset import VideoDataset
from src.models.two_classes.vivit.vivit_model import create_vivit_model

# Configura il logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_batch(batch, device=None, non_blocking=False):
    """Sposta il batch di dati sul dispositivo corretto."""
    pixel_values = batch["pixel_values"].to(device, non_blocking=non_blocking)
    labels = batch["labels"].to(device, non_blocking=non_blocking)
    return pixel_values, labels


def load_model_from_mlflow(model_uri):
    """
    Carica il modello da MLflow
    """
    logger.info(f"Caricamento modello da: {model_uri}")
    model = mlflow.pytorch.load_model(model_uri)
    return model


def evaluate_model(model, test_loader, device, labels):
    """
    Valuta il modello ViViT sui dati di test
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    logger.info("Esecuzione valutazione con modello ViViT...")

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i % 5 == 0:
                logger.info(f"  Processando batch {i+1}/{len(test_loader)}")

            # Prepara il batch
            pixel_values, batch_labels = prepare_batch(
                batch, device, non_blocking=False
            )

            # Inferenza
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            # Raccogli le predizioni
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def print_detailed_results(y_true, y_pred, y_probs, labels, video_names=None):
    """
    Stampa risultati dettagliati della valutazione
    """
    print("\n" + "=" * 70)
    print("RISULTATI TEST GOLDEN LABELS - MODELLO VIVIT (2 CLASSI)")
    print("=" * 70)

    # Metriche generali
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    precision_macro = precision_score(y_true, y_pred, average="macro")
    precision_weighted = precision_score(y_true, y_pred, average="weighted")
    recall_macro = recall_score(y_true, y_pred, average="macro")
    recall_weighted = recall_score(y_true, y_pred, average="weighted")

    # Calcola weighted accuracy personalizzata
    class_weights = np.bincount(y_true) / len(y_true)
    class_accuracies = []
    for i in range(len(labels)):
        mask = y_true == i
        if np.sum(mask) > 0:
            class_acc = accuracy_score(y_true[mask], y_pred[mask])
            class_accuracies.append(class_acc)
        else:
            class_accuracies.append(0.0)
    weighted_accuracy = np.sum(
        [class_weights[i] * class_accuracies[i] for i in range(len(labels))]
    )

    print(f"\n=== METRICHE PRINCIPALI ===")
    print(f"Accuracy (Standard): {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(
        f"Accuracy (Balanced): {balanced_accuracy:.4f} ({balanced_accuracy*100:.2f}%)"
    )
    print(
        f"Accuracy (Weighted): {weighted_accuracy:.4f} ({weighted_accuracy*100:.2f}%)"
    )
    print(f"\nF1-Score (Macro): {f1_macro:.4f} ({f1_macro*100:.2f}%)")
    print(f"F1-Score (Weighted): {f1_weighted:.4f} ({f1_weighted*100:.2f}%)")
    print(f"\nPrecision (Macro): {precision_macro:.4f} ({precision_macro*100:.2f}%)")
    print(
        f"Precision (Weighted): {precision_weighted:.4f} ({precision_weighted*100:.2f}%)"
    )
    print(f"\nRecall (Macro): {recall_macro:.4f} ({recall_macro*100:.2f}%)")
    print(f"Recall (Weighted): {recall_weighted:.4f} ({recall_weighted*100:.2f}%)")

    print(f"\nNumero totale di campioni: {len(y_true)}")

    # Distribuzione delle predizioni
    print(f"\nDistribuzione delle predizioni:")
    for i, label in enumerate(labels):
        count = np.sum(y_pred == i)
        percentage = count / len(y_pred) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")

    # Distribuzione delle etichette reali
    print(f"\nDistribuzione delle etichette reali:")
    for i, label in enumerate(labels):
        count = np.sum(y_true == i)
        percentage = count / len(y_true) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")

    # Analisi delle probabilità per classe
    print(f"\nAnalisi delle probabilità per classe:")
    for i, label in enumerate(labels):
        class_mask = y_true == i
        if np.sum(class_mask) > 0:
            avg_prob = np.mean(y_probs[class_mask, i])
            max_prob = np.max(y_probs[class_mask, i])
            min_prob = np.min(y_probs[class_mask, i])
            print(
                f"  Probabilità per campioni {label}: media={avg_prob:.3f}, max={max_prob:.3f}, min={min_prob:.3f}"
            )

    # Statistiche generali delle probabilità
    print(f"\nStatistiche generali delle probabilità:")
    for i, label in enumerate(labels):
        print(
            f"  Range probabilità {label}: [{y_probs[:, i].min():.3f}, {y_probs[:, i].max():.3f}]"
        )

    # Mostra quanti campioni hanno alta confidenza per ogni classe
    print(f"\nCampioni con alta confidenza per classe:")
    for i, label in enumerate(labels):
        high_conf = np.sum(y_probs[:, i] > 0.5)
        very_high_conf = np.sum(y_probs[:, i] > 0.7)
        print(f"  {label}: >50% = {high_conf}, >70% = {very_high_conf}")

    # Verifica confidenza predizioni
    confident_predictions = np.max(y_probs, axis=1) > 0.7
    print(
        f"  Predizioni confident (>70%): {np.sum(confident_predictions)} su {len(y_probs)} ({np.sum(confident_predictions)/len(y_probs)*100:.1f}%)"
    )

    # Mostra alcuni esempi di probabilità
    print(f"\nPrimi 10 campioni (vere etichette vs probabilità):")
    for i in range(min(10, len(y_true))):
        true_label = labels[y_true[i]]
        pred_label = labels[y_pred[i]]
        probs_str = " ".join(
            [f"P({labels[j][:3]})={y_probs[i, j]:.3f}" for j in range(len(labels))]
        )
        confidence = np.max(y_probs[i])
        status = "✅" if y_true[i] == y_pred[i] else "❌"
        video_info = f" [{video_names[i]}]" if video_names is not None else ""
        print(
            f"  {i+1:2d}: True={true_label:<8} Pred={pred_label:<8} {probs_str} Conf={confidence:.3f} {status}{video_info}"
        )

    # Confusion Matrix
    print(f"\nMatrice di Confusione:")
    cm = confusion_matrix(y_true, y_pred)
    print("    Predicted")
    print("    " + "  ".join([f"{label:>8}" for label in labels]))
    for i, label in enumerate(labels):
        print(
            f"{label:>6} " + "  ".join([f"{cm[i][j]:>8}" for j in range(len(labels))])
        )

    # Classification Report
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=labels))

    # F1-Score per classe
    print(f"\n=== F1-SCORE PER CLASSE ===")
    f1_per_class = f1_score(y_true, y_pred, average=None)
    for i, label in enumerate(labels):
        print(f"  {label}: {f1_per_class[i]:.4f} ({f1_per_class[i]*100:.2f}%)")

    # Check if all classes are being predicted
    print(f"\n" + "=" * 50)
    print("ANALISI DISTRIBUZIONE PREDIZIONI")
    print("=" * 50)
    classes_predicted = [np.sum(y_pred == i) > 0 for i in range(len(labels))]
    if all(classes_predicted):
        print("  ✅ SUCCESSO: Il modello predice tutte le classi!")
    else:
        missing_classes = [
            labels[i] for i in range(len(labels)) if not classes_predicted[i]
        ]
        print(f"  ⚠️  ATTENZIONE: Il modello NON predice: {', '.join(missing_classes)}")


def main(args):
    """
    Funzione principale per il test dei golden labels con ViViT (2 classi)
    """
    print("=" * 80)
    print("TEST GOLDEN LABELS CON MODELLO VIVIT (2 CLASSI)")
    print("=" * 80)

    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Utilizzo dispositivo: {device}")

    # Percorsi dei dati
    video_dir = os.path.join(
        BASE_DIR, "data", "raw", "ASLLRP", "batch_utterance_video_v3_1"
    )
    annotations_file = os.path.join(
        BASE_DIR, "data", "processed", "golden_label_sentiment.csv"
    )

    logger.info(f"Video directory: {video_dir}")
    logger.info(f"Annotations file: {annotations_file}")

    # Carica il modello
    logger.info("\nCaricamento modello...")
    model = load_model_from_mlflow(args.model_uri)
    model = model.to(device)
    logger.info("Modello caricato con successo!")

    # Ottieni l'image processor dal modello o crea uno nuovo
    from transformers import VivitImageProcessor

    image_processor = VivitImageProcessor.from_pretrained(
        args.model_name if args.model_name else "google/vivit-b-16x2-kinetics400"
    )

    # Ottieni il numero di frame dal modello
    num_frames = model.config.num_frames
    logger.info(f"Numero di frame per video: {num_frames}")

    # Carica il dataset
    logger.info("\nCaricamento dataset golden labels...")
    test_dataset = VideoDataset(
        annotations_file=annotations_file,
        video_root_dir=video_dir,
        image_processor=image_processor,
        num_frames=num_frames,
    )

    if len(test_dataset) == 0:
        logger.error("ERRORE: Nessun campione trovato nel dataset!")
        return

    logger.info(f"Dataset caricato: {len(test_dataset)} campioni")
    logger.info(f"Classi: {test_dataset.labels}")

    # Crea DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Per evitare problemi con MPS
        pin_memory=False,  # Disabilitato per MPS
    )

    # Valutazione
    logger.info("\nEsecuzione valutazione...")
    y_pred, y_true, y_probs = evaluate_model(
        model, test_loader, device, test_dataset.labels
    )

    # Ottieni i nomi dei video per riferimento
    video_names = test_dataset.video_info["video_name"].tolist()

    # Stampa risultati
    print_detailed_results(y_true, y_pred, y_probs, test_dataset.labels, video_names)

    # Salva risultati dettagliati in CSV
    if args.save_results:
        # Calcola tutte le metriche per il salvataggio
        accuracy = accuracy_score(y_true, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro")
        f1_weighted = f1_score(y_true, y_pred, average="weighted")
        precision_macro = precision_score(y_true, y_pred, average="macro")
        precision_weighted = precision_score(y_true, y_pred, average="weighted")
        recall_macro = recall_score(y_true, y_pred, average="macro")
        recall_weighted = recall_score(y_true, y_pred, average="weighted")

        # Calcola weighted accuracy personalizzata
        class_weights = np.bincount(y_true) / len(y_true)
        class_accuracies = []
        for i in range(len(test_dataset.labels)):
            mask = y_true == i
            if np.sum(mask) > 0:
                class_acc = accuracy_score(y_true[mask], y_pred[mask])
                class_accuracies.append(class_acc)
            else:
                class_accuracies.append(0.0)
        weighted_accuracy = np.sum(
            [
                class_weights[i] * class_accuracies[i]
                for i in range(len(test_dataset.labels))
            ]
        )

        # Salva risultati per-sample
        results_dict = {
            "video_name": video_names,
            "true_label": [test_dataset.labels[i] for i in y_true],
            "predicted_label": [test_dataset.labels[i] for i in y_pred],
        }

        # Add confidence for each class
        for i, label in enumerate(test_dataset.labels):
            results_dict[f"confidence_{label.lower()}"] = y_probs[:, i]

        results_dict["max_confidence"] = np.max(y_probs, axis=1)
        results_dict["correct_prediction"] = y_true == y_pred

        results_df = pd.DataFrame(results_dict)

        output_file = os.path.join(
            BASE_DIR,
            "results",
            "vivit_golden_labels_test_results_2_classes.csv",
        )
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        results_df.to_csv(output_file, index=False)
        logger.info(f"\nRisultati dettagliati salvati in: {output_file}")

        # Salva metriche aggregate
        metrics_df = pd.DataFrame(
            {
                "metric": [
                    "accuracy_standard",
                    "accuracy_balanced",
                    "accuracy_weighted",
                    "f1_macro",
                    "f1_weighted",
                    "precision_macro",
                    "precision_weighted",
                    "recall_macro",
                    "recall_weighted",
                ],
                "value": [
                    accuracy,
                    balanced_accuracy,
                    weighted_accuracy,
                    f1_macro,
                    f1_weighted,
                    precision_macro,
                    precision_weighted,
                    recall_macro,
                    recall_weighted,
                ],
                "percentage": [
                    accuracy * 100,
                    balanced_accuracy * 100,
                    weighted_accuracy * 100,
                    f1_macro * 100,
                    f1_weighted * 100,
                    precision_macro * 100,
                    precision_weighted * 100,
                    recall_macro * 100,
                    recall_weighted * 100,
                ],
                "description": [
                    "Accuracy standard: % di predizioni corrette sul totale",
                    "Accuracy bilanciata: accuracy robusta al class imbalance",
                    "Accuracy pesata: accuracy che considera la distribuzione delle classi",
                    "F1-Score macro: media non pesata dell'F1 per ogni classe",
                    "F1-Score weighted: F1 pesato per il supporto di ogni classe",
                    "Precision macro: media non pesata della precision per ogni classe",
                    "Precision weighted: precision pesata per il supporto di ogni classe",
                    "Recall macro: media non pesata del recall per ogni classe",
                    "Recall weighted: recall pesato per il supporto di ogni classe",
                ],
            }
        )

        metrics_file = os.path.join(
            BASE_DIR,
            "results",
            "vivit_golden_labels_metrics_summary_2_classes.csv",
        )
        metrics_df.to_csv(metrics_file, index=False)
        logger.info(f"Metriche aggregate salvate in: {metrics_file}")

        # Salva anche F1-Score per classe
        f1_per_class = f1_score(y_true, y_pred, average=None)
        f1_class_df = pd.DataFrame(
            {
                "class": test_dataset.labels,
                "f1_score": f1_per_class,
                "percentage": f1_per_class * 100,
            }
        )

        f1_class_file = os.path.join(
            BASE_DIR, "results", "vivit_golden_labels_f1_per_class_2_classes.csv"
        )
        f1_class_df.to_csv(f1_class_file, index=False)
        logger.info(f"F1-Score per classe salvato in: {f1_class_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test del modello ViViT (2 classi) sui Golden Labels"
    )
    parser.add_argument(
        "--model_uri", type=str, required=True, help="URI del modello MLflow"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Dimensione del batch per l'inferenza (raccomandato: 1 per ViViT)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/vivit-b-16x2-kinetics400",
        help="Nome del modello ViViT per caricare l'image processor corretto",
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Salva i risultati dettagliati in un file CSV",
    )

    args = parser.parse_args()
    main(args)
