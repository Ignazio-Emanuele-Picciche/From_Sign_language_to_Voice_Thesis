# =================================================================================================
# TEST SCRIPT PER GOLDEN LABELS CON NORMALIZZAZIONE CORRETTA
# =================================================================================================
#
# Questo script utilizza il modello addestrato con la normalizzazione corretta
# per testare i golden labels. Risolve il problema identificato nell'analisi
# dove i dati di training e golden labels erano su scale completamente diverse.
#
# COMANDO PER ESEGUIRE IL TEST:
# python src/models/test_golden_labels_fixed.py --model_uri mlartifacts/505616919019850588/models/m-c8c87c699cf14e4a91b1b9765feb9943/artifacts --batch_size 32 --save_results
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
)
from torch.utils.data import DataLoader

# --- Setup del Percorso di Base e Import delle Utilità ---
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
sys.path.insert(0, BASE_DIR)

from src.utils.training_utils import prepare_batch
from src.models.fix_golden_labels_normalization import FixedGoldenLabelDataset


def load_model_from_mlflow(model_uri):
    """
    Carica il modello da MLflow
    """
    print(f"Caricamento modello da: {model_uri}")
    model = mlflow.pytorch.load_model(model_uri)
    return model


def evaluate_model_fixed(model, test_loader, device, labels):
    """
    Valuta il modello sui dati di test con normalizzazione corretta
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    print("Esecuzione valutazione con normalizzazione corretta...")

    with torch.no_grad():
        for i, (batch_data, batch_labels) in enumerate(test_loader):
            if i % 10 == 0:
                print(f"  Processando batch {i+1}/{len(test_loader)}")

            # Prepara il batch usando la stessa logica di run_train.py
            batch_data, batch_labels = prepare_batch(
                (batch_data, batch_labels), device, "lstm", non_blocking=False
            )

            # Inferenza
            outputs = model(batch_data.float())
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            # Raccogli le predizioni
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def print_detailed_results_fixed(y_true, y_pred, y_probs, labels):
    """
    Stampa risultati dettagliati della valutazione con normalizzazione corretta
    """
    print("\n" + "=" * 70)
    print("RISULTATI CON NORMALIZZAZIONE CORRETTA")
    print("=" * 70)

    # Metriche generali
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    precision_macro = precision_score(y_true, y_pred, average="macro")
    recall_macro = recall_score(y_true, y_pred, average="macro")

    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"F1-Score (Macro): {f1_macro:.4f} ({f1_macro*100:.2f}%)")
    print(f"F1-Score (Weighted): {f1_weighted:.4f} ({f1_weighted*100:.2f}%)")
    print(f"Precision (Macro): {precision_macro:.4f} ({precision_macro*100:.2f}%)")
    print(f"Recall (Macro): {recall_macro:.4f} ({recall_macro*100:.2f}%)")

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
    print(
        f"  Range probabilità Negative: [{y_probs[:, 0].min():.3f}, {y_probs[:, 0].max():.3f}]"
    )
    print(
        f"  Range probabilità Positive: [{y_probs[:, 1].min():.3f}, {y_probs[:, 1].max():.3f}]"
    )
    print(f"  Soglia di decisione: 0.5")
    print(
        f"  Campioni con prob(Positive) > 0.5: {np.sum(y_probs[:, 1] > 0.5)} su {len(y_probs)}"
    )
    print(
        f"  Campioni con prob(Positive) > 0.7: {np.sum(y_probs[:, 1] > 0.7)} su {len(y_probs)}"
    )
    print(
        f"  Campioni con prob(Negative) > 0.7: {np.sum(y_probs[:, 0] > 0.7)} su {len(y_probs)}"
    )

    # Verifica se il modello è ancora indeciso
    confident_predictions = np.maximum(y_probs[:, 0], y_probs[:, 1]) > 0.7
    print(
        f"  Predizioni confident (>70%): {np.sum(confident_predictions)} su {len(y_probs)} ({np.sum(confident_predictions)/len(y_probs)*100:.1f}%)"
    )

    # Mostra alcuni esempi di probabilità
    print(f"\nPrimi 10 campioni (vere etichette vs probabilità):")
    for i in range(min(10, len(y_true))):
        true_label = labels[y_true[i]]
        prob_neg, prob_pos = y_probs[i, 0], y_probs[i, 1]
        pred_label = labels[y_pred[i]]
        confidence = max(prob_neg, prob_pos)
        status = "✅" if y_true[i] == y_pred[i] else "❌"
        print(
            f"  {i+1:2d}: True={true_label:<8} Pred={pred_label:<8} P(Neg)={prob_neg:.3f} P(Pos)={prob_pos:.3f} Conf={confidence:.3f} {status}"
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

    # Confronto con risultati precedenti
    print(f"\n" + "=" * 50)
    print("CONFRONTO CON RISULTATI PRECEDENTI")
    print("=" * 50)
    print("PRIMA (normalizzazione errata):")
    print("  - Tutte le probabilità ~50% (range 0.482-0.518)")
    print("  - Nessuna predizione Positive")
    print("  - Accuracy: 49.5%")
    print("")
    print("DOPO (normalizzazione corretta):")
    print(
        f"  - Range probabilità più ampio: Neg[{y_probs[:, 0].min():.3f}-{y_probs[:, 0].max():.3f}], Pos[{y_probs[:, 1].min():.3f}-{y_probs[:, 1].max():.3f}]"
    )
    print(f"  - Predizioni Positive: {np.sum(y_pred == 1)} su {len(y_pred)}")
    print(f"  - Accuracy: {accuracy*100:.1f}%")

    if np.sum(y_pred == 1) > 0:
        print("  ✅ SUCCESSO: Il modello ora predice entrambe le classi!")
    else:
        print("  ❌ PROBLEMA: Il modello ancora non predice Positive")


def main(args):
    """
    Funzione principale per il test dei golden labels con normalizzazione corretta
    """
    print("=" * 80)
    print("TEST GOLDEN LABELS CON NORMALIZZAZIONE CORRETTA")
    print("=" * 80)

    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Utilizzo dispositivo: {device}")

    # Percorsi dei dati
    landmarks_dir = os.path.join(
        BASE_DIR, "data", "raw", "ASLLRP", "mediapipe_output_golden_label", "json"
    )
    processed_file = os.path.join(
        BASE_DIR, "data", "processed", "golden_label_sentiment.csv"
    )

    print(f"Landmarks directory: {landmarks_dir}")
    print(f"Processed file: {processed_file}")

    # Carica il dataset con normalizzazione corretta
    print("\nCaricamento dataset con normalizzazione corretta...")
    test_dataset = FixedGoldenLabelDataset(
        landmarks_dir,
        processed_file,
        train_mean=159.878159,  # Dalle statistiche di training
        train_std=256.977173,
    )

    if len(test_dataset) == 0:
        print("ERRORE: Nessun campione trovato nel dataset!")
        return

    print(f"Dataset caricato: {len(test_dataset)} campioni")

    # Mostra statistiche del dataset corretto
    print(f"\nControllo rapido normalizzazione:")
    sample_features, sample_label = test_dataset[0]
    print(f"  Primo campione - Shape: {sample_features.shape}")
    print(f"  Range: [{sample_features.min():.1f}, {sample_features.max():.1f}]")
    print(f"  Media: {sample_features.mean():.1f} (target: ~160)")
    print(f"  Std: {sample_features.std():.1f} (target: ~257)")

    # Crea DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,  # Disabilitato per MPS
    )

    # Carica il modello
    print("\nCaricamento modello...")
    model = load_model_from_mlflow(args.model_uri)
    model = model.to(device)
    print("Modello caricato con successo!")

    # Valutazione
    print("\nEsecuzione valutazione...")
    y_pred, y_true, y_probs = evaluate_model_fixed(
        model, test_loader, device, test_dataset.labels
    )

    # Stampa risultati
    print_detailed_results_fixed(y_true, y_pred, y_probs, test_dataset.labels)

    # Salva risultati dettagliati in CSV
    if args.save_results:
        results_df = pd.DataFrame(
            {
                "video_name": test_dataset.processed["video_name"],
                "true_label": [test_dataset.labels[i] for i in y_true],
                "predicted_label": [test_dataset.labels[i] for i in y_pred],
                "confidence_negative": y_probs[:, 0],
                "confidence_positive": y_probs[:, 1],
                "max_confidence": np.maximum(y_probs[:, 0], y_probs[:, 1]),
                "correct_prediction": y_true == y_pred,
            }
        )

        output_file = os.path.join(
            BASE_DIR, "results", "golden_labels_test_results_fixed.csv"
        )
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        results_df.to_csv(output_file, index=False)
        print(f"\nRisultati dettagliati salvati in: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test del modello sui Golden Labels con normalizzazione corretta"
    )
    parser.add_argument(
        "--model_uri", type=str, required=True, help="URI del modello MLflow"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Dimensione del batch per l'inferenza",
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Salva i risultati dettagliati in un file CSV",
    )

    args = parser.parse_args()
    main(args)
