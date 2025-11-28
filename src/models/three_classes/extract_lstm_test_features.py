"""
================================================================================
ESTRAZIONE FEATURE LSTM (SOFTMAX) - TEST SET (GOLDEN LABELS)
================================================================================
Basato su: extract_lstm_train_features.py
Obiettivo: Generare i vettori di probabilità per il Meta-Learner usando il Golden Set.
"""

# python src/models/three_classes/extract_lstm_test_features.py --model_uri mlartifacts/816580482732370733/models/m-0f27528663e84dfe91e0b2c7f4a15495/artifacts


import torch
import numpy as np
import pandas as pd
import os
import sys
import argparse
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader

# --- Setup Percorsi ---
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
sys.path.insert(0, BASE_DIR)

from src.utils.three_classes.training_utils import prepare_batch
from src.models.three_classes.fix_golden_labels_normalization import (
    FixedGoldenLabelDataset,
)
import src.models.three_classes.lstm_model as lstm_model_module

sys.modules["src.models.three_classes.lstm_model"] = lstm_model_module

# ==============================================================================
# CONFIGURAZIONE TEST SET
# ==============================================================================
TEST_CONFIG = {
    "name": "Golden Test Set",
    # Percorso standard del CSV di test
    "csv": "data/processed/golden_test_set.csv",
    # Percorso standard dei JSON di test (Golden Label)
    "json_dir": "data/raw/ASLLRP/mediapipe_output_golden_label/json",
}

# Output file atteso dal Meta-Learner
OUTPUT_FILENAME = "src/models/three_classes/text_plus_video_metalearner_to_sentiment/data/test/lstm_features_test_set.csv"


def extract_softmax_vectors(model, dataloader, device, labels_map):
    """
    Esegue l'inferenza pura e raccoglie solo i vettori softmax.
    """
    model.eval()

    results = {"true_label": []}
    for label in labels_map:
        results[f"lstm_prob_{label.lower()}"] = []

    print("Inizio estrazione vettori Softmax...")

    with torch.no_grad():
        for i, (batch_data, batch_labels) in enumerate(dataloader):
            # Preparazione input
            inputs = batch_data.float().to(device)

            # Inferenza
            outputs = model(inputs)

            # Softmax (trasforma logits in probabilità 0-1)
            probs = torch.softmax(outputs, dim=1)

            # Raccolta dati
            current_probs = probs.cpu().numpy()
            current_labels = batch_labels.numpy()

            results["true_label"].extend([labels_map[l] for l in current_labels])

            for idx, label_name in enumerate(labels_map):
                results[f"lstm_prob_{label_name.lower()}"].extend(current_probs[:, idx])

    return results


def main(args):
    print("=" * 60)
    print("ESTRAZIONE FEATURE LSTM (SOFTMAX) - TEST SET")
    print("=" * 60)

    # 1. Setup Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Dispositivo: {device}")

    # 2. Setup Percorsi Dati
    csv_path = os.path.join(BASE_DIR, TEST_CONFIG["csv"])
    json_dir = os.path.join(BASE_DIR, TEST_CONFIG["json_dir"])

    print(f"Dataset: {TEST_CONFIG['name']}")
    print(f"  CSV: {csv_path}")
    print(f"  JSON Dir: {json_dir}")

    if not os.path.exists(csv_path) or not os.path.exists(json_dir):
        print("❌ ERRORE: File di input non trovati. Controlla i percorsi.")
        return

    # 3. Inizializzazione Dataset
    # IMPORTANTE: Usiamo train_mean e train_std fissi per coerenza con il modello addestrato
    dataset = FixedGoldenLabelDataset(
        json_dir,
        csv_path,
        train_mean=159.878159,
        train_std=256.977173,
    )
    print(f"  Campioni trovati: {len(dataset)}")

    # Shuffle=False è fondamentale per mantenere l'ordine allineato con il CSV finale
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # 4. Caricamento Modello
    print(f"\nCaricamento modello da: {args.model_uri}")
    model = mlflow.pytorch.load_model(args.model_uri)
    model = model.to(device)

    # 5. Estrazione Features
    results_dict = extract_softmax_vectors(model, dataloader, device, dataset.labels)

    # Aggiungi video names (chiave primaria per il join)
    results_dict["video_name"] = dataset.processed["video_name"].tolist()

    # Creazione DataFrame
    final_df = pd.DataFrame(results_dict)

    # Riordino colonne: Video -> Label -> Probabilità
    cols = ["video_name", "true_label"] + [c for c in final_df.columns if "prob" in c]
    final_df = final_df[cols]

    # 6. Salvataggio
    output_path = OUTPUT_FILENAME
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    final_df.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print(f"✅ ESTRAZIONE COMPLETATA")
    print(f"📄 File salvato in: {output_path}")
    print(f"📊 Shape: {final_df.shape}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_uri", type=str, required=True, help="URI del modello MLflow"
    )
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()
    main(args)
