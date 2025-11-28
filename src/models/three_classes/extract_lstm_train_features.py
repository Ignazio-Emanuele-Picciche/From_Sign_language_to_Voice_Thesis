#  experiment id cdadbf4c393f487aae9d08521611e805
#  PREDITE TUTTE LE CLASSI
# python src/models/three_classes/lstm_softmax_vector.py --model_uri mlartifacts/816580482732370733/models/m-0f27528663e84dfe91e0b2c7f4a15495/artifacts
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
# CONFIGURAZIONE DEI DUE DATASET
# Qui definiamo le coppie (CSV, Cartella JSON) per processarli separatamente
# ==============================================================================
DATASETS_CONFIG = [
    {
        "name": "ASLLRP",
        "csv": "data/processed/asllrp_video_sentiment_data_with_neutral_0.34_without_golden.csv",
        "json_dir": "data/raw/ASLLRP/mediapipe_output_with_neutral_0.34/json",
    },
    {
        "name": "How2Sign",
        "csv": "data/processed/train/video_sentiment_data_with_neutral_0.34.csv",
        "json_dir": "data/raw/train/openpose_output_train/json",
    },
]


def extract_softmax_vectors(model, dataloader, device, labels_map):
    """
    Esegue l'inferenza pura e raccoglie solo i vettori softmax.
    Restituisce un dizionario di liste.
    """
    model.eval()

    results = {"true_label": []}
    for label in labels_map:
        results[f"lstm_prob_{label.lower()}"] = []

    with torch.no_grad():
        for i, (batch_data, batch_labels) in enumerate(dataloader):
            # Preparazione input
            inputs = batch_data.float().to(device)

            # Inferenza
            outputs = model(inputs)

            # Softmax
            probs = torch.softmax(outputs, dim=1)

            # Raccolta dati
            current_probs = probs.cpu().numpy()
            current_labels = batch_labels.numpy()

            results["true_label"].extend([labels_map[l] for l in current_labels])

            for idx, label_name in enumerate(labels_map):
                results[f"lstm_prob_{label_name.lower()}"].extend(current_probs[:, idx])

    return results


def process_single_dataset(config, model, device, batch_size):
    """
    Processa una singola coppia CSV/JSON Directory
    """
    print(f"\n--- Processando dataset: {config['name']} ---")

    csv_path = os.path.join(BASE_DIR, config["csv"])
    json_dir = os.path.join(BASE_DIR, config["json_dir"])

    # Check esistenza file
    if not os.path.exists(csv_path):
        # Fallback: prova a cercarli in data/processed
        csv_path = csv_path.replace("/train/", "/processed/")
        if not os.path.exists(csv_path):
            print(f"❌ ERRORE: CSV non trovato: {csv_path}")
            return None

    if not os.path.exists(json_dir):
        print(f"❌ ERRORE: Directory JSON non trovata: {json_dir}")
        return None

    print(f"  CSV: {csv_path}")
    print(f"  JSON Dir: {json_dir}")

    # Inizializzazione Dataset
    # Usiamo le statistiche di training fisse per coerenza
    dataset = FixedGoldenLabelDataset(
        json_dir,  # 1° posizionale: cartella JSON specifica
        csv_path,  # 2° posizionale: file CSV specifico
        train_mean=159.878159,
        train_std=256.977173,
    )

    print(f"  Campioni trovati: {len(dataset)}")

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # Estrazione features
    results = extract_softmax_vectors(model, dataloader, device, dataset.labels)

    # Aggiungi video names
    results["video_name"] = dataset.processed["video_name"].tolist()

    # Converti in DataFrame parziale
    df = pd.DataFrame(results)
    return df


def main(args):
    print("=" * 60)
    print("ESTRAZIONE FEATURE LSTM (SOFTMAX) - MULTI DATASET")
    print("=" * 60)

    # 1. Setup Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Dispositivo: {device}")

    # 2. Caricamento Modello (una volta sola)
    print(f"\nCaricamento modello da: {args.model_uri}")
    model = mlflow.pytorch.load_model(args.model_uri)
    model = model.to(device)

    all_dataframes = []

    # 3. Loop sui dataset
    for config in DATASETS_CONFIG:
        df_part = process_single_dataset(config, model, device, args.batch_size)
        if df_part is not None:
            all_dataframes.append(df_part)

    if not all_dataframes:
        print("❌ Nessun dato estratto. Esco.")
        return

    # 4. Concatenazione finale
    print("\nConcatenazione risultati...")
    final_df = pd.concat(all_dataframes, ignore_index=True)

    # Riordino colonne
    cols = ["video_name", "true_label"] + [c for c in final_df.columns if "prob" in c]
    final_df = final_df[cols]

    # 5. Salvataggio
    output_path = os.path.join(BASE_DIR, "results", "lstm_features_train_set.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    final_df.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print(f"✅ ESTRAZIONE COMPLETATA CON SUCCESSO")
    print(f"📄 File salvato in: {output_path}")
    print(f"📊 Totale righe: {len(final_df)}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_uri", type=str, required=True, help="URI del modello MLflow"
    )
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()
    main(args)
