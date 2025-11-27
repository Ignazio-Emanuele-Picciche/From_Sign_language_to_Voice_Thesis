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

# Importiamo le classi necessarie
from src.utils.three_classes.training_utils import prepare_batch
from src.models.three_classes.fix_golden_labels_normalization import (
    FixedGoldenLabelDataset,
)

# Import necessario per permettere a MLflow di caricare la classe (pickle fix)
import src.models.three_classes.lstm_model as lstm_model_module

sys.modules["src.models.three_classes.lstm_model"] = lstm_model_module

# --- CONFIGURAZIONE DATASET ---
# Nota: I percorsi sono relativi a BASE_DIR
TRAIN_FILE_1 = (
    "data/processed/asllrp_video_sentiment_data_with_neutral_0.34_without_golden.csv"
)
TRAIN_FILE_2 = "data/processed/train/video_sentiment_data_with_neutral_0.34.csv"

# Path ai JSON del training set (ASLLRP originale)
DEFAULT_LANDMARKS_DIR = os.path.join(
    BASE_DIR, "data", "raw", "ASLLRP", "mediapipe_output_json"
)


def load_and_combine_data():
    """Carica e unisce i due CSV di training."""
    path1 = os.path.join(BASE_DIR, TRAIN_FILE_1)
    path2 = os.path.join(BASE_DIR, TRAIN_FILE_2)

    # Verifica esistenza file per evitare crash brutali
    if not os.path.exists(path1):
        # Fallback: prova a cercarli in data/processed se non sono in data/train
        path1 = path1.replace("/train/", "/processed/")
        if not os.path.exists(path1):
            print(f"⚠️ ATTENZIONE: File non trovato: {path1}")

    if not os.path.exists(path2):
        path2 = path2.replace("/train/", "/processed/")
        if not os.path.exists(path2):
            print(f"⚠️ ATTENZIONE: File non trovato: {path2}")

    print(f"1. Caricamento file: {path1}")
    df1 = pd.read_csv(path1)
    print(f"2. Caricamento file: {path2}")
    df2 = pd.read_csv(path2)

    combined_df = pd.concat([df1, df2], ignore_index=True)
    print(f"   Totale campioni combinati: {len(combined_df)}")

    # Salviamo un file temporaneo per passarlo al Dataset
    temp_path = os.path.join(
        BASE_DIR, "data", "processed", "temp_combined_train_for_features.csv"
    )
    combined_df.to_csv(temp_path, index=False)
    return temp_path


def extract_softmax_vectors(model, dataloader, device, labels_map):
    """
    Esegue l'inferenza pura e raccoglie solo i vettori softmax.
    """
    model.eval()

    # Dizionario per raccogliere i dati
    results = {"true_label": []}
    # Inizializza liste per le probabilità (es. lstm_prob_negative, etc.)
    for label in labels_map:
        results[f"lstm_prob_{label.lower()}"] = []

    print("Inizio estrazione vettori Softmax...")

    with torch.no_grad():
        for i, (batch_data, batch_labels) in enumerate(dataloader):
            if i % 20 == 0:  # Feedback più frequente
                print(f"  Processando batch {i}/{len(dataloader)}...")

            # Preparazione input
            inputs = batch_data.float().to(device)

            # --- INFERENZA ---
            outputs = model(inputs)

            # --- ESTRAZIONE PROBABILITÀ (SOFTMAX) ---
            probs = torch.softmax(outputs, dim=1)

            # Spostiamo su CPU per salvare
            current_probs = probs.cpu().numpy()
            current_labels = batch_labels.numpy()

            # Raccogliamo le label reali
            results["true_label"].extend([labels_map[l] for l in current_labels])

            # Raccogliamo le probabilità colonna per colonna
            for idx, label_name in enumerate(labels_map):
                results[f"lstm_prob_{label_name.lower()}"].extend(current_probs[:, idx])

    return results


def main(args):
    print("=" * 60)
    print("ESTRAZIONE FEATURE LSTM (SOFTMAX) DAL TRAINING SET")
    print("=" * 60)

    # 1. Setup Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Dispositivo: {device}")

    # 2. Preparazione Dati (Concatenazione)
    try:
        temp_csv_path = load_and_combine_data()
    except FileNotFoundError as e:
        print(
            f"❌ ERRORE: File di dati non trovato. Controlla i percorsi in alto nello script."
        )
        return

    # Usiamo FixedGoldenLabelDataset
    print("\nInizializzazione Dataset...")

    # --- CORREZIONE QUI SOTTO: Argomenti posizionali invece di keyword ---
    dataset = FixedGoldenLabelDataset(
        args.landmarks_dir,  # 1° Argomento: directory json
        temp_csv_path,  # 2° Argomento: file csv
        train_mean=159.878159,
        train_std=256.977173,
    )
    # ---------------------------------------------------------------------

    # IMPORTANTE: shuffle=False per mantenere l'ordine esatto dei video_name
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # Recuperiamo i video names
    video_names = dataset.processed["video_name"].tolist()

    # 3. Caricamento Modello MLflow
    print(f"\nCaricamento modello da: {args.model_uri}")
    model = mlflow.pytorch.load_model(args.model_uri)
    model = model.to(device)

    # 4. Estrazione Vettori
    results_dict = extract_softmax_vectors(model, dataloader, device, dataset.labels)

    # Verifica di sicurezza
    if len(video_names) != len(results_dict["true_label"]):
        print(
            "❌ ERRORE CRITICO: Il numero di video non corrisponde al numero di predizioni!"
        )
        print(f"  Video nel dataset: {len(video_names)}")
        print(f"  Predizioni effettuate: {len(results_dict['true_label'])}")
        return

    # Aggiungiamo la chiave primaria
    results_dict["video_name"] = video_names

    # 5. Creazione DataFrame Finale
    df_results = pd.DataFrame(results_dict)

    # Riordiniamo le colonne
    cols = ["video_name", "true_label"] + [c for c in df_results.columns if "prob" in c]
    df_results = df_results[cols]

    # Salvataggio
    output_path = os.path.join(BASE_DIR, "results", "lstm_features_train_set.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_results.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print(f"✅ ESTRAZIONE COMPLETATA")
    print(f"📄 File salvato in: {output_path}")
    print(f"📊 Shape del dataset: {df_results.shape}")
    print("=" * 60)

    # Pulizia file temporaneo
    if os.path.exists(temp_csv_path):
        os.remove(temp_csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estrai vettori Softmax (Features) dai file di Training"
    )
    parser.add_argument(
        "--model_uri", type=str, required=True, help="URI del modello MLflow"
    )
    parser.add_argument(
        "--landmarks_dir",
        type=str,
        default=DEFAULT_LANDMARKS_DIR,
        help="Path alla cartella JSON dei landmarks (Training)",
    )
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()
    main(args)
