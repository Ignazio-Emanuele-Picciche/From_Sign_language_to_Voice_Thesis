# python src/models/finetuned_roberta/extract_roberta_train_features.py --model_path models/finetuned_roberta_sentiment_mlflow

import pandas as pd
import torch
import os
import sys
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- CONFIGURAZIONE DATASET ---
TRAIN_FILE_1 = (
    "data/train/asllrp_video_sentiment_data_with_neutral_0.34_without_golden.csv"
)
TRAIN_FILE_2 = "data/train/video_sentiment_data_with_neutral_0.34.csv"


def load_and_combine_data():
    """Carica e unisce i due CSV di training."""
    path1 = TRAIN_FILE_1
    path2 = TRAIN_FILE_2

    print(f"1. Caricamento file: {path1}")
    df1 = pd.read_csv(path1)
    print(f"2. Caricamento file: {path2}")
    df2 = pd.read_csv(path2)

    combined_df = pd.concat([df1, df2], ignore_index=True)

    # Verifica colonne necessarie
    required = ["video_name", "caption", "emotion"]  # emotion è la ground truth
    if not set(required).issubset(combined_df.columns):
        raise ValueError(f"I CSV devono contenere le colonne: {required}")

    print(f"   Totale campioni combinati: {len(combined_df)}")
    return combined_df


def get_label_mapping(model_config):
    """
    Determina il mapping ID -> Label.
    Di solito: 0: Negative, 1: Neutral, 2: Positive.
    """
    # Se il modello ha un mapping salvato, usiamo quello
    if hasattr(model_config, "id2label") and model_config.id2label:
        return model_config.id2label

    # Fallback standard (basato sul tuo codice precedente)
    print("⚠️  Mapping non trovato nel config, uso default: 0=NEG, 1=NEU, 2=POS")
    return {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}


def extract_roberta_features(model_path, df, batch_size=32):
    """
    Esegue l'inferenza e estrae le probabilità softmax.
    """
    # Setup Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Utilizzo dispositivo: {device}")

    # Caricamento Tokenizer e Modello
    print(f"Caricamento modello da: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # Recuperiamo il mapping delle label per nominare le colonne correttamente
    id2label = get_label_mapping(model.config)

    # Prepariamo le liste per i risultati
    all_probs = []

    # Creiamo batch
    captions = df["caption"].astype(str).tolist()
    total_batches = (len(captions) + batch_size - 1) // batch_size

    print("Inizio estrazione feature RoBERTa...")

    with torch.no_grad():
        for i in tqdm(range(0, len(captions), batch_size), total=total_batches):
            batch_texts = captions[i : i + batch_size]

            # Tokenizzazione
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(device)

            # Inferenza
            outputs = model(**inputs)

            # Calcolo Softmax
            probs = torch.softmax(outputs.logits, dim=1)
            all_probs.append(probs.cpu().numpy())

    # Concateniamo tutti i batch di probabilità (N_samples, 3)
    final_probs = np.vstack(all_probs)

    return final_probs, id2label


import numpy as np  # Assicuriamoci che numpy sia importato


def main(args):
    print("=" * 60)
    print("ESTRAZIONE FEATURE ROBERTA (SOFTMAX) DAL TRAINING SET")
    print("=" * 60)

    # 1. Caricamento Dati
    try:
        df = load_and_combine_data()
    except Exception as e:
        print(f"❌ Errore caricamento dati: {e}")
        return

    # 2. Estrazione
    probs, id2label = extract_roberta_features(args.model_path, df, args.batch_size)

    # 3. Costruzione DataFrame Finale
    # Creiamo un nuovo DF partendo da video_name e true_label
    results_df = pd.DataFrame(
        {
            "video_name": df["video_name"],
            "true_label": df["emotion"].str.upper(),  # Standardizziamo
        }
    )

    # Aggiungiamo le colonne di probabilità mappate correttamente
    # id2label es: {0: 'NEGATIVE', 1: 'NEUTRAL', 2: 'POSITIVE'}
    # Assumiamo che l'indice della colonna di probs corrisponda alla key di id2label
    for idx in range(probs.shape[1]):
        label_name = id2label.get(idx, f"LABEL_{idx}")
        # Pulizia nome: NEGATIVE -> negative
        clean_name = label_name.lower()
        if "label_" in clean_name:
            # Fallback se il mapping fallisce
            if idx == 0:
                clean_name = "negative"
            elif idx == 1:
                clean_name = "neutral"
            elif idx == 2:
                clean_name = "positive"

        col_name = f"roberta_prob_{clean_name}"
        results_df[col_name] = probs[:, idx]

    # 4. Salvataggio
    output_path = os.path.join("results", "roberta_features_train_set.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Riordino colonne per sicurezza
    cols = ["video_name", "true_label"] + [
        c for c in results_df.columns if "roberta_prob" in c
    ]
    results_df = results_df[cols]

    results_df.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print(f"✅ ESTRAZIONE COMPLETATA")
    print(f"📄 File salvato in: {output_path}")
    print(f"📊 Shape: {results_df.shape}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estrai vettori Softmax da RoBERTa")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/finetuned_roberta_sentiment_mlflow",
        help="Percorso locale del modello RoBERTa salvato",
    )
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()
    main(args)
