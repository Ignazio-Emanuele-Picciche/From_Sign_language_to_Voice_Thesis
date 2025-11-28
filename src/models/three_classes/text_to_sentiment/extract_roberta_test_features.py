import pandas as pd
import torch
import os
import sys
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# CONFIGURAZIONE
TEST_FILE = "data/test/golden_test_set.csv"
MODEL_PATH = "models/finetuned_roberta_sentiment_mlflow"  # Il tuo modello salvato
OUTPUT_FILE = os.path.join("results", "roberta_features_test_set.csv")


def main():
    print("=" * 60)
    print("ESTRAZIONE FEATURE ROBERTA (TEST SET)")
    print("=" * 60)

    # 1. Setup Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # 2. Caricamento Dati
    input_path = TEST_FILE
    if not os.path.exists(input_path):
        print(f"❌ Errore: File non trovato: {input_path}")
        return

    df = pd.read_csv(input_path)
    print(f"Caricato Test Set: {len(df)} righe")

    # 3. Caricamento Modello
    full_model_path = MODEL_PATH
    print(f"Caricamento modello da: {full_model_path}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(full_model_path)
        model = AutoModelForSequenceClassification.from_pretrained(full_model_path)
    except OSError:
        print(f"❌ Errore: Modello non trovato in {full_model_path}")
        print("Assicurati di aver eseguito il fine-tuning di RoBERTa.")
        return

    model.to(device)
    model.eval()

    # 4. Inferenza
    all_probs = []
    captions = df["caption"].astype(str).tolist()
    batch_size = 32

    print("Inizio inferenza...")
    with torch.no_grad():
        for i in tqdm(range(0, len(captions), batch_size)):
            batch = captions[i : i + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(device)
            outputs = model(**inputs)
            all_probs.append(torch.softmax(outputs.logits, dim=1).cpu().numpy())

    probs = np.vstack(all_probs)

    # 5. Salvataggio Risultati
    results_df = pd.DataFrame(
        {"video_name": df["video_name"], "true_label": df["emotion"].str.upper()}
    )

    # Recupero nomi label (o fallback)
    id2label = (
        model.config.id2label
        if hasattr(model.config, "id2label")
        else {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
    )

    for idx in range(probs.shape[1]):
        # Pulizia nomi colonne (es. LABEL_0 -> negative)
        label_name = id2label.get(idx, f"LABEL_{idx}").lower()
        if "label_" in label_name:
            label_name = ["negative", "neutral", "positive"][idx]

        results_df[f"roberta_prob_{label_name}"] = probs[:, idx]

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    results_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ Features RoBERTa salvate in: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
