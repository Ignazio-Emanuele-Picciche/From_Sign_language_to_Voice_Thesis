"""
================================================================================
VALIDAZIONE SENTIMENT SU GOLDEN SET: ROBERTA (CardiffNLP)
================================================================================

INPUT: golden_label_sentiment_with_neutral.csv
MODELLO: cardiffnlp/twitter-roberta-base-sentiment-latest

DESCRIZIONE:
Questo script esegue l'inferenza sul dataset "Golden Label" utilizzando il modello
RoBERTa, che è lo stato dell'arte per la sentiment analysis su 3 classi.
Poiché il modello restituisce nativamente 'positive', 'negative', 'neutral',
il processo è diretto e non richiede mapping complessi di emozioni.

STRUTTURA INPUT RICHIESTA:
- video_name: ID del video
- caption: Testo da analizzare
- emotion: Ground Truth (POSITIVE, NEGATIVE, NEUTRAL)
================================================================================
"""

import pandas as pd
import torch
from transformers import pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    balanced_accuracy_score,
    confusion_matrix,
    cohen_kappa_score,
)
from tqdm import tqdm
import os

# --- CONFIGURAZIONE ---
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
BATCH_SIZE = 16
DEVICE = 0 if torch.cuda.is_available() else -1

# Percorsi
INPUT_FILE = "data/processed/golden_test_set.csv"
OUTPUT_DIR = "src/models/three_classes/text_to_sentiment/golden_validation/"

print(f"Using device: {'GPU' if DEVICE == 0 else 'CPU'}")
print(f"Caricamento modello {MODEL_NAME}...")

# Pipeline Sentiment Analysis
# Questo modello restituisce label: "positive", "negative", "neutral"
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=MODEL_NAME,
    device=DEVICE,
    tokenizer=MODEL_NAME,
    max_length=512,
    truncation=True,
)


def clean_label(label):
    """
    Normalizza l'etichetta del modello (lowercase) per il confronto (UPPERCASE).
    Es: 'positive' -> 'POSITIVE'
    """
    return label.upper()


def run_golden_validation_roberta():
    # 1. CARICAMENTO DATASET
    print(f"\n--- 1. Caricamento Golden Label: {INPUT_FILE} ---")
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Errore: File non trovato in {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)

    # Verifica colonne minime
    required_cols = {"video_name", "caption", "emotion"}
    if not required_cols.issubset(df.columns):
        print(f"❌ Errore: Il CSV deve contenere le colonne: {required_cols}")
        return

    print(f"✅ Dataset caricato: {len(df)} righe.")

    # 2. INFERENZA
    print("\n--- 2. Esecuzione Inferenza (RoBERTa) ---")
    texts = df["caption"].astype(str).tolist()
    predictions = []

    # Inferenza in batch
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[i : i + BATCH_SIZE]
        results = sentiment_pipeline(batch_texts)

        for res in results:
            # res è un dizionario {'label': 'neutral', 'score': 0.98}
            label = res["label"]
            predictions.append(clean_label(label))

    # Aggiungiamo i risultati al DataFrame
    df["predicted_sentiment"] = predictions

    # 3. SALVATAGGIO CSV DI CONFRONTO
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_csv = os.path.join(OUTPUT_DIR, "roberta_golden_comparison.csv")
    df.to_csv(output_csv, index=False)
    print(f"✅ File risultati salvato: {output_csv}")

    # 4. CALCOLO METRICHE
    print("\n--- 3. Calcolo Performance ---")

    # Normalizzazione Ground Truth (Rimuove spazi e mette in uppercase)
    y_true = df["emotion"].astype(str).str.upper().str.strip()
    y_pred = df["predicted_sentiment"]
    labels_order = ["NEGATIVE", "NEUTRAL", "POSITIVE"]

    # Calcolo Metriche
    acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    report = classification_report(y_true, y_pred, target_names=labels_order)

    # Creazione Report Testuale
    metrics_text = (
        f"GOLDEN LABEL VALIDATION REPORT (RoBERTa)\n"
        f"========================================\n"
        f"Input File: {os.path.basename(INPUT_FILE)}\n"
        f"Model: {MODEL_NAME}\n"
        f"Total Samples: {len(df)}\n\n"
        f"PERFORMANCE METRICS:\n"
        f"-----------------\n"
        f"Accuracy:                   {acc:.4f}\n"
        f"Balanced Accuracy:          {balanced_acc:.4f}\n"
        f"Cohen's Kappa:              {kappa:.4f}\n"
        f"F1 Score (Weighted):        {f1_weighted:.4f}\n\n"
        f"CONFUSION MATRIX (Rows=True, Cols=Pred):\n"
        f"Labels: {labels_order}\n"
        f"{cm}\n\n"
        f"DETAILED REPORT:\n"
        f"{report}\n"
    )

    output_txt = os.path.join(OUTPUT_DIR, "roberta_golden_metrics.txt")
    with open(output_txt, "w") as f:
        f.write(metrics_text)

    print(f"✅ Report metriche salvato: {output_txt}")
    print("-" * 30)
    print(metrics_text)

    # 5. SALVATAGGIO DISACCORDI
    disagreements = df[y_true != y_pred]
    output_err = os.path.join(OUTPUT_DIR, "roberta_golden_errors.csv")
    disagreements.to_csv(output_err, index=False)
    print(f"✅ Errori salvati per analisi ({len(disagreements)} righe): {output_err}")


if __name__ == "__main__":
    run_golden_validation_roberta()
