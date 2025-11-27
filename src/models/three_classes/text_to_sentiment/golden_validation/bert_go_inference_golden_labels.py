"""
================================================================================
VALIDAZIONE SENTIMENT SU GOLDEN SET: BERT-GO-EMOTION
================================================================================

INPUT: golden_label_sentiment_with_neutral.csv
MODELLO: bhadresh-savani/bert-base-go-emotion (28 classi -> 3 Sentiment)

DESCRIZIONE:
Questo script esegue l'inferenza su un dataset "Golden Label" (riferimento affidabile)
utilizzando un modello addestrato su 28 emozioni.
Le 28 emozioni vengono mappate in 3 macro-categorie (Positive, Negative, Neutral)
e confrontate con l'etichetta presente nel file per calcolare le metriche di performance.

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
MODEL_NAME = "bhadresh-savani/bert-base-go-emotion"
BATCH_SIZE = 16
DEVICE = 0 if torch.cuda.is_available() else -1

# Percorsi
INPUT_FILE = "data/processed/golden_label_sentiment_with_neutral.csv"
OUTPUT_DIR = "src/models/three_classes/text_to_sentiment/golden_validation/"

print(f"Using device: {'GPU' if DEVICE == 0 else 'CPU'}")
print(f"Caricamento modello {MODEL_NAME}...")

# Pipeline Text Classification (Top-1 emotion)
emotion_pipeline = pipeline(
    "text-classification",
    model=MODEL_NAME,
    device=DEVICE,
    tokenizer=MODEL_NAME,
    max_length=512,
    truncation=True,
    top_k=1,
)


def map_emotion_to_sentiment(label):
    """
    Mappa le 28 classi di GoEmotions nelle 3 classi di Sentiment.
    """
    positive_emotions = {
        "admiration",
        "amusement",
        "approval",
        "caring",
        "desire",
        "excitement",
        "gratitude",
        "joy",
        "love",
        "optimism",
        "pride",
        "relief",
    }

    negative_emotions = {
        "anger",
        "annoyance",
        "disappointment",
        "disapproval",
        "disgust",
        "embarrassment",
        "fear",
        "grief",
        "nervousness",
        "remorse",
        "sadness",
    }

    # 'surprise', 'curiosity', 'confusion', 'realization' + 'neutral' -> NEUTRAL
    if label in positive_emotions:
        return "POSITIVE"
    elif label in negative_emotions:
        return "NEGATIVE"
    else:
        return "NEUTRAL"


def run_golden_validation():
    # 1. CARICAMENTO DATASET
    print(f"\n--- 1. Caricamento Golden Label: {INPUT_FILE} ---")
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Errore: File non trovato in {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)

    # Verifica colonne
    required_cols = {"video_name", "caption", "emotion"}
    if not required_cols.issubset(df.columns):
        print(f"❌ Errore: Il CSV deve contenere le colonne: {required_cols}")
        print(f"   Colonne trovate: {df.columns.tolist()}")
        return

    print(f"✅ Dataset caricato: {len(df)} righe.")

    # 2. INFERENZA
    print("\n--- 2. Esecuzione Inferenza (BERT-GoEmotion) ---")
    texts = df["caption"].astype(str).tolist()

    raw_emotions = []
    mapped_sentiments = []

    # Eseguiamo a batch
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[i : i + BATCH_SIZE]
        results = emotion_pipeline(batch_texts)

        for res in results:
            # Estrazione label (gestione formato lista o dict)
            top_prediction = res[0] if isinstance(res, list) else res
            label = top_prediction["label"]

            raw_emotions.append(label)
            mapped_sentiments.append(map_emotion_to_sentiment(label))

    # Aggiungiamo i risultati al DataFrame
    df["predicted_emotion_raw"] = raw_emotions
    df["predicted_sentiment"] = mapped_sentiments

    # 3. SALVATAGGIO CSV DI CONFRONTO
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_csv = os.path.join(OUTPUT_DIR, "golden_comparison_results.csv")
    df.to_csv(output_csv, index=False)
    print(f"✅ File risultati salvato: {output_csv}")

    # 4. CALCOLO METRICHE
    print("\n--- 3. Calcolo Performance ---")

    # Normalizzazione (assicuriamoci che GT e PRED siano uppercase e senza spazi)
    y_true = df["emotion"].astype(str).str.upper().str.strip()
    y_pred = df["predicted_sentiment"]
    labels_order = ["NEGATIVE", "NEUTRAL", "POSITIVE"]

    # Calcolo
    acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    report = classification_report(y_true, y_pred, target_names=labels_order)

    # Creazione Report Testuale
    metrics_text = (
        f"GOLDEN LABEL VALIDATION REPORT\n"
        f"========================================\n"
        f"Input File: {os.path.basename(INPUT_FILE)}\n"
        f"Model: {MODEL_NAME} (Mapped 28->3)\n"
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

    output_txt = os.path.join(OUTPUT_DIR, "golden_metrics.txt")
    with open(output_txt, "w") as f:
        f.write(metrics_text)

    print(f"✅ Report metriche salvato: {output_txt}")
    print("-" * 30)
    print(metrics_text)

    # 5. SALVATAGGIO DISACCORDI (Analisi Errori)
    disagreements = df[y_true != y_pred]
    output_err = os.path.join(OUTPUT_DIR, "golden_errors.csv")
    disagreements.to_csv(output_err, index=False)
    print(f"✅ Errori salvati per analisi ({len(disagreements)} righe): {output_err}")


if __name__ == "__main__":
    run_golden_validation()
