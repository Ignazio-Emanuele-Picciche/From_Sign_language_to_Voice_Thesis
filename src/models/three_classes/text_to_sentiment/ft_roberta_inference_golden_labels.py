"""
================================================================================
VALIDAZIONE SENTIMENT SU GOLDEN SET: ROBERTA FINE-TUNED
================================================================================

INPUT: golden_label_sentiment_with_neutral.csv
MODELLO: Modello Fine-Tunato Locale (src/models/.../finetuned_roberta_sentiment_mlflow)

DESCRIZIONE:
Questo script esegue l'inferenza sul dataset "Golden Label" utilizzando il
Tuo modello RoBERTa specifico (addestrato su ASL/How2Sign), invece del modello base.

STRUTTURA INPUT RICHIESTA:
- video_name: ID del video
- caption: Testo da analizzare
- emotion: Ground Truth (POSITIVE, NEGATIVE, NEUTRAL)
================================================================================
"""

import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
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
# Percorso del modello salvato dallo script di fine-tuning
MODEL_PATH = "models/finetuned_roberta_sentiment_mlflow"

BATCH_SIZE = 64

# Configurazione Device (Supporto Mac MPS / CUDA / CPU)
if torch.cuda.is_available():
    DEVICE = 0  # CUDA ID
    print("Using device: CUDA (GPU)")
elif torch.backends.mps.is_available():
    DEVICE = "mps"  # MPS ID
    print("Using device: MPS (Mac GPU)")
else:
    DEVICE = -1  # CPU
    print("Using device: CPU")

# Percorsi File
INPUT_FILE = "data/test/how2sign_sentiment_analyzed.csv"
OUTPUT_DIR = "golden_validation_finetuned/"


def load_pipeline():
    print(f"Caricamento modello fine-tunato da: {MODEL_PATH}...")

    # Verifica esistenza
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"❌ Modello non trovato in: {MODEL_PATH}\nHai eseguito lo script di training?"
        )

    # Carichiamo esplicitamente tokenizer e modello per sicurezza
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    # Creazione pipeline
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=DEVICE,
        max_length=512,
        truncation=True,
    )
    return sentiment_pipeline


def clean_label(label):
    """
    Normalizza l'etichetta del modello.
    RoBERTa base usa 'positive'/'negative', ma il fine-tuning potrebbe aver salvato
    LABEL_0/LABEL_1 se id2label non è stato salvato correttamente.
    Questo controllo gestisce entrambi i casi.
    """
    label = label.upper()

    # Mapping di sicurezza nel caso il modello restituisca LABEL_X
    # (Dipende dalla config salvata nel json)
    if label == "LABEL_0":
        return "NEGATIVE"
    if label == "LABEL_1":
        return "NEUTRAL"
    if label == "LABEL_2":
        return "POSITIVE"

    return label


def run_golden_validation_finetuned():
    # 1. CARICAMENTO DATASET
    print(f"\n--- 1. Caricamento Golden Label: {INPUT_FILE} ---")
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Errore: File non trovato in {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)

    # Verifica colonne minime
    # required_cols = {"video_name", "caption", "emotion"}  # golden label mapping
    required_cols = {"sentence_name", "sentence", "sentiment"}  # how2sign mapping
    if not required_cols.issubset(df.columns):
        print(f"❌ Errore: Il CSV deve contenere le colonne: {required_cols}")
        return

    print(f"✅ Dataset caricato: {len(df)} righe.")

    # 2. INFERENZA
    print("\n--- 2. Esecuzione Inferenza (RoBERTa Fine-Tuned) ---")

    try:
        sentiment_pipeline = load_pipeline()
    except Exception as e:
        print(f"Errore caricamento modello: {e}")
        return

    texts = df["sentence"].astype(str).tolist()
    predictions = []

    # Inferenza in batch
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[i : i + BATCH_SIZE]

        # Gestione errori su batch specifici
        try:
            results = sentiment_pipeline(batch_texts)
            for res in results:
                predictions.append(clean_label(res["label"]))
        except Exception as e:
            print(f"Errore nel batch {i}: {e}")
            # Fallback: riempiamo con 'NEUTRAL' per non rompere l'allineamento
            predictions.extend(["NEUTRAL"] * len(batch_texts))

    # Aggiungiamo i risultati al DataFrame
    df["predicted_sentiment"] = predictions

    # 3. SALVATAGGIO CSV DI CONFRONTO
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_csv = os.path.join(OUTPUT_DIR, "finetuned_golden_comparison.csv")
    df.to_csv(output_csv, index=False)
    print(f"✅ File risultati salvato: {output_csv}")

    # 4. CALCOLO METRICHE
    print("\n--- 3. Calcolo Performance ---")

    # Normalizzazione Ground Truth (Rimuove spazi e mette in uppercase)
    y_true = df["sentiment"].astype(str).str.upper().str.strip()
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
        f"GOLDEN LABEL VALIDATION REPORT (Fine-Tuned RoBERTa)\n"
        f"========================================\n"
        f"Input File: {os.path.basename(INPUT_FILE)}\n"
        f"Model Path: {MODEL_PATH}\n"
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

    output_txt = os.path.join(OUTPUT_DIR, "finetuned_golden_metrics.txt")
    with open(output_txt, "w") as f:
        f.write(metrics_text)

    print(f"✅ Report metriche salvato: {output_txt}")
    print("-" * 30)
    print(metrics_text)

    # 5. SALVATAGGIO DISACCORDI
    disagreements = df[y_true != y_pred]
    output_err = os.path.join(OUTPUT_DIR, "finetuned_golden_errors.csv")
    disagreements.to_csv(output_err, index=False)
    print(f"✅ Errori salvati per analisi ({len(disagreements)} righe): {output_err}")


if __name__ == "__main__":
    run_golden_validation_finetuned()
