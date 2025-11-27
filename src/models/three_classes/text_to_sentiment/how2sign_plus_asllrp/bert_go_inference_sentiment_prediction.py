"""
================================================================================
PIPELINE DI VALIDAZIONE: VADER vs BERT-GO-EMOTION (28 Classes -> 3 Sentiment)
================================================================================

AUTORE: [Tuo Nome]
DATA: Novembre 2025
CONTESTO: Text-to-Sentiment su dataset di Lingua dei Segni (How2Sign + ASLLRP)

--------------------------------------------------------------------------------
1. OBIETTIVO DELL'ANALISI
--------------------------------------------------------------------------------
Confrontiamo l'annotazione "Rule-based" di VADER (Proxy Ground Truth) con un
modello di Deep Learning avanzato basato sul dataset Google GoEmotions.

--------------------------------------------------------------------------------
2. MOTIVAZIONE SCELTA MODELLO (bhadresh-savani/bert-base-go-emotion)
--------------------------------------------------------------------------------
A differenza dei classici modelli di sentiment (solo Pos/Neg), questo modello
identifica 28 diverse sfumature emotive.
STRATEGIA DI MAPPING:
Per rendere l'output confrontabile con VADER (e utile per il business), aggreghiamo
le 28 emozioni in 3 macro-categorie:
  - POSITIVE: joy, love, admiration, caring, excitement, gratitude, ecc.
  - NEGATIVE: sadness, anger, fear, disgust, remorse, nervousness, ecc.
  - NEUTRAL:  neutral, realization, curiosity, confusion, surprise.

Questa granularità sottostante permette al modello di catturare sfumature che
sfuggono ai modelli binari, specialmente in frasi ambigue, garantendo però
un output standardizzato.

--------------------------------------------------------------------------------
3. IMPLICAZIONI DEI DATI
--------------------------------------------------------------------------------
Il dataset è fortemente sbilanciato verso il Neutro (~60%).
Utilizziamo il Cohen's Kappa per verificare se l'accordo tra VADER e il modello
è reale o frutto del caso.
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
# Modello Go-Emotion (28 classi)
MODEL_NAME = "bhadresh-savani/bert-base-go-emotion"
BATCH_SIZE = 16
DEVICE = 0 if torch.cuda.is_available() else -1

# Percorsi
INPUT_HOW2SIGN = "data/processed/text_to_sentiment/how2sign_sentiment_analyzed.csv"
INPUT_ASLLRP = "data/processed/text_to_sentiment/asllrp_sentiment_analyzed.csv"
OUTPUT_DIR = "src/models/three_classes/text_to_sentiment/"

print(f"Using device: {'GPU' if DEVICE == 0 else 'CPU'}")
print(f"Caricamento modello {MODEL_NAME}...")

# Pipeline: top_k=1 prende solo l'emozione predominante
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
    Mappa le 28 classi di GoEmotions nelle 3 classi di Sentiment (VADER-compatible).
    """
    # Definiamo i gruppi
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

    # Nota: 'surprise', 'curiosity', 'confusion', 'realization' sono spesso
    # considerate cognitive/neutre in assenza di contesto positivo/negativo esplicito.
    # 'neutral' è nativo nel modello.

    if label in positive_emotions:
        return "POSITIVE"
    elif label in negative_emotions:
        return "NEGATIVE"
    else:
        return "NEUTRAL"


def load_and_normalize(file_path, id_col, text_col, gt_col, source_name):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"⚠️ File non trovato: {file_path}")

    df = pd.read_csv(file_path)
    df = df.rename(columns={id_col: "ID", text_col: "Text", gt_col: "Sentiment_GT"})
    df = df[["ID", "Text", "Sentiment_GT"]]
    df["Source_Dataset"] = source_name
    print(f" -> Caricato {source_name}: {len(df)} righe")
    return df


def run_global_analysis():
    # ---------------------------------------------------------
    # 1. CARICAMENTO E UNIONE DEI DATASET
    # ---------------------------------------------------------
    print("\n--- 1. Caricamento e Unione Dataset ---")
    try:
        df_h2s = load_and_normalize(
            INPUT_HOW2SIGN, "sentence_name", "sentence", "sentiment", "How2Sign"
        )
        df_asl = load_and_normalize(
            INPUT_ASLLRP, "video_name", "caption", "sentiment", "ASLLRP"
        )
        combined_df = pd.concat([df_h2s, df_asl], ignore_index=True)
        print(f"\n✅ DATASET UNIFICATO CREATO. Totale campioni: {len(combined_df)}")
    except Exception as e:
        print(f"❌ Errore critico nel caricamento dei dati: {e}")
        return

    # ---------------------------------------------------------
    # 2. INFERENZA SU TUTTO IL DATASET (GoEmotion -> Sentiment)
    # ---------------------------------------------------------
    print("\n--- 2. Esecuzione Inferenza GoEmotion (Global) ---")
    texts = combined_df["Text"].astype(str).tolist()

    # Liste per salvare sia l'emozione originale che il sentiment mappato
    raw_emotions = []
    mapped_sentiments = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[i : i + BATCH_SIZE]
        results = emotion_pipeline(batch_texts)

        for res in results:
            # res è una lista di dict con top_k, prendiamo il primo elemento
            # La pipeline text-classification restituisce direttamente una lista di dict se top_k=1 non è lista di liste
            # OCCHIO: Con pipeline "text-classification" e top_k=1, il formato è solitamente [{'label': 'joy', 'score': 0.9}]
            # Se fosse lista di liste lo gestiamo.

            top_prediction = res[0] if isinstance(res, list) else res
            label = top_prediction["label"]

            raw_emotions.append(label)
            mapped_sentiments.append(map_emotion_to_sentiment(label))

    combined_df["Emotion_Raw_Bert"] = raw_emotions
    combined_df["Sentiment_Pred_Bert"] = mapped_sentiments

    # Salvataggio CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_csv = os.path.join(OUTPUT_DIR, "bert_go_global_sentiment_comparison.csv")
    combined_df.to_csv(output_csv, index=False)
    print(f"✅ File confronto salvato: {output_csv}")

    # ---------------------------------------------------------
    # 3. CALCOLO METRICHE GLOBALI & AGREEMENT
    # ---------------------------------------------------------
    print("\n--- 3. Calcolo Metriche Globali & Agreement ---")

    y_true = combined_df["Sentiment_GT"].str.upper()  # VADER
    y_pred = combined_df["Sentiment_Pred_Bert"]  # BERT (Mappato)
    labels_order = ["NEGATIVE", "NEUTRAL", "POSITIVE"]

    acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    report = classification_report(y_true, y_pred, target_names=labels_order)

    metrics_text = (
        f"GLOBAL METRICS REPORT (Unified Test Set)\n"
        f"========================================\n"
        f"Model: {MODEL_NAME} (Mapped from 28 emotions)\n"
        f"Benchmark: VADER\n"
        f"Total Samples: {len(combined_df)}\n\n"
        f"KEY METRICS:\n"
        f"-----------------\n"
        f"Accuracy:                   {acc:.4f}\n"
        f"Balanced Accuracy:          {balanced_acc:.4f}\n"
        f"Cohen's Kappa:              {kappa:.4f}\n"
        f"F1 Score (Weighted):        {f1_weighted:.4f}\n\n"
        f"CONFUSION MATRIX (Rows=VADER, Cols=BERT-GoEmotion):\n"
        f"{cm}\n\n"
        f"DETAILED REPORT:\n"
        f"{report}\n"
    )

    output_txt = os.path.join(OUTPUT_DIR, "berto_go_global_metrics.txt")
    with open(output_txt, "w") as f:
        f.write(metrics_text)

    print(f"✅ Report metriche salvato: {output_txt}")
    print("-" * 30)
    print(metrics_text)

    # ---------------------------------------------------------
    # 4. ESTRAZIONE DISACCORDI
    # ---------------------------------------------------------
    disagreements = combined_df[y_true != y_pred]
    output_disagreements = os.path.join(OUTPUT_DIR, "bert_go_global_disagreements.csv")
    disagreements.to_csv(output_disagreements, index=False)
    print(f"✅ Disaccordi salvati ({len(disagreements)} righe): {output_disagreements}")


if __name__ == "__main__":
    run_global_analysis()
